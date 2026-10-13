// NFSU compute passes.
//
// The game-side mod calls these through the NFSU_* exports in d3d9_nfsu.cpp, on the game's render thread
// and in between its D3D9 calls. Each call queues one block of raw Vulkan compute dispatches on the CS
// thread using DXVK's external rendering interface. All images stay in their default layouts, so the
// blocks only need global memory barriers around and between their dispatches.
#include "d3d9_device.h"
#include "d3d9_texture.h"
#include "d3d9_initializer.h"
#include "d3d9_nfsu.h"

#include <nfsu_cluster.h>
#include <nfsu_light.h>
#include <nfsu_fog.h>
#include <nfsu_fogres.h>

namespace dxvk {

  HRESULT D3D9DeviceEx::NfsuCreateStorageTexture(UINT Width, UINT Height, IDirect3DTexture9** ppTexture) {
    D3D9DeviceLock lock = LockDevice();
    InitReturnPtr(ppTexture);
    if (!ppTexture || !Width || !Height)
      return D3DERR_INVALIDCALL;

    D3D9_COMMON_TEXTURE_DESC desc;
    desc.Width              = Width;
    desc.Height             = Height;
    desc.Depth              = 1;
    desc.ArraySize          = 1;
    desc.MipLevels          = 1;
    desc.Usage              = D3DUSAGE_RENDERTARGET;           // GPU only, no staging copies
    desc.Format             = D3D9Format::A16B16G16R16F;       // storage support is mandatory for this format
    desc.Pool               = D3DPOOL_DEFAULT;
    desc.Discard            = FALSE;
    desc.MultiSample        = D3DMULTISAMPLE_NONE;
    desc.MultisampleQuality = 0;
    desc.IsBackBuffer       = FALSE;
    desc.IsAttachmentOnly   = FALSE;
    desc.IsLockable         = FALSE;
    desc.ImageUsage         = VK_IMAGE_USAGE_STORAGE_BIT;

    HRESULT hr = D3D9CommonTexture::NormalizeTextureProperties(this, D3DRTYPE_TEXTURE, &desc);
    if (FAILED(hr))
      return hr;

    try {
      const Com<D3D9Texture2D> texture = new D3D9Texture2D(this, &desc, m_d3dCompatibility.test(D3DCompatibility::D3D9Ex), nullptr);
      m_initializer->InitTexture(texture->GetCommonTexture(), nullptr);
      *ppTexture = texture.ref();
      m_losableResourceCounter++;
      auto img = texture->GetCommonTexture()->GetImage();
      nfsu::Log("compute: storage texture %ux%u, vk format %d, layout %d, usage 0x%x", Width, Height,
        int(img->info().format), int(img->info().layout), unsigned(img->info().usage));
      return D3D_OK;
    } catch (const DxvkError& e) {
      nfsu::Log("compute: storage texture failed: %s", e.message().c_str());
      return D3DERR_OUTOFVIDEOMEMORY;
    }
  }


  static constexpr uint32_t NfsuMaxLights   = 256u;
  static constexpr uint32_t NfsuTileWords   = 65u;            // count + 64 indices, MAXL in the shaders

  static DxvkImageViewKey NfsuViewKey(const Rc<DxvkImage>& img, VkImageUsageFlags usage, VkImageAspectFlags aspect) {
    DxvkImageViewKey k;
    k.viewType   = VK_IMAGE_VIEW_TYPE_2D;
    k.format     = img->info().format;
    k.usage      = usage;
    k.layout     = img->info().layout;
    k.aspects    = aspect;
    k.mipIndex   = 0; k.mipCount   = 1;
    k.layerIndex = 0; k.layerCount = 1;
    return k;
  }

  // Clustered lighting. The screen is split into 16x16 pixel tiles. The first pass builds a light list
  // per tile from the scene depth and computes the lamp haze of each tile; the second pass accumulates
  // lamp light per pixel from those lists.
  //
  // pDepth   INTZ scene depth
  // pHeat    optional storage texture, half the screen size: lights per tile heat map
  // pLight   optional storage texture: rgb diffuse sum, a specular sum
  // pHaze    optional storage texture, two texels per tile: haze at the nearest and farthest depth
  // pAtlas   optional L16 atlas of baked lamp shadow maps, four tiles per row
  // pLights  16 floats per light in view space: position.xyz range | colour.rgb falloff |
  //          direction.xyz cosOuter | cosInner noHaze - atlasTile
  // pParams  32 floats:
  //    [0]      unused
  //    [1]      time (s)
  //    [2]      heat map on
  //    [3]      haze on
  //    [4..7]   projection P00 P11 P22 P32
  //    [8..14]  filled in here
  //    [15]     ray length for sky pixels (m)
  //    [16]     specular exponent
  //    [17]     specular on
  //    [18]     atlas depth bias (m), 0 turns atlas shadows off
  //    [19]     dual output: pLight is twice as wide and its right half receives the diffuse light of
  //             moving lights only (falloff < 0)
  //    [20..22] world up in view space
  //    [23]     atlas far - near (m)
  //    [24..26] world X in view space
  //    [27]     atlas 1 / tan(fov / 2)
  //    [28..30] world Y in view space
  //    [31]     atlas near (m)
  HRESULT D3D9DeviceEx::NfsuCluster(
          IDirect3DBaseTexture9*  pDepth,
          IDirect3DBaseTexture9*  pHeat,
          IDirect3DBaseTexture9*  pLight,
          IDirect3DBaseTexture9*  pHaze,
          IDirect3DBaseTexture9*  pAtlas,
    const float*                  pParams,
    const float*                  pLights,
          UINT                    LightCount) {
    D3D9DeviceLock lock = LockDevice();
    auto storageTex = [](IDirect3DBaseTexture9* t) -> D3D9CommonTexture* {
      D3D9CommonTexture* c = t ? GetCommonTexture(t) : nullptr;
      return (c && (c->Desc()->ImageUsage & VK_IMAGE_USAGE_STORAGE_BIT)) ? c : nullptr;
    };
    D3D9CommonTexture* heat  = storageTex(pHeat);
    D3D9CommonTexture* light = storageTex(pLight);
    D3D9CommonTexture* haze  = storageTex(pHaze);
    D3D9CommonTexture* atlas = pAtlas ? GetCommonTexture(pAtlas) : nullptr;
    D3D9CommonTexture* depth = pDepth ? GetCommonTexture(pDepth) : nullptr;
    if (!pParams || !depth || (!heat && !light && !haze))
      return D3DERR_INVALIDCALL;

    if (!m_nfsuTileLayout) {
      static const std::array<DxvkDescriptorSetLayoutBinding, 5> tb = {{
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
      }};
      m_nfsuTileLayout = m_dxvkDevice->createBuiltInPipelineLayout(0u, VK_SHADER_STAGE_COMPUTE_BIT, 16 * sizeof(float), tb.size(), tb.data());
      util::DxvkBuiltInShaderStage st;
      st.size = sizeof(nfsu_cluster); st.code = nfsu_cluster; st.spec = nullptr;
      m_nfsuTilePipe = m_dxvkDevice->createBuiltInComputePipeline(m_nfsuTileLayout, st);

      static const std::array<DxvkDescriptorSetLayoutBinding, 5> lb = {{
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },   // shadow map atlas
      }};
      m_nfsuLightLayout = m_dxvkDevice->createBuiltInPipelineLayout(0u, VK_SHADER_STAGE_COMPUTE_BIT, 28 * sizeof(float), lb.size(), lb.data());
      st.size = sizeof(nfsu_light); st.code = nfsu_light;
      m_nfsuLightPipe = m_dxvkDevice->createBuiltInComputePipeline(m_nfsuLightLayout, st);
      nfsu::Log("compute: cluster pipeline %s, light pipeline %s", m_nfsuTilePipe ? "created" : "FAILED", m_nfsuLightPipe ? "created" : "FAILED");
    }
    if (!m_nfsuTilePipe || !m_nfsuLightPipe)
      return D3DERR_INVALIDCALL;

    Rc<DxvkImage> heatImage  = heat  ? heat->GetImage()  : nullptr;
    Rc<DxvkImage> lightImage = light ? light->GetImage() : nullptr;
    Rc<DxvkImage> hazeImage  = haze  ? haze->GetImage()  : nullptr;
    Rc<DxvkImage> depthImage = depth->GetImage();
    uint32_t sw = depthImage->info().extent.width;
    uint32_t sh = depthImage->info().extent.height;
    uint32_t tx = (sw + 15u) / 16u, ty = (sh + 15u) / 16u;

    if (m_nfsuLights == nullptr) {
      DxvkBufferCreateInfo bi;
      bi.size   = NfsuMaxLights * 16u * sizeof(float);
      bi.usage  = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      bi.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      bi.access = VK_ACCESS_SHADER_READ_BIT;
      bi.debugName = "NFSU lights";
      m_nfsuLights = m_dxvkDevice->createBuffer(bi, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    if (m_nfsuTiles == nullptr || m_nfsuTileCount < tx * ty) {
      DxvkBufferCreateInfo bi;
      bi.size   = VkDeviceSize(tx * ty) * NfsuTileWords * sizeof(uint32_t);
      bi.usage  = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      bi.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      bi.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      bi.debugName = "NFSU tiles";
      m_nfsuTiles = m_dxvkDevice->createBuffer(bi, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      m_nfsuTileCount = tx * ty;
      nfsu::Log("compute: tile lists %ux%u tiles (%u KB)", tx, ty, unsigned(bi.size / 1024u));
    }
    uint32_t nl = std::min<uint32_t>(pLights ? LightCount : 0u, NfsuMaxLights);
    Rc<DxvkResourceAllocation> lightSlice = m_nfsuLights->allocateStorage();
    if (nl)
      std::memcpy(lightSlice->mapPtr(), pLights, nl * 16u * sizeof(float));

    // dual output: the light image is twice the logical width, its right half receives moving lights only
    const bool dual = pParams[19] > 0.5f && lightImage != nullptr && (lightImage->info().extent.width % 2u) == 0u;
    if (!m_nfsuLogged) {
      m_nfsuLogged = true;
      nfsu::Log("compute: first cluster dispatch (light output %s): heat %s, light %s %ux%u, haze %s %ux%u, depth fmt %d layout %d, %u lights, %ux%u tiles",
        dual ? "dual: moving lights in the right half" : "single", heatImage != nullptr ? "yes" : "no",
        lightImage != nullptr ? "yes" : "no", lightImage != nullptr ? lightImage->info().extent.width : 0u, lightImage != nullptr ? lightImage->info().extent.height : 0u,
        hazeImage != nullptr ? "yes" : "no", hazeImage != nullptr ? hazeImage->info().extent.width : 0u, hazeImage != nullptr ? hazeImage->info().extent.height : 0u,
        int(depthImage->info().format), int(depthImage->info().layout), nl, tx, ty);
    }

    std::array<float, 16> tpc;                                  // tile pass
    for (uint32_t i = 0; i < 16; i++) tpc[i] = pParams[i];
    tpc[2]  = heatImage != nullptr ? pParams[2] : 0.0f;
    tpc[3]  = hazeImage != nullptr && pParams[3] > 0.5f ? 1.0f : 0.0f;
    tpc[8]  = heatImage != nullptr ? float(heatImage->info().extent.width)  : 0.0f;
    tpc[9]  = heatImage != nullptr ? float(heatImage->info().extent.height) : 0.0f;
    tpc[10] = float(sw); tpc[11] = float(sh);
    tpc[12] = float(nl); tpc[13] = float(tx); tpc[14] = float(ty);
    std::array<float, 28> lpc = { };                             // light pass
    lpc[0] = pParams[16]; lpc[1] = pParams[17]; lpc[2] = dual ? 1.0f : 0.0f;
    for (uint32_t i = 4; i < 8; i++) lpc[i] = pParams[i];
    lpc[8]  = lightImage != nullptr ? float(lightImage->info().extent.width / (dual ? 2u : 1u)) : 0.0f;
    lpc[9]  = lightImage != nullptr ? float(lightImage->info().extent.height) : 0.0f;
    lpc[10] = float(sw); lpc[11] = float(sh);
    lpc[12] = float(nl); lpc[13] = float(tx);
    lpc[16] = pParams[20]; lpc[17] = pParams[21]; lpc[18] = pParams[22];
    // atlas: four tiles per row, tile size = atlas width / 4
    Rc<DxvkImage> atlasImage = atlas ? atlas->GetImage() : nullptr;
    const bool useAtlas = atlasImage != nullptr && pParams[18] > 0.0f;
    lpc[3]  = useAtlas ? pParams[18] : 0.0f;
    lpc[14] = useAtlas ? float(atlasImage->info().extent.width / 4u) : 0.0f; lpc[15] = 4.0f;
    lpc[19] = pParams[23];
    for (uint32_t i = 0; i < 4; i++) { lpc[20 + i] = pParams[24 + i]; lpc[24 + i] = pParams[28 + i]; }

    EmitCs([
      cHeat = std::move(heatImage), cLight = std::move(lightImage), cHaze = std::move(hazeImage), cDepth = std::move(depthImage),
      cLights = m_nfsuLights, cSlice = std::move(lightSlice), cTiles = m_nfsuTiles,
      cTLayout = m_nfsuTileLayout, cTPipe = m_nfsuTilePipe, cLLayout = m_nfsuLightLayout, cLPipe = m_nfsuLightPipe,
      cTpc = tpc, cLpc = lpc, cGroups = VkExtent2D { tx, ty }, cLw = uint32_t(lpc[8]), cAtlas = std::move(atlasImage)
    ] (DxvkContext* ctx) mutable {
      ctx->invalidateBuffer(cLights, std::move(cSlice));        // this frame's light list
      Rc<DxvkCommandList> cmd = ctx->beginExternalRendering();

      Rc<DxvkImageView> heatView  = cHeat  != nullptr ? cHeat->createView(NfsuViewKey(cHeat,  VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_ASPECT_COLOR_BIT)) : nullptr;
      Rc<DxvkImageView> lightView = cLight != nullptr ? cLight->createView(NfsuViewKey(cLight, VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_ASPECT_COLOR_BIT)) : nullptr;
      Rc<DxvkImageView> hazeView  = cHaze  != nullptr ? cHaze->createView(NfsuViewKey(cHaze,  VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_ASPECT_COLOR_BIT)) : nullptr;
      Rc<DxvkImageView> depthView = cDepth->createView(NfsuViewKey(cDepth, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT));

      VkPipelineStageFlags2 imgStages = 0; VkAccessFlags2 imgAccess = 0;
      for (const Rc<DxvkImage>* i : { &cHeat, &cLight, &cHaze, &cDepth })
        if (*i != nullptr) { imgStages |= (*i)->info().stages; imgAccess |= (*i)->info().access; }

      auto barrier = [&](VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds, VkAccessFlags2 da) {
        VkMemoryBarrier2 b = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        b.srcStageMask = ss; b.srcAccessMask = sa; b.dstStageMask = ds; b.dstAccessMask = da;
        VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.memoryBarrierCount = 1u; dep.pMemoryBarriers = &b;
        cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep);
      };
      const VkPipelineStageFlags2 CS = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      const VkAccessFlags2 RW = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;

      // before: depth just written as an attachment, our images sampled last frame, lights written by the host
      barrier(imgStages | CS | VK_PIPELINE_STAGE_2_HOST_BIT, imgAccess | RW | VK_ACCESS_2_HOST_WRITE_BIT, CS, RW);

      // pass 1: tile lists + per-tile haze (+ heat map)
      std::array<DxvkDescriptorWrite, 5> td = { };
      td[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;  td[0].descriptor = heatView  != nullptr ? heatView->getDescriptor()  : nullptr;
      td[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;  td[1].descriptor = depthView->getDescriptor();
      td[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; td[2].buffer = cLights->getSliceInfo();
      td[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; td[3].buffer = cTiles->getSliceInfo();
      td[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;  td[4].descriptor = hazeView  != nullptr ? hazeView->getDescriptor()  : nullptr;
      cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, cTPipe);
      cmd->bindResources(DxvkCmdBuffer::ExecBuffer, cTLayout, td.size(), td.data(), sizeof(cTpc), cTpc.data());
      cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer, cGroups.width, cGroups.height, 1u);

      // pass 2: lamp light from the lists (needs the tile buffer written above)
      if (lightView != nullptr) {
        barrier(CS, VK_ACCESS_2_SHADER_WRITE_BIT, CS, VK_ACCESS_2_SHADER_READ_BIT);
        std::array<DxvkDescriptorWrite, 5> ld = { };
        Rc<DxvkImageView> atlasView = cAtlas != nullptr ? cAtlas->createView(NfsuViewKey(cAtlas, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT))
                                                        : depthView;     // no atlas: any valid image, never read
        ld[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;  ld[0].descriptor = lightView->getDescriptor();
        ld[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;  ld[1].descriptor = depthView->getDescriptor();
        ld[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; ld[2].buffer = cLights->getSliceInfo();
        ld[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; ld[3].buffer = cTiles->getSliceInfo();
        ld[4].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;  ld[4].descriptor = atlasView->getDescriptor();
        cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, cLPipe);
        cmd->bindResources(DxvkCmdBuffer::ExecBuffer, cLLayout, ld.size(), ld.data(), sizeof(cLpc), cLpc.data());
        cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer, (cLw + 7u) / 8u, (cLight->info().extent.height + 7u) / 8u, 1u);
      }

      // after: our images are sampled by the game's passes, depth returns to attachment use
      barrier(CS, RW, imgStages | CS, imgAccess | VK_ACCESS_2_SHADER_READ_BIT);

      for (const Rc<DxvkImage>* i : { &cHeat, &cLight, &cHaze })
        if (*i != nullptr) cmd->track(*i, DxvkAccess::Write);
      cmd->track(cDepth, DxvkAccess::Read);
      cmd->track(cLights, DxvkAccess::Read);
      cmd->track(cTiles, DxvkAccess::Write);
      if (cAtlas != nullptr) cmd->track(cAtlas, DxvkAccess::Read);
    });
    return D3D_OK;
  }


  // Froxel fog. Every 16x16 pixel tile gets a column of 64 exponential depth slices. The first pass culls
  // lights per tile, fills each froxel with fog density and in-scattered light, blends it with the
  // reprojected result of the previous frame and integrates the column front to back. The second pass
  // resolves the volume into pOut at each pixel's depth: rgb glow, a transmittance.
  //
  // pDepth       INTZ scene depth
  // pOut         storage texture receiving the resolved fog
  // pShadowMaps  12 INTZ shadow maps, static and dynamic geometry for each of six lamps, may be null
  // pAtlas       optional L16 atlas of baked shadow maps for up to 16 more lamps
  // pLights      as in NfsuCluster, with extra = cosInner noFog fogGain shadowSlot
  // pParams      240 floats, as vec4 P[60] in nfsu_fog.comp:
  //    P[0]       projection P00 P11 P22 P32
  //    P[1..3]    inverse view matrix rows
  //    P[4]       camera position
  //    P[5]       fog density, height (m), max distance (m), time (s)
  //    P[6]       noise scale, drift, ground offset, strength
  //    P[7]       falloff, shadow map 0 on, shadow map 1 on
  //    P[8..15]   view projection rows of shadow maps 0 and 1
  //    P[16]      shadow map P22, P32, bias, texel size
  //    P[17]      weight of the new frame in the history blend
  //    P[18..21]  world to view matrix rows of the previous frame
  //    P[22]      projection of the previous frame
  //    P[23]      near slice distance (m), point light fog weight, previous frame valid
  //    P[24..39]  view projection rows of shadow maps 2 to 5
  //    P[40]      shadow maps 2 to 5 on
  //    P[42..57]  atlas tiles: lamp position (world), weight
  //    P[58]      atlas 1 / tan(fov / 2), near, far - near, tiles per row
  //    P[59]      atlas tile size (px)
  HRESULT D3D9DeviceEx::NfsuFog(
          IDirect3DBaseTexture9*  pDepth,
          IDirect3DBaseTexture9*  pOut,
          IDirect3DBaseTexture9* const* pLs,
          IDirect3DBaseTexture9*  pAtlas,
    const float*                  pParams,
    const float*                  pLights,
          UINT                    LightCount) {
    D3D9DeviceLock lock = LockDevice();
    D3D9CommonTexture* out   = pOut ? GetCommonTexture(pOut) : nullptr;
    D3D9CommonTexture* atlas = pAtlas ? GetCommonTexture(pAtlas) : nullptr;
    D3D9CommonTexture* depth = pDepth ? GetCommonTexture(pDepth) : nullptr;
    if (!pParams || !pLs || !depth || !out || !(out->Desc()->ImageUsage & VK_IMAGE_USAGE_STORAGE_BIT))
      return D3DERR_INVALIDCALL;
    constexpr uint32_t Slices = 64u;
    constexpr uint32_t LsCount = 12u;
    constexpr uint32_t ParamCount = 240u;

    if (!m_nfsuFogLayout) {
      static const std::array<DxvkDescriptorSetLayoutBinding, 18> fb = {{
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },   // shadow map atlas
      }};
      m_nfsuFogLayout = m_dxvkDevice->createBuiltInPipelineLayout(0u, VK_SHADER_STAGE_COMPUTE_BIT, 12 * sizeof(float), fb.size(), fb.data());
      util::DxvkBuiltInShaderStage st;
      st.size = sizeof(nfsu_fog); st.code = nfsu_fog; st.spec = nullptr;
      m_nfsuFogPipe = m_dxvkDevice->createBuiltInComputePipeline(m_nfsuFogLayout, st);
      static const std::array<DxvkDescriptorSetLayoutBinding, 3> rb = {{
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT },
      }};
      m_nfsuFogResLayout = m_dxvkDevice->createBuiltInPipelineLayout(0u, VK_SHADER_STAGE_COMPUTE_BIT, 16 * sizeof(float), rb.size(), rb.data());
      st.size = sizeof(nfsu_fogres); st.code = nfsu_fogres;
      m_nfsuFogResPipe = m_dxvkDevice->createBuiltInComputePipeline(m_nfsuFogResLayout, st);
      nfsu::Log("compute: fog pipeline %s, fog resolve pipeline %s", m_nfsuFogPipe ? "created" : "FAILED", m_nfsuFogResPipe ? "created" : "FAILED");
    }
    if (!m_nfsuFogPipe || !m_nfsuFogResPipe)
      return D3DERR_INVALIDCALL;

    Rc<DxvkImage> depthImage = depth->GetImage();
    Rc<DxvkImage> outImage   = out->GetImage();
    uint32_t sw = depthImage->info().extent.width, sh = depthImage->info().extent.height;
    uint32_t tx = (sw + 15u) / 16u, ty = (sh + 15u) / 16u;
    uint32_t ow = outImage->info().extent.width, oh = outImage->info().extent.height;

    bool volInit = false;
    if (m_nfsuFogVol == nullptr || m_nfsuFogVol->info().extent.width != tx || m_nfsuFogVol->info().extent.height != ty) {
      DxvkImageCreateInfo ii;
      ii.type        = VK_IMAGE_TYPE_3D;
      ii.format      = VK_FORMAT_R16G16B16A16_SFLOAT;
      ii.sampleCount = VK_SAMPLE_COUNT_1_BIT;
      ii.extent      = { tx, ty, Slices };
      ii.numLayers   = 1u;
      ii.mipLevels   = 1u;
      ii.usage       = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;   // cleared by initImage
      ii.stages      = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
      ii.access      = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      ii.tiling      = VK_IMAGE_TILING_OPTIMAL;
      ii.layout      = VK_IMAGE_LAYOUT_GENERAL;
      ii.debugName   = "NFSU fog volume";
      m_nfsuFogVol = m_dxvkDevice->createImage(ii, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      ii.debugName = "NFSU fog history";
      for (int i = 0; i < 2; i++) m_nfsuFogInj[i] = m_dxvkDevice->createImage(ii, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      DxvkImageViewKey k;
      k.viewType = VK_IMAGE_VIEW_TYPE_3D;
      k.format   = ii.format;
      k.usage    = VK_IMAGE_USAGE_STORAGE_BIT;
      k.layout   = VK_IMAGE_LAYOUT_GENERAL;
      k.aspects  = VK_IMAGE_ASPECT_COLOR_BIT;
      k.mipIndex = 0; k.mipCount = 1; k.layerIndex = 0; k.layerCount = 1;
      m_nfsuFogVolView = m_nfsuFogVol->createView(k);
      for (int i = 0; i < 2; i++) m_nfsuFogInjView[i] = m_nfsuFogInj[i]->createView(k);
      m_nfsuFogFrame = 0;
      volInit = true;
      nfsu::Log("compute: fog volume %ux%ux%u", tx, ty, Slices);
    }
    for (Rc<DxvkBuffer>* b : { &m_nfsuFogLights, &m_nfsuFogParams }) {
      if (*b != nullptr) continue;
      DxvkBufferCreateInfo bi;
      bi.size   = (b == &m_nfsuFogLights ? NfsuMaxLights * 16u : ParamCount) * sizeof(float);
      bi.usage  = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      bi.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      bi.access = VK_ACCESS_SHADER_READ_BIT;
      bi.debugName = b == &m_nfsuFogLights ? "NFSU fog lights" : "NFSU fog params";
      *b = m_dxvkDevice->createBuffer(bi, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    uint32_t nl = std::min<uint32_t>(pLights ? LightCount : 0u, NfsuMaxLights);
    Rc<DxvkResourceAllocation> lightSlice = m_nfsuFogLights->allocateStorage();
    if (nl)
      std::memcpy(lightSlice->mapPtr(), pLights, nl * 16u * sizeof(float));
    Rc<DxvkResourceAllocation> parSlice = m_nfsuFogParams->allocateStorage();
    std::memcpy(parSlice->mapPtr(), pParams, ParamCount * sizeof(float));

    // shadow maps: a missing map is replaced by the scene depth, its flag in the parameters is 0 so it is never read
    std::array<Rc<DxvkImage>, LsCount> ls;
    for (uint32_t i = 0; i < LsCount; i++) {
      D3D9CommonTexture* c = pLs[i] ? GetCommonTexture(pLs[i]) : nullptr;
      ls[i] = c ? c->GetImage() : depthImage;
    }
    Rc<DxvkImage> atlasImage = atlas ? atlas->GetImage() : nullptr;

    // temporal: R2 sequence inside the tile (px) and slice (0..1); history only when the images are not new
    // and the caller says last frame's matrices are valid
    uint32_t fi = m_nfsuFogFrame++;
    float jx = std::fmod(0.5f + 0.7548777f * float(fi), 1.0f), jy = std::fmod(0.5f + 0.5698403f * float(fi), 1.0f);
    float jz = std::fmod(0.5f + 0.6180340f * float(fi), 1.0f);
    bool hist = !volInit && fi > 0u && pParams[94] > 0.5f;
    std::array<float, 12> fpc = { float(nl), float(tx), float(ty), float(Slices), float(sw), float(sh), pParams[92], pParams[93],
                                  (jx - 0.5f) * 16.0f, (jy - 0.5f) * 16.0f, jz, hist ? 1.0f : 0.0f };
    Rc<DxvkImage> injCur = m_nfsuFogInj[fi & 1u], injPrev = m_nfsuFogInj[(fi + 1u) & 1u];
    Rc<DxvkImageView> injCurView = m_nfsuFogInjView[fi & 1u], injPrevView = m_nfsuFogInjView[(fi + 1u) & 1u];
    std::array<float, 16> rpc = { pParams[0], pParams[1], pParams[2], pParams[3],
                                  float(ow), float(oh), float(sw), float(sh),
                                  pParams[92], pParams[22], float(Slices), pParams[27],
                                  float(tx), float(ty), 0.0f, 0.0f };
    if (!m_nfsuFogLogged) {
      m_nfsuFogLogged = true;
      int nm = 0;
      for (uint32_t i = 0; i < LsCount; i += 2u) nm += pLs[i] != nullptr;
      nfsu::Log("compute: first fog dispatch: %u lights, %ux%u tiles x %u slices, out %ux%u, near %.2f max %.1f m, %d of %u shadow maps",
        nl, tx, ty, Slices, ow, oh, pParams[92], pParams[22], nm, LsCount / 2u);
    }

    EmitCs([
      cDepth = std::move(depthImage), cOut = std::move(outImage), cLs = ls, cAtlas = std::move(atlasImage),
      cVol = m_nfsuFogVol, cVolView = m_nfsuFogVolView, cInj = std::move(injCur), cInjView = std::move(injCurView),
      cPrev = std::move(injPrev), cPrevView = std::move(injPrevView),
      cLights = m_nfsuFogLights, cLSlice = std::move(lightSlice), cPar = m_nfsuFogParams, cPSlice = std::move(parSlice),
      cFLayout = m_nfsuFogLayout, cFPipe = m_nfsuFogPipe, cRLayout = m_nfsuFogResLayout, cRPipe = m_nfsuFogResPipe,
      cFpc = fpc, cRpc = rpc, cTiles = VkExtent2D { tx, ty }, cOutSize = VkExtent2D { ow, oh }, cVolInit = volInit
    ] (DxvkContext* ctx) mutable {
      if (cVolInit) {
        ctx->initImage(cVol, VK_IMAGE_LAYOUT_UNDEFINED);
        ctx->initImage(cInj, VK_IMAGE_LAYOUT_UNDEFINED);
        ctx->initImage(cPrev, VK_IMAGE_LAYOUT_UNDEFINED);
      }
      ctx->invalidateBuffer(cLights, std::move(cLSlice));
      ctx->invalidateBuffer(cPar, std::move(cPSlice));
      Rc<DxvkCommandList> cmd = ctx->beginExternalRendering();

      Rc<DxvkImageView> outView   = cOut->createView(NfsuViewKey(cOut, VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_ASPECT_COLOR_BIT));
      Rc<DxvkImageView> depthView = cDepth->createView(NfsuViewKey(cDepth, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT));
      std::array<Rc<DxvkImageView>, 12> lsView;
      for (uint32_t i = 0; i < lsView.size(); i++)
        lsView[i] = cLs[i]->createView(NfsuViewKey(cLs[i], VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT));
      Rc<DxvkImageView> atlasView = cAtlas != nullptr ? cAtlas->createView(NfsuViewKey(cAtlas, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT))
                                                      : lsView[0];   // no atlas: any valid image, never read

      VkPipelineStageFlags2 imgStages = 0; VkAccessFlags2 imgAccess = 0;
      for (const Rc<DxvkImage>* i : { &cDepth, &cOut })
        { imgStages |= (*i)->info().stages; imgAccess |= (*i)->info().access; }
      for (const Rc<DxvkImage>& i : cLs)
        { imgStages |= i->info().stages; imgAccess |= i->info().access; }
      if (cAtlas != nullptr) { imgStages |= cAtlas->info().stages; imgAccess |= cAtlas->info().access; }
      auto barrier = [&](VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds, VkAccessFlags2 da) {
        VkMemoryBarrier2 b = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        b.srcStageMask = ss; b.srcAccessMask = sa; b.dstStageMask = ds; b.dstAccessMask = da;
        VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.memoryBarrierCount = 1u; dep.pMemoryBarriers = &b;
        cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep);
      };
      const VkPipelineStageFlags2 CS = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      const VkAccessFlags2 RW = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;

      // before: depth + shadow maps just written as attachments, out sampled last frame, buffers written by
      // the host, the volume read by last frame's resolve
      barrier(imgStages | CS | VK_PIPELINE_STAGE_2_HOST_BIT, imgAccess | RW | VK_ACCESS_2_HOST_WRITE_BIT, CS, RW);

      std::array<DxvkDescriptorWrite, 18> fd = { };
      fd[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; fd[0].buffer = cLights->getSliceInfo();
      fd[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; fd[1].buffer = cPar->getSliceInfo();
      fd[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;  fd[2].descriptor = cVolView->getDescriptor();
      for (int i = 0; i < 4; i++) { fd[3 + i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; fd[3 + i].descriptor = lsView[i]->getDescriptor(); }
      fd[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;  fd[7].descriptor = cInjView->getDescriptor();
      fd[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;  fd[8].descriptor = cPrevView->getDescriptor();
      for (int i = 0; i < 8; i++) { fd[9 + i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; fd[9 + i].descriptor = lsView[4 + i]->getDescriptor(); }
      fd[17].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; fd[17].descriptor = atlasView->getDescriptor();
      cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, cFPipe);
      cmd->bindResources(DxvkCmdBuffer::ExecBuffer, cFLayout, fd.size(), fd.data(), sizeof(cFpc), cFpc.data());
      cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer, cTiles.width, cTiles.height, 1u);

      barrier(CS, VK_ACCESS_2_SHADER_WRITE_BIT, CS, VK_ACCESS_2_SHADER_READ_BIT);

      std::array<DxvkDescriptorWrite, 3> rd = { };
      rd[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;  rd[0].descriptor = outView->getDescriptor();
      rd[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;  rd[1].descriptor = depthView->getDescriptor();
      rd[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;  rd[2].descriptor = cVolView->getDescriptor();
      cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, cRPipe);
      cmd->bindResources(DxvkCmdBuffer::ExecBuffer, cRLayout, rd.size(), rd.data(), sizeof(cRpc), cRpc.data());
      cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer, (cOutSize.width + 7u) / 8u, (cOutSize.height + 7u) / 8u, 1u);

      // after: out is sampled by the game's passes, depth + shadow maps return to attachment use
      barrier(CS, RW, imgStages | CS, imgAccess | VK_ACCESS_2_SHADER_READ_BIT);

      cmd->track(cOut, DxvkAccess::Write);
      cmd->track(cVol, DxvkAccess::Write);
      cmd->track(cInj, DxvkAccess::Write);
      cmd->track(cPrev, DxvkAccess::Read);
      cmd->track(cDepth, DxvkAccess::Read);
      for (const Rc<DxvkImage>& i : cLs)
        cmd->track(i, DxvkAccess::Read);
      if (cAtlas != nullptr) cmd->track(cAtlas, DxvkAccess::Read);
      cmd->track(cLights, DxvkAccess::Read);
      cmd->track(cPar, DxvkAccess::Read);
    });
    return D3D_OK;
  }


  void D3D9DeviceEx::NfsuDestroy() {
    auto vk = m_dxvkDevice->vkd();

    for (VkPipeline pipeline : { m_nfsuTilePipe, m_nfsuLightPipe, m_nfsuFogPipe, m_nfsuFogResPipe })
      vk->vkDestroyPipeline(vk->device(), pipeline, nullptr);

    // Destroys the FSR pipelines, no CS chunk refers to the state any more
    m_nfsuFsr = nullptr;
  }

}
