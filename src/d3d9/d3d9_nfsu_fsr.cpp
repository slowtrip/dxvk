// NFSU: AMD FSR 3.1 upscaler ("Native AA" at 1.0x, upscaling when the output is larger than the inputs) as our own
// Vulkan compute passes inside DXVK.
// Shaders: the FidelityFX SDK v1.1.4 GLSL passes (MIT), built by nfsu/fsr/gen_fsr.py into nfsu/fsr/nfsu_fsr_spv.h.
// Host side: a port of the dispatch logic of the SDK's ffx_fsr3upscaler.cpp (constants, internal resources, pass order,
// ping-pong, clears) - NOT the SDK's Vulkan backend, which stores Vulkan handles in void* (truncated on 32-bit) and
// would own memory, descriptors and layouts behind DXVK's back. Called by the game-side mod through NFSU_Fsr
// (d3d9_nfsu.cpp) on the game's render thread, on the HDR scene before post processing.
#include "d3d9_device.h"
#include "d3d9_texture.h"
#include "d3d9_nfsu.h"
#include "nfsu/fsr/nfsu_fsr_spv.h"

#include <cmath>

namespace dxvk {

  namespace {

    enum FsrPass : uint32_t {
      FsrPrepareInputs, FsrLumaPyramid, FsrShadingChangePyramid, FsrShadingChange, FsrPrepareReactivity,
      FsrLumaInstability, FsrAccumulate, FsrAccumulateSharpen, FsrRcas, FsrDebugView, FsrPassCount
    };

    // internal images (render size unless "display"; "half" = the SDK's maxRenderSize / 2). Display size (SDK
    // maxUpscaleSize): new locks and the two upscaled-colour histories only (ffx_fsr3upscaler.cpp internalSurfaceDesc)
    enum FsrImage : uint32_t {
      FiDilatedMv, FiDilatedDepth, FiRecPrevDepth, FiIntermediate, FiLuma0, FiLuma1, FiSpdAtomic, FiFrameInfo, FiSpdMips,
      FiFarthestMip1, FiShadingChange, FiDilatedReactive, FiNewLocks, FiAcc0, FiAcc1, FiLumaHist0, FiLumaHist1,
      FiUpscaled0, FiUpscaled1, FiDefaultExposure, FiDefaultReactive, FiCount
    };

    struct FsrImageDesc { VkFormat format; bool half; bool tiny; bool mips; const char* name; bool display = false; };
    const FsrImageDesc g_fsrImages[FiCount] = {
      { VK_FORMAT_R16G16_SFLOAT,       false, false, false, "FSR dilated MV" },
      { VK_FORMAT_R32_SFLOAT,          false, false, false, "FSR dilated depth" },
      { VK_FORMAT_R32_UINT,            false, false, false, "FSR reconstructed prev depth" },
      { VK_FORMAT_R16_SFLOAT,          false, false, false, "FSR intermediate fp16 (farthest depth / luma instability)" },
      { VK_FORMAT_R16_SFLOAT,          false, false, false, "FSR luma 1" },
      { VK_FORMAT_R16_SFLOAT,          false, false, false, "FSR luma 2" },
      { VK_FORMAT_R32_UINT,            false, true,  false, "FSR SPD atomic" },
      { VK_FORMAT_R32G32B32A32_SFLOAT, false, true,  false, "FSR frame info" },
      { VK_FORMAT_R16G16_SFLOAT,       true,  false, true,  "FSR SPD mips" },
      { VK_FORMAT_R16_SFLOAT,          true,  false, false, "FSR farthest depth mip1" },
      { VK_FORMAT_R8_UNORM,            true,  false, false, "FSR shading change" },
      { VK_FORMAT_R8G8B8A8_UNORM,      false, false, false, "FSR dilated reactive masks" },
      { VK_FORMAT_R8_UNORM,            false, false, false, "FSR new locks", true },
      { VK_FORMAT_R8_UNORM,            false, false, false, "FSR accumulation 1" },
      { VK_FORMAT_R8_UNORM,            false, false, false, "FSR accumulation 2" },
      { VK_FORMAT_R16G16B16A16_SFLOAT, false, false, false, "FSR luma history 1" },
      { VK_FORMAT_R16G16B16A16_SFLOAT, false, false, false, "FSR luma history 2" },
      { VK_FORMAT_R16G16B16A16_SFLOAT, false, false, false, "FSR internal upscaled 1", true },
      { VK_FORMAT_R16G16B16A16_SFLOAT, false, false, false, "FSR internal upscaled 2", true },
      { VK_FORMAT_R32G32_SFLOAT,       false, true,  false, "FSR default exposure (0 = 1.0)" },
      { VK_FORMAT_R8_UNORM,            false, true,  false, "FSR default reactive (0)" },
    };

    // SDK Fsr3UpscalerConstants (std140 cbFSR3Upscaler)
    struct FsrConstants {
      int32_t renderSize[2], previousFrameRenderSize[2], upscaleSize[2], previousFrameUpscaleSize[2];
      int32_t maxRenderSize[2], maxUpscaleSize[2];
      float   deviceToViewDepth[4];
      float   jitterOffset[2], previousFrameJitterOffset[2];
      float   motionVectorScale[2], downscaleFactor[2];
      float   motionVectorJitterCancellation[2];
      float   tanHalfFOV, jitterPhaseCount;
      float   deltaTime, deltaPreExposure, viewSpaceToMetersFactor, frameIndex;
      float   velocityFactor, reactivenessScale, shadingChangeScale, accumulationAddedPerFrame, minDisocclusionAccumulation;
      float   pad[3];
    };
    static_assert(sizeof(FsrConstants) == 160, "cbFSR3Upscaler layout (148 bytes std140 + pad)");
    struct FsrSpdConstants  { uint32_t mips, numWorkGroups, workGroupOffset[2], renderSize[2], pad[2]; };
    struct FsrRcasConstants { uint32_t rcasConfig[4]; };

    uint16_t FsrHalf(float f) {                                   // float -> IEEE half (RCAS packHalf2x16 constant)
      uint32_t x; std::memcpy(&x, &f, 4);
      uint32_t s = (x >> 16) & 0x8000u, e = (x >> 23) & 0xffu, m = x & 0x7fffffu;
      int32_t he = int32_t(e) - 127 + 15;
      if (e == 0xffu) return uint16_t(s | 0x7c00u | (m ? 0x200u : 0u));
      if (he >= 31) return uint16_t(s | 0x7c00u);
      if (he <= 0) {
        if (he < -10) return uint16_t(s);
        m |= 0x800000u;
        uint32_t sh = uint32_t(14 - he);
        uint32_t r = (m >> sh) + ((m >> (sh - 1)) & 1u);
        return uint16_t(s | r);
      }
      uint32_t r = (uint32_t(he) << 10) | (m >> 13);
      r += (m >> 12) & 1u;                                         // round half up (carry into the exponent is correct)
      return uint16_t(s | r);
    }

    DxvkImageViewKey FsrViewKey(const Rc<DxvkImage>& img, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                                uint32_t mip = 0u, uint32_t mipCount = 1u) {
      DxvkImageViewKey k;
      k.viewType   = VK_IMAGE_VIEW_TYPE_2D;
      k.format     = img->info().format;
      k.usage      = usage;
      k.layout     = img->info().layout;
      k.aspects    = aspect;
      k.mipIndex   = mip; k.mipCount = mipCount;
      k.layerIndex = 0; k.layerCount = 1;
      return k;
    }

  }

  // everything FSR owns; shared_ptr so d3d9_device.h needs only the forward declaration
  struct NfsuFsrState {
    ~NfsuFsrState() {
      if (vkd != nullptr) {
        for (VkPipeline p : pipe)
          vkd->vkDestroyPipeline(vkd->device(), p, nullptr);
      }
    }

    Rc<vk::DeviceFn>           vkd;
    const DxvkPipelineLayout*  layout[FsrPassCount] = { };
    VkPipeline                 pipe[FsrPassCount] = { };
    bool                       created = false, failed = false, waveOps = false, fp16 = false, logged = false;
    uint32_t                   w = 0u, h = 0u, ow = 0u, oh = 0u, spdMipCount = 0u;
    Rc<DxvkImage>              img[FiCount];
    Rc<DxvkImageView>          sv[FiCount];                      // storage view (mip 0)
    Rc<DxvkImageView>          rv[FiCount];                      // sampled view (all mips)
    Rc<DxvkImageView>          spdMip[6];                        // SPD storage views, mips 0..5 (clamped)
    Rc<DxvkBuffer>             cb, cbSpd, cbRcas;
    FsrConstants               k = { };
    bool                       first = true;
    uint32_t                   frame = 0u;
    float                      preExposure = 0.0f;
  };

  // the per-frame resource selection handed to the CS thread (Rc copies, so a resize on the next call is safe)
  struct NfsuFsrFrame {
    Rc<DxvkImage>      img[FiCount];
    Rc<DxvkImageView>  sv[FiCount], rv[FiCount], spdMip[6];
    Rc<DxvkImage>      color, depth, motion, reactive, out;
  };


  // fp16 ("_h" modules, FFX_HALF=1) only when DXVK enabled shaderFloat16 + shaderInt16 (DXVK turns both on whenever the
  // driver supports them) and the caller did not ask for fp32. The fp16 SPD pyramids WITH subgroup ops would also need
  // shaderSubgroupExtendedTypes, which DXVK never enables (lavapipe validation) -> with wave ops the pyramids stay fp32.
  static bool NfsuFsrCreatePipelines(const Rc<DxvkDevice>& dev, NfsuFsrState& s, bool allowFp16, bool forceNoWave) {
    using namespace nfsu_fsr;
    s.vkd = dev->vkd();
    for (VkPipeline& pipeline : s.pipe)                          // second call: retry after a failed fp16 attempt
      s.vkd->vkDestroyPipeline(s.vkd->device(), std::exchange(pipeline, VK_NULL_HANDLE), nullptr);
    const auto& p = dev->properties().vk11;
    s.waveOps = !forceNoWave && (p.subgroupSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT) && (p.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_QUAD_BIT)
             && p.subgroupSize >= 4u;
    const bool f16 = dev->features().vk12.shaderFloat16, i16 = dev->features().core.features.shaderInt16;
    s.fp16 = allowFp16 && f16 && i16;
    const bool h = s.fp16;
    struct Mod { const uint32_t* code; size_t size; const VkDescriptorType* types; uint32_t n; };
#define NFSU_FSR_MOD1(x) Mod { x##_spv, sizeof(x##_spv), x##_types, uint32_t(sizeof(x##_types) / sizeof(x##_types[0])) }
#define NFSU_FSR_MOD(x) (h ? NFSU_FSR_MOD1(x##_h) : NFSU_FSR_MOD1(x))
    const Mod mods[FsrPassCount] = {
      NFSU_FSR_MOD(prepare_inputs),
      s.waveOps ? NFSU_FSR_MOD1(luma_pyramid) : NFSU_FSR_MOD(luma_pyramid_nowave),
      s.waveOps ? NFSU_FSR_MOD1(shading_change_pyramid) : NFSU_FSR_MOD(shading_change_pyramid_nowave),
      NFSU_FSR_MOD(shading_change),
      NFSU_FSR_MOD(prepare_reactivity),
      NFSU_FSR_MOD(luma_instability),
      NFSU_FSR_MOD(accumulate),
      NFSU_FSR_MOD(accumulate_sharpen),
      NFSU_FSR_MOD(rcas),
      NFSU_FSR_MOD(debug_view),
    };
#undef NFSU_FSR_MOD
#undef NFSU_FSR_MOD1
    static const char* names[FsrPassCount] = { "prepare_inputs", "luma_pyramid", "shading_change_pyramid", "shading_change",
      "prepare_reactivity", "luma_instability", "accumulate", "accumulate_sharpen", "rcas", "debug_view" };
    bool ok = true;
    for (uint32_t i = 0; i < FsrPassCount; i++) {
      std::array<DxvkDescriptorSetLayoutBinding, 16> b;
      for (uint32_t j = 0; j < mods[i].n; j++)
        b[j] = DxvkDescriptorSetLayoutBinding(mods[i].types[j], 1u, VK_SHADER_STAGE_COMPUTE_BIT);
      s.layout[i] = dev->createBuiltInPipelineLayout(0u, 0u, 0u, mods[i].n, b.data());
      util::DxvkBuiltInShaderStage st;
      st.code = mods[i].code; st.size = mods[i].size; st.spec = nullptr;
      s.pipe[i] = s.layout[i] ? dev->createBuiltInComputePipeline(s.layout[i], st) : VK_NULL_HANDLE;
      if (!s.pipe[i]) { ok = false; nfsu::Log("fsr: pipeline %s FAILED", names[i]); }
    }
    nfsu::Log("fsr: %u pipelines %s (FSR 3.1.4 upscaler GLSL, %s; device shaderFloat16 %d shaderInt16 %d%s), SPD %s (subgroup size %u, ops 0x%x, stages 0x%x)",
      unsigned(FsrPassCount), ok ? "created" : "NOT all created", s.fp16 ? "fp16" : "fp32", int(f16), int(i16),
      allowFp16 ? "" : ", fp16 off by the caller", s.waveOps ? "quad ops (fp32 pyramids)" : forceNoWave ? "no wave ops (LDS, forced)" : "no wave ops (LDS)",
      p.subgroupSize, unsigned(p.subgroupSupportedOperations), unsigned(p.subgroupSupportedStages));
    return ok;
  }


  static void NfsuFsrCreateImages(const Rc<DxvkDevice>& dev, NfsuFsrState& s, uint32_t w, uint32_t h, uint32_t ow, uint32_t oh) {
    uint32_t hw = std::max(1u, w / 2u), hh = std::max(1u, h / 2u);
    s.spdMipCount = 1u;
    for (uint32_t m = std::max(hw, hh); m > 1u; m >>= 1u) s.spdMipCount++;
    VkDeviceSize bytes = 0;
    for (uint32_t i = 0; i < FiCount; i++) {
      const FsrImageDesc& d = g_fsrImages[i];
      DxvkImageCreateInfo ii;
      ii.type        = VK_IMAGE_TYPE_2D;
      ii.format      = d.format;
      ii.sampleCount = VK_SAMPLE_COUNT_1_BIT;
      ii.extent      = d.tiny ? VkExtent3D { 1u, 1u, 1u } : d.half ? VkExtent3D { hw, hh, 1u } : d.display ? VkExtent3D { ow, oh, 1u } : VkExtent3D { w, h, 1u };
      ii.numLayers   = 1u;
      ii.mipLevels   = d.mips ? s.spdMipCount : 1u;
      ii.usage       = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      ii.stages      = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
      ii.access      = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      ii.tiling      = VK_IMAGE_TILING_OPTIMAL;
      ii.layout      = VK_IMAGE_LAYOUT_GENERAL;
      ii.debugName   = d.name;
      s.img[i] = dev->createImage(ii, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      s.sv[i]  = s.img[i]->createView(FsrViewKey(s.img[i], VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_ASPECT_COLOR_BIT));
      s.rv[i]  = s.img[i]->createView(FsrViewKey(s.img[i], VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0u, ii.mipLevels));
      bytes += s.img[i]->getMemoryInfo().size;
    }
    for (uint32_t m = 0; m < 6u; m++)
      s.spdMip[m] = s.img[FiSpdMips]->createView(FsrViewKey(s.img[FiSpdMips], VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                                                            std::min(m, s.spdMipCount - 1u)));
    s.w = w; s.h = h; s.ow = ow; s.oh = oh;
    s.first = true;
    nfsu::Log("fsr: %u internal images for render %ux%u -> display %ux%u (SPD %ux%u, %u mips), %u KB", unsigned(FiCount), w, h, ow, oh,
      hw, hh, s.spdMipCount, unsigned(bytes / 1024u));
  }


  // params[32]:
  // [0..1] jitter in render pixels, FSR convention (ffxFsr3UpscalerGetJitterOffset: x right, y DOWN as in the SDK sample's
  //        projection translation (2*jx/w, -2*jy/h)) | [2..3] motion-vector scale as FfxFsr3UpscalerDispatchDescription::
  //        motionVectorScale (MV texture in UV units, previous - current: (w, h); in pixels: (1, 1)) |
  // [4..7] unjittered projection P00 P11 P22 P32 (D3D row-vector, standard 0..1 depth) | [8] frame time ms | [9] reset |
  // [10] sharpness 0..1 (< 0: no RCAS, accumulate writes the output) | [11] pre-exposure (0 = 1) | [12] debug view |
  // [13] view-space-to-metres (0 = 1) | [14] force no subgroup ops (read once, at pipeline creation) |
  // [15] 1 = fp32 only (else fp16 when the device has shaderFloat16 + shaderInt16; read once) | [16..31] 0
  // Sizes: colour, depth, motion and reactive at the RENDER size; out at the DISPLAY size (>= render; equal = Native AA).
  // The jitter sequence must have ffxFsr3UpscalerGetJitterPhaseCount = int(8 * (displayW / renderW)^2) phases (caller side).
  HRESULT D3D9DeviceEx::NfsuFsr(IDirect3DBaseTexture9* pColor, IDirect3DBaseTexture9* pDepth, IDirect3DBaseTexture9* pMotion,
                                IDirect3DBaseTexture9* pReactive, IDirect3DBaseTexture9* pOut, const float* pParams, UINT ParamCount) {
    D3D9DeviceLock lock = LockDevice();
    float p[32] = { };
    if (!pParams)
      return D3DERR_INVALIDCALL;
    for (uint32_t i = 0; i < std::min<uint32_t>(ParamCount, 32u); i++) p[i] = pParams[i];

    D3D9CommonTexture* color    = pColor    ? GetCommonTexture(pColor)    : nullptr;
    D3D9CommonTexture* depth    = pDepth    ? GetCommonTexture(pDepth)    : nullptr;
    D3D9CommonTexture* motion   = pMotion   ? GetCommonTexture(pMotion)   : nullptr;
    D3D9CommonTexture* reactive = pReactive ? GetCommonTexture(pReactive) : nullptr;
    D3D9CommonTexture* out      = pOut      ? GetCommonTexture(pOut)      : nullptr;
    if (!color || !depth || !motion || !out || !(out->Desc()->ImageUsage & VK_IMAGE_USAGE_STORAGE_BIT))
      return D3DERR_INVALIDCALL;

    if (!m_nfsuFsr)
      m_nfsuFsr = std::make_shared<NfsuFsrState>();
    NfsuFsrState& s = *m_nfsuFsr;
    if (s.failed)
      return D3DERR_INVALIDCALL;
    if (!s.created) {
      s.created = true;
      bool ok = NfsuFsrCreatePipelines(m_dxvkDevice, s, p[15] < 0.5f, p[14] > 0.5f);   // [15] 1 = fp32 only, [14] 1 = no wave ops
      if (!ok && s.fp16) {                                         // an fp16 pipeline failed: retry everything in fp32
        nfsu::Log("fsr: fp16 pipelines failed - retrying fp32");
        ok = NfsuFsrCreatePipelines(m_dxvkDevice, s, false, p[14] > 0.5f);
      }
      for (Rc<DxvkBuffer>* b : { &s.cb, &s.cbSpd, &s.cbRcas }) {
        DxvkBufferCreateInfo bi;
        bi.size   = b == &s.cb ? sizeof(FsrConstants) : b == &s.cbSpd ? sizeof(FsrSpdConstants) : sizeof(FsrRcasConstants);
        bi.usage  = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        bi.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        bi.access = VK_ACCESS_UNIFORM_READ_BIT;
        bi.debugName = "NFSU FSR constants";
        *b = m_dxvkDevice->createBuffer(bi, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      }
      if (!ok) { s.failed = true; return D3DERR_INVALIDCALL; }
    }

    Rc<DxvkImage> colorImage = color->GetImage(), depthImage = depth->GetImage(), motionImage = motion->GetImage();
    Rc<DxvkImage> outImage = out->GetImage(), reactiveImage = reactive ? reactive->GetImage() : nullptr;
    const uint32_t w = colorImage->info().extent.width, h = colorImage->info().extent.height;
    const uint32_t ow = outImage->info().extent.width, oh = outImage->info().extent.height;
    auto same = [&](const Rc<DxvkImage>& i) { return i->info().extent.width == w && i->info().extent.height == h; };
    if (!same(depthImage) || !same(motionImage) || (reactiveImage != nullptr && !same(reactiveImage)) || ow < w || oh < h)
      return D3DERR_INVALIDCALL;                                   // inputs at the render size, output at least as large

    bool newImages = false;
    if (s.img[0] == nullptr || s.w != w || s.h != h || s.ow != ow || s.oh != oh) {
      NfsuFsrCreateImages(m_dxvkDevice, s, w, h, ow, oh);
      newImages = true;
    }

    // ---- constants (SDK fsr3upscalerDispatch) ----
    const bool reset = s.first || p[9] > 0.5f;
    FsrConstants& k = s.k;
    if (s.first) {
      k = FsrConstants { };
      k.velocityFactor = 1.0f; k.reactivenessScale = 1.0f; k.shadingChangeScale = 1.0f;
      k.accumulationAddedPerFrame = 1.0f / 3.0f; k.minDisocclusionAccumulation = -1.0f / 3.0f;
    }
    k.previousFrameJitterOffset[0] = k.jitterOffset[0]; k.previousFrameJitterOffset[1] = k.jitterOffset[1];
    k.jitterOffset[0] = p[0]; k.jitterOffset[1] = p[1];
    k.previousFrameRenderSize[0] = k.renderSize[0]; k.previousFrameRenderSize[1] = k.renderSize[1];
    k.renderSize[0] = k.maxRenderSize[0] = int32_t(w); k.renderSize[1] = k.maxRenderSize[1] = int32_t(h);
    k.previousFrameUpscaleSize[0] = k.upscaleSize[0]; k.previousFrameUpscaleSize[1] = k.upscaleSize[1];
    k.upscaleSize[0] = k.maxUpscaleSize[0] = int32_t(ow); k.upscaleSize[1] = k.maxUpscaleSize[1] = int32_t(oh);
    k.downscaleFactor[0] = float(w) / float(ow); k.downscaleFactor[1] = float(h) / float(oh);
    // deviceToViewDepth straight from the projection: z_view = P32 / (d - P22) (= the SDK's near/far form for a
    // standard, non-infinite D3D projection: [0] = f/(f-n) = P22, [1] = -n f/(f-n) = P32); x/y factors 1/P00, 1/P11
    // NFSU's projection is INFINITE (log: P22 = 1.0, P32 = -0.5 -> near 0.5, far inf) and y-flipped (P11 < 0): like the SDK's
    // "infinite" case, P22 = 1 gets + FLT_EPSILON so the sky (d = 1) stays finite; x/y factors use |P00|, |P11|
    k.deviceToViewDepth[0] = p[6] >= 1.0f ? p[6] + 1.1920929e-7f : p[6]; k.deviceToViewDepth[1] = p[7];
    k.deviceToViewDepth[2] = p[4] != 0.0f ? 1.0f / std::fabs(p[4]) : 1.0f; k.deviceToViewDepth[3] = p[5] != 0.0f ? 1.0f / std::fabs(p[5]) : 1.0f;
    k.tanHalfFOV = k.deviceToViewDepth[2];                         // tan(horizontal fov / 2) = 1 / P00
    k.viewSpaceToMetersFactor = p[13] > 0.0f ? p[13] : 1.0f;
    k.motionVectorScale[0] = p[2] / float(w); k.motionVectorScale[1] = p[3] / float(h);
    k.motionVectorJitterCancellation[0] = k.motionVectorJitterCancellation[1] = 0.0f;
    const float phases = float(int32_t(8.0f * std::pow(float(ow) / float(w), 2.0f)));   // ffxFsr3UpscalerGetJitterPhaseCount
    if (reset || k.jitterPhaseCount == 0.0f) k.jitterPhaseCount = phases;
    else if (phases > k.jitterPhaseCount) k.jitterPhaseCount += 1.0f;             // SDK: ramp one phase per frame
    else if (phases < k.jitterPhaseCount) k.jitterPhaseCount -= 1.0f;
    k.deltaTime = std::max(0.0f, std::min(1.0f, p[8] / 1000.0f));
    float prevPre = s.preExposure;
    s.preExposure = p[11] != 0.0f ? p[11] : 1.0f;
    k.deltaPreExposure = prevPre > 0.0f ? s.preExposure / prevPre : 1.0f;
    k.frameIndex = reset ? 0.0f : k.frameIndex + 1.0f;

    FsrSpdConstants spd = { };
    {                                                              // ffxSpdSetup(rect 0,0,w,h, mips -1)
      uint32_t gx = (w - 1u) / 64u + 1u, gy = (h - 1u) / 64u + 1u;
      spd.workGroupOffset[0] = spd.workGroupOffset[1] = 0u;
      spd.numWorkGroups = gx * gy;
      spd.mips = uint32_t(std::min(std::floor(std::log2(float(std::max(w, h)))), 12.0f));
      spd.renderSize[0] = w; spd.renderSize[1] = h;
    }
    const bool sharpen = p[10] >= 0.0f;
    FsrRcasConstants rcas = { };
    {                                                              // FsrRcasCon(con, -2 * sharpness + 2)
      float sh = std::exp2(-((-2.0f * std::max(0.0f, std::min(1.0f, p[10]))) + 2.0f));
      std::memcpy(&rcas.rcasConfig[0], &sh, 4);
      uint32_t hs = FsrHalf(sh);
      rcas.rcasConfig[1] = hs | (hs << 16);
    }

    Rc<DxvkResourceAllocation> cbSlice = s.cb->allocateStorage(), spdSlice = s.cbSpd->allocateStorage(), rcasSlice = s.cbRcas->allocateStorage();
    std::memcpy(cbSlice->mapPtr(), &k, sizeof(k));
    std::memcpy(spdSlice->mapPtr(), &spd, sizeof(spd));
    std::memcpy(rcasSlice->mapPtr(), &rcas, sizeof(rcas));

    NfsuFsrFrame f;
    for (uint32_t i = 0; i < FiCount; i++) { f.img[i] = s.img[i]; f.sv[i] = s.sv[i]; f.rv[i] = s.rv[i]; }
    for (uint32_t m = 0; m < 6u; m++) f.spdMip[m] = s.spdMip[m];
    f.color = std::move(colorImage); f.depth = std::move(depthImage); f.motion = std::move(motionImage);
    f.reactive = std::move(reactiveImage); f.out = std::move(outImage);

    const bool odd = (s.frame & 1u) != 0u;
    struct Sel { uint32_t accSrv, accUav, upSrv, upUav, lhSrv, lhUav, lumaCur, lumaPrev; } sel = {
      odd ? FiAcc1 : FiAcc0, odd ? FiAcc0 : FiAcc1, odd ? FiUpscaled1 : FiUpscaled0, odd ? FiUpscaled0 : FiUpscaled1,
      odd ? FiLumaHist1 : FiLumaHist0, odd ? FiLumaHist0 : FiLumaHist1, odd ? FiLuma1 : FiLuma0, odd ? FiLuma0 : FiLuma1 };
    s.frame = (s.frame + 1u) % 16u;                                // SDK FSR3UPSCALER_MAX_QUEUED_FRAMES (even)
    s.first = false;

    const uint32_t gx = (w + 7u) / 8u, gy = (h + 7u) / 8u;
    const VkExtent2D gSpd = { (w - 1u) / 64u + 1u, (h - 1u) / 64u + 1u };
    const VkExtent2D gShade = { (uint32_t(float(w) * 0.5f) + 7u) / 8u, (uint32_t(float(h) * 0.5f) + 7u) / 8u };
    const VkExtent2D gRcas = { (ow + 15u) / 16u, (oh + 15u) / 16u };
    const VkExtent2D gDst = { (ow + 7u) / 8u, (oh + 7u) / 8u };    // accumulate / debug view run at the display size

    if (!s.logged) {
      s.logged = true;
      nfsu::Log("fsr: first dispatch render %ux%u -> display %ux%u (%.0f jitter phases), color fmt %d, depth fmt %d, motion fmt %d, reactive %s, out fmt %d, "
        "jitter %.3f %.3f px, MV scale %.1f %.1f, P %.4f %.4f %.6f %.6f, sharpen %s %.2f, debug %s",
        w, h, ow, oh, k.jitterPhaseCount, int(f.color->info().format), int(f.depth->info().format), int(f.motion->info().format),
        f.reactive != nullptr ? "bound" : "default 0", int(f.out->info().format), p[0], p[1], p[2], p[3],
        p[4], p[5], p[6], p[7], sharpen ? "RCAS" : "off", p[10], p[12] > 0.5f ? "on" : "off");
    }

    EmitCs([
      cF = std::move(f), cSel = sel,
      cS = m_nfsuFsr, cCb = s.cb, cCbSlice = std::move(cbSlice), cSpd = s.cbSpd, cSpdSlice = std::move(spdSlice),
      cRcas = s.cbRcas, cRcasSlice = std::move(rcasSlice), cNew = newImages, cReset = reset, cSharpen = sharpen,
      cDebug = p[12] > 0.5f, cG = VkExtent2D { gx, gy }, cGSpd = gSpd, cGShade = gShade, cGRcas = gRcas, cGDst = gDst
    ] (DxvkContext* ctx) mutable {
      const NfsuFsrState& st = *cS;                               // pipelines / layouts never change after creation
      if (cNew)
        for (uint32_t i = 0; i < FiCount; i++) ctx->initImage(cF.img[i], VK_IMAGE_LAYOUT_UNDEFINED);   // -> GENERAL, zeroed
      ctx->invalidateBuffer(cCb, std::move(cCbSlice));
      ctx->invalidateBuffer(cSpd, std::move(cSpdSlice));
      ctx->invalidateBuffer(cRcas, std::move(cRcasSlice));
      Rc<DxvkCommandList> cmd = ctx->beginExternalRendering();

      const VkImageAspectFlags C = VK_IMAGE_ASPECT_COLOR_BIT;
      Rc<DxvkImageView> colorV  = cF.color->createView(FsrViewKey(cF.color, VK_IMAGE_USAGE_SAMPLED_BIT, C));
      Rc<DxvkImageView> depthV  = cF.depth->createView(FsrViewKey(cF.depth, VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT));
      Rc<DxvkImageView> motionV = cF.motion->createView(FsrViewKey(cF.motion, VK_IMAGE_USAGE_SAMPLED_BIT, C));
      Rc<DxvkImageView> reactV  = cF.reactive != nullptr ? cF.reactive->createView(FsrViewKey(cF.reactive, VK_IMAGE_USAGE_SAMPLED_BIT, C))
                                                         : cF.rv[FiDefaultReactive];
      Rc<DxvkImageView> outV    = cF.out->createView(FsrViewKey(cF.out, VK_IMAGE_USAGE_STORAGE_BIT, C));

      VkPipelineStageFlags2 extStages = 0; VkAccessFlags2 extAccess = 0;
      for (const Rc<DxvkImage>* i : { &cF.color, &cF.depth, &cF.motion, &cF.reactive, &cF.out })
        if (*i != nullptr) { extStages |= (*i)->info().stages; extAccess |= (*i)->info().access; }
      const VkPipelineStageFlags2 CS = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, XF = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
      const VkAccessFlags2 RW = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
      auto barrier = [&](VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds, VkAccessFlags2 da) {
        VkMemoryBarrier2 b = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        b.srcStageMask = ss; b.srcAccessMask = sa; b.dstStageMask = ds; b.dstAccessMask = da;
        VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.memoryBarrierCount = 1u; dep.pMemoryBarriers = &b;
        cmd->cmdPipelineBarrier(DxvkCmdBuffer::ExecBuffer, &dep);
      };

      // before: inputs just rendered by the game, our images used last frame, constants written by the host
      barrier(extStages | CS | XF | VK_PIPELINE_STAGE_2_HOST_BIT, extAccess | RW | VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_HOST_WRITE_BIT,
              CS | XF, RW | VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_UNIFORM_READ_BIT);

      // clears (SDK: every frame reconstructed depth = far (float 1.0 bits, atomicMin target), SPD counter + mips = 0;
      // on reset also this frame's accumulation source, SPD mips, frame info = (-1, 1, 0, 0))
      auto clear = [&](uint32_t i, float r, float g, float b, float a) {
        VkClearColorValue v; v.float32[0] = r; v.float32[1] = g; v.float32[2] = b; v.float32[3] = a;
        VkImageSubresourceRange rg = { VK_IMAGE_ASPECT_COLOR_BIT, 0u, VK_REMAINING_MIP_LEVELS, 0u, 1u };
        cmd->cmdClearColorImage(DxvkCmdBuffer::ExecBuffer, cF.img[i]->handle(), VK_IMAGE_LAYOUT_GENERAL, &v, 1u, &rg);
      };
      clear(FiRecPrevDepth, 1.0f, 1.0f, 1.0f, 1.0f);
      clear(FiSpdAtomic, 0.0f, 0.0f, 0.0f, 0.0f);
      clear(FiSpdMips, 0.0f, 0.0f, 0.0f, 0.0f);
      if (cReset) {
        clear(cSel.accSrv, 0.0f, 0.0f, 0.0f, 0.0f);
        clear(FiFrameInfo, -1.0f, 1.0f, 0.0f, 0.0f);
      }
      barrier(XF, VK_ACCESS_2_TRANSFER_WRITE_BIT, CS, RW);

      auto S = [](const Rc<DxvkImageView>& v) { DxvkDescriptorWrite d; d.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; d.descriptor = v->getDescriptor(); return d; };
      auto U = [](const Rc<DxvkImageView>& v) { DxvkDescriptorWrite d; d.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; d.descriptor = v->getDescriptor(); return d; };
      auto B = [](const Rc<DxvkBuffer>& b) { DxvkDescriptorWrite d; d.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; d.buffer = b->getSliceInfo(); return d; };
      auto run = [&](FsrPass pass, std::initializer_list<DxvkDescriptorWrite> dw, uint32_t x, uint32_t y, bool last = false) {
        std::array<DxvkDescriptorWrite, 16> d;
        uint32_t n = 0;
        for (const DxvkDescriptorWrite& e : dw) d[n++] = e;
        cmd->cmdBindPipeline(DxvkCmdBuffer::ExecBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, st.pipe[pass]);
        cmd->bindResources(DxvkCmdBuffer::ExecBuffer, st.layout[pass], n, d.data(), 0u, nullptr);
        cmd->cmdDispatch(DxvkCmdBuffer::ExecBuffer, x, y, 1u);
        if (!last) barrier(CS, VK_ACCESS_2_SHADER_WRITE_BIT, CS, RW);
      };
      const auto& rv = cF.rv; const auto& sv = cF.sv;
      const Rc<DxvkImageView>& exposure = rv[FiDefaultExposure];   // no exposure texture: 0 -> 1.0 in Exposure()

      run(FsrPrepareInputs, { S(motionV), S(depthV), S(colorV), U(sv[FiDilatedMv]), U(sv[FiDilatedDepth]), U(sv[FiRecPrevDepth]),
                              U(sv[FiIntermediate]), U(sv[cSel.lumaCur]), B(cCb) }, cG.width, cG.height);
      run(FsrLumaPyramid, { S(rv[cSel.lumaCur]), S(rv[FiIntermediate]), U(sv[FiSpdAtomic]), U(sv[FiFrameInfo]),
                            U(cF.spdMip[0]), U(cF.spdMip[1]), U(cF.spdMip[2]), U(cF.spdMip[3]), U(cF.spdMip[4]), U(cF.spdMip[5]),
                            U(sv[FiFarthestMip1]), B(cCb), B(cSpd) }, cGSpd.width, cGSpd.height);
      run(FsrShadingChangePyramid, { S(rv[cSel.lumaCur]), S(rv[cSel.lumaPrev]), S(rv[FiDilatedMv]), S(exposure), U(sv[FiSpdAtomic]),
                                     U(cF.spdMip[0]), U(cF.spdMip[1]), U(cF.spdMip[2]), U(cF.spdMip[3]), U(cF.spdMip[4]), U(cF.spdMip[5]),
                                     B(cCb), B(cSpd) }, cGSpd.width, cGSpd.height);
      run(FsrShadingChange, { S(rv[FiSpdMips]), U(sv[FiShadingChange]), B(cCb) }, cGShade.width, cGShade.height);
      run(FsrPrepareReactivity, { S(rv[FiRecPrevDepth]), S(rv[FiDilatedMv]), S(rv[FiDilatedDepth]), S(reactV), S(rv[FiDefaultReactive]),
                                  S(rv[cSel.accSrv]), S(rv[FiShadingChange]), S(rv[cSel.lumaCur]), S(exposure),
                                  U(sv[FiDilatedReactive]), U(sv[FiNewLocks]), U(sv[cSel.accUav]), B(cCb) }, cG.width, cG.height);
      run(FsrLumaInstability, { S(exposure), S(rv[FiDilatedReactive]), S(rv[FiDilatedMv]), S(rv[FiFrameInfo]), S(rv[cSel.lhSrv]),
                                S(rv[FiFarthestMip1]), S(rv[cSel.lumaCur]), U(sv[cSel.lhUav]), U(sv[FiIntermediate]), B(cCb) }, cG.width, cG.height);
      // accumulate: r_lanczos_lut is never read (reference Lanczos permutation) -> any valid image
      run(cSharpen ? FsrAccumulateSharpen : FsrAccumulate,
          { S(exposure), S(rv[FiDilatedReactive]), S(rv[FiDilatedMv]), S(rv[cSel.upSrv]), S(exposure), S(rv[FiFarthestMip1]),
            S(rv[cSel.lumaCur]), S(rv[FiIntermediate]), S(colorV), U(sv[cSel.upUav]), U(outV), U(sv[FiNewLocks]), B(cCb) },
          cGDst.width, cGDst.height, !cSharpen && !cDebug);
      if (cSharpen)
        run(FsrRcas, { S(exposure), S(rv[cSel.upUav]), U(outV), B(cCb), B(cRcas) }, cGRcas.width, cGRcas.height, !cDebug);
      if (cDebug)
        run(FsrDebugView, { S(rv[FiDilatedReactive]), S(rv[FiDilatedMv]), S(rv[FiDilatedDepth]), S(rv[cSel.upUav]), S(exposure),
                            U(outV), B(cCb) }, cGDst.width, cGDst.height, true);

      // after: out is read by the game-side mod, the inputs return to their D3D9 use
      barrier(CS, RW, extStages | CS | XF, extAccess | RW | VK_ACCESS_2_TRANSFER_WRITE_BIT);

      for (uint32_t i = 0; i < FiCount; i++) cmd->track(cF.img[i], DxvkAccess::Write);
      cmd->track(cF.out, DxvkAccess::Write);
      for (const Rc<DxvkImage>* i : { &cF.color, &cF.depth, &cF.motion, &cF.reactive })
        if (*i != nullptr) cmd->track(*i, DxvkAccess::Read);
      cmd->track(cCb, DxvkAccess::Read);
      cmd->track(cSpd, DxvkAccess::Read);
      cmd->track(cRcas, DxvkAccess::Read);
    });
    return D3D_OK;
  }

}
