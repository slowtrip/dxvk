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

}
