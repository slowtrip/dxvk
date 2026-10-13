// NFSU: end-to-end validation run of the NFSU_* compute passes through the real D3D9 front end, using a
// native Linux build of this tree (libdxvk_d3d9.so), an SDL2 window, lavapipe and the Khronos validation
// layer. Unlike fsr/test/fsr_lvp_test.cpp, which only runs the FSR shaders, this exercises the DXVK glue:
// external rendering between D3D9 draws, descriptors, barriers, image layouts and buffer invalidation.
//
// Build and run with run_d3d9_test.sh next to this file.
// argv[1] = variant bits: 1 fp32 only, 2 no subgroup ops, 4 upscale (render 2/3 size), 8 no RCAS,
//           16 FSR debug view on the last frame
// argv[2] = number of frames
#include <SDL2/SDL.h>
#include <windows.h>
#include <d3d9.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
  const char* NFSU_GetVersion();
  HRESULT NFSU_CreateStorageTexture(IDirect3DDevice9*, UINT, UINT, IDirect3DTexture9**);
  HRESULT NFSU_ClusterLighting(IDirect3DDevice9*, IDirect3DTexture9*, IDirect3DTexture9*, IDirect3DTexture9*, IDirect3DTexture9*,
                               IDirect3DTexture9*, const float*, const float*, UINT);
  HRESULT NFSU_FroxelFog(IDirect3DDevice9*, IDirect3DTexture9*, IDirect3DTexture9*, IDirect3DTexture9* const*, IDirect3DTexture9*,
                         const float*, const float*, UINT);
  HRESULT NFSU_Fsr(IDirect3DDevice9*, IDirect3DTexture9*, IDirect3DTexture9*, IDirect3DTexture9*, IDirect3DTexture9*,
                   IDirect3DTexture9*, const float*, UINT);
}

static int g_fail = 0;
#define CHECK(hr, what) do { HRESULT h_ = (hr); if (FAILED(h_)) { printf("FAIL %s: 0x%08x\n", what, unsigned(h_)); g_fail++; } } while (0)

struct Vtx { float x, y, z, rhw; DWORD c; float u, v; };
static const DWORD FVF = D3DFVF_XYZRHW | D3DFVF_DIFFUSE | D3DFVF_TEX1;

static void quad(IDirect3DDevice9* dev, float x0, float y0, float x1, float y1, float z, DWORD c) {
  Vtx v[4] = { { x0, y0, z, 1, c, 0, 0 }, { x1, y0, z, 1, c, 1, 0 }, { x0, y1, z, 1, c, 0, 1 }, { x1, y1, z, 1, c, 1, 1 } };
  dev->SetFVF(FVF);
  dev->DrawPrimitiveUP(D3DPT_TRIANGLESTRIP, 2, v, sizeof(Vtx));
}

static float half(uint16_t h) {
  int e = (h >> 10) & 31, m = h & 1023;
  float f = e == 0 ? std::ldexp(float(m), -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp(float(m + 1024), e - 25);
  return (h & 0x8000) ? -f : f;
}

// mean of |rgba| of a storage texture (rgba16f), read through StretchRect -> render target -> system memory
static bool readback(IDirect3DDevice9* dev, IDirect3DTexture9* tex, const char* name, bool expectNonZero) {
  D3DSURFACE_DESC d; tex->GetLevelDesc(0, &d);
  IDirect3DSurface9 *src = nullptr, *rt = nullptr, *sys = nullptr;
  tex->GetSurfaceLevel(0, &src);
  HRESULT hr = dev->CreateRenderTarget(d.Width, d.Height, D3DFMT_A16B16G16R16F, D3DMULTISAMPLE_NONE, 0, FALSE, &rt, nullptr);
  if (SUCCEEDED(hr)) hr = dev->CreateOffscreenPlainSurface(d.Width, d.Height, D3DFMT_A16B16G16R16F, D3DPOOL_SYSTEMMEM, &sys, nullptr);
  if (SUCCEEDED(hr)) hr = dev->StretchRect(src, nullptr, rt, nullptr, D3DTEXF_NONE);
  if (SUCCEEDED(hr)) hr = dev->GetRenderTargetData(rt, sys);
  double sum[4] = { }; size_t nan = 0;
  D3DLOCKED_RECT lr;
  if (SUCCEEDED(hr) && SUCCEEDED(hr = sys->LockRect(&lr, nullptr, D3DLOCK_READONLY))) {
    for (UINT y = 0; y < d.Height; y++) {
      const uint16_t* p = reinterpret_cast<const uint16_t*>(static_cast<const char*>(lr.pBits) + y * lr.Pitch);
      for (UINT x = 0; x < d.Width; x++)
        for (int c = 0; c < 4; c++) {
          float f = half(p[x * 4 + c]);
          if (std::isfinite(f)) sum[c] += std::fabs(f); else nan++;
        }
    }
    sys->UnlockRect();
  }
  double n = double(d.Width) * d.Height;
  bool nonZero = sum[0] + sum[1] + sum[2] + sum[3] > 0.0;
  bool ok = SUCCEEDED(hr) && (!expectNonZero || nonZero);
  printf("%s %-10s %ux%u mean |rgba| %.4f %.4f %.4f %.4f, non-finite %zu%s\n", ok ? "ok  " : "FAIL", name, d.Width, d.Height,
         sum[0] / n, sum[1] / n, sum[2] / n, sum[3] / n, nan, SUCCEEDED(hr) ? "" : " (readback failed)");
  if (!ok) g_fail++;
  if (sys) sys->Release();
  if (rt) rt->Release();
  if (src) src->Release();
  return ok;
}

int main(int argc, char** argv) {
  const int variant = argc > 1 ? atoi(argv[1]) : 0;
  const int frames  = argc > 2 ? atoi(argv[2]) : 5;
  const UINT W = 640, H = 480;
  const UINT RW = (variant & 4) ? 426 : W, RH = (variant & 4) ? 320 : H;   // render size (FSR inputs)

  if (SDL_Init(SDL_INIT_VIDEO)) { printf("SDL_Init: %s\n", SDL_GetError()); return 2; }
  SDL_Window* win = SDL_CreateWindow("nfsu d3d9 test", 0, 0, W, H, SDL_WINDOW_VULKAN);
  if (!win) { printf("SDL_CreateWindow: %s\n", SDL_GetError()); return 2; }

  printf("%s, variant %d, render %ux%u -> display %ux%u\n", NFSU_GetVersion(), variant, RW, RH, W, H);
  IDirect3D9* d3d = Direct3DCreate9(D3D_SDK_VERSION);
  D3DPRESENT_PARAMETERS pp = { };
  pp.Windowed = TRUE; pp.SwapEffect = D3DSWAPEFFECT_DISCARD; pp.BackBufferFormat = D3DFMT_X8R8G8B8;
  pp.BackBufferWidth = W; pp.BackBufferHeight = H; pp.BackBufferCount = 1; pp.hDeviceWindow = reinterpret_cast<HWND>(win);
  pp.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
  IDirect3DDevice9* dev = nullptr;
  HRESULT hr = d3d->CreateDevice(0, D3DDEVTYPE_HAL, pp.hDeviceWindow, D3DCREATE_HARDWARE_VERTEXPROCESSING, &pp, &dev);
  if (FAILED(hr) || !dev) { printf("CreateDevice failed 0x%08x\n", unsigned(hr)); return 2; }

  const D3DFORMAT INTZ = D3DFORMAT(MAKEFOURCC('I', 'N', 'T', 'Z'));
  IDirect3DTexture9 *depth = nullptr, *color = nullptr, *motion = nullptr, *atlas = nullptr, *lampMap[2] = { }, *lampRt = nullptr;
  CHECK(dev->CreateTexture(RW, RH, 1, D3DUSAGE_DEPTHSTENCIL, INTZ, D3DPOOL_DEFAULT, &depth, nullptr), "INTZ depth");
  CHECK(dev->CreateTexture(RW, RH, 1, D3DUSAGE_RENDERTARGET, D3DFMT_A16B16G16R16F, D3DPOOL_DEFAULT, &color, nullptr), "HDR colour");
  CHECK(dev->CreateTexture(RW, RH, 1, D3DUSAGE_RENDERTARGET, D3DFMT_G16R16F, D3DPOOL_DEFAULT, &motion, nullptr), "motion");
  CHECK(dev->CreateTexture(512, 512, 1, 0, D3DFMT_L16, D3DPOOL_MANAGED, &atlas, nullptr), "atlas L16");
  for (int i = 0; i < 2; i++)
    CHECK(dev->CreateTexture(256, 256, 1, D3DUSAGE_DEPTHSTENCIL, INTZ, D3DPOOL_DEFAULT, &lampMap[i], nullptr), "lamp map");
  CHECK(dev->CreateTexture(256, 256, 1, D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &lampRt, nullptr), "lamp rt");
  if (g_fail) return 1;
  {
    D3DLOCKED_RECT lr;
    if (SUCCEEDED(atlas->LockRect(0, &lr, nullptr, 0))) {
      for (int y = 0; y < 512; y++) {
        uint16_t* p = reinterpret_cast<uint16_t*>(static_cast<char*>(lr.pBits) + y * lr.Pitch);
        for (int x = 0; x < 512; x++) p[x] = uint16_t(((x / 32 + y / 32) & 1) ? 60000 : 20000);
      }
      atlas->UnlockRect(0);
    }
  }

  // storage textures: dual light image, 2 haze texels per 16 px tile
  IDirect3DTexture9 *light = nullptr, *haze = nullptr, *heat = nullptr, *fogOut = nullptr, *fsrOut = nullptr;
  CHECK(NFSU_CreateStorageTexture(dev, RW / 2 * 2, RH / 2, &light), "storage light");
  CHECK(NFSU_CreateStorageTexture(dev, (RW + 15) / 16 * 2, (RH + 15) / 16, &haze), "storage haze");
  CHECK(NFSU_CreateStorageTexture(dev, RW / 2, RH / 2, &heat), "storage heat");
  CHECK(NFSU_CreateStorageTexture(dev, RW / 2, RH / 2, &fogOut), "storage fog");
  CHECK(NFSU_CreateStorageTexture(dev, W, H, &fsrOut), "storage fsr out");
  if (g_fail) return 1;

  IDirect3DSurface9 *bb = nullptr, *colorS = nullptr, *depthS = nullptr, *motionS = nullptr, *lampS[2] = { }, *lampRtS = nullptr;
  dev->GetRenderTarget(0, &bb);
  color->GetSurfaceLevel(0, &colorS); depth->GetSurfaceLevel(0, &depthS); motion->GetSurfaceLevel(0, &motionS);
  lampMap[0]->GetSurfaceLevel(0, &lampS[0]); lampMap[1]->GetSurfaceLevel(0, &lampS[1]); lampRt->GetSurfaceLevel(0, &lampRtS);

  const float P00 = 1.0f, P11 = -1.333f, P22 = 1.0f, P32 = -0.5f;     // NFSU-like: infinite far, y flipped, near 0.5

  // lights: pos.xyz range | colour.rgb r0 | dir.xyz cosOuter | cosInner extra.yzw   (view space)
  std::vector<float> lights;
  for (int i = 0; i < 24; i++) {
    float l[16] = { float(i % 6) * 4.0f - 10.0f, 2.0f - float(i / 6), 6.0f + float(i) * 1.5f, 18.0f,
                    1.0f, 0.8f, 0.5f, i % 5 == 0 ? -1.0f : 1.0f,     // r0 < 0: moving light (right half of the dual image)
                    0.0f, -1.0f, 0.0f, -1.0f,
                    -1.0f, 0.0f, 0.0f, i < 4 ? 6.0f + float(i) + 0.05f + 0.4f : 0.0f };   // first 4: atlas tiles 0..3
    lights.insert(lights.end(), l, l + 16);
  }

  for (int f = 0; f < frames; f++) {
    dev->BeginScene();

    // lamp shadow maps (depth only content)
    for (int i = 0; i < 2; i++) {
      dev->SetRenderTarget(0, lampRtS); dev->SetDepthStencilSurface(lampS[i]);
      dev->Clear(0, nullptr, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0, 1.0f, 0);
      dev->SetRenderState(D3DRS_ZENABLE, TRUE); dev->SetRenderState(D3DRS_ZWRITEENABLE, TRUE);
      quad(dev, 40, 40, 200, 200, 0.6f, 0xffffffff);
    }

    // scene: HDR colour + INTZ depth, then motion vectors
    dev->SetRenderTarget(0, colorS); dev->SetDepthStencilSurface(depthS);
    dev->Clear(0, nullptr, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0xff203040, 1.0f, 0);
    dev->SetTexture(0, nullptr);
    quad(dev, 0, RH * 0.5f, float(RW), float(RH), 0.95f, 0xff808080);                           // ground
    quad(dev, 100 + f * 3.0f, 80, 300 + f * 3.0f, 300, 0.90f, 0xffe0a040);                      // moving box
    dev->SetRenderTarget(0, motionS); dev->SetDepthStencilSurface(nullptr);
    dev->Clear(0, nullptr, D3DCLEAR_TARGET, 0, 1.0f, 0);

    // ---- cluster lighting (lists + haze + heat + dual light output + atlas shadows) ----
    float cp[32] = { 0.0f, f * 0.016f, 1.0f, 1.0f, P00, P11, P22, P32, 0, 0, 0, 0, 0, 0, 0, 100.0f,
                     32.0f, 1.0f, 0.01f, 1.0f, 0.0f, 1.0f, 0.0f, 100.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.5f };
    CHECK(NFSU_ClusterLighting(dev, depth, heat, light, haze, atlas, cp, lights.data(), UINT(lights.size() / 16)), "NFSU_ClusterLighting");
    if (f == 0)                                                   // once without the atlas and the optional outputs
      CHECK(NFSU_ClusterLighting(dev, depth, nullptr, light, nullptr, nullptr, cp, lights.data(), UINT(lights.size() / 16)), "NFSU_ClusterLighting (light only)");

    // ---- froxel fog (2 lamp maps + atlas, the other 10 slots missing) ----
    float fp[240] = { };
    fp[0] = P00; fp[1] = P11; fp[2] = P22; fp[3] = P32;
    fp[4] = 1.0f; fp[9] = 1.0f; fp[14] = 1.0f;                    // inverse view rows (identity 3x4)
    for (int i = 16; i < 32; i++) fp[i] = 0.5f;                   // camera / mist / mist2 blocks: plausible non-zero values
    fp[22] = 80.0f; fp[27] = 0.02f;
    fp[30] = 1.0f; fp[31] = 1.0f;                                 // lamp map 0 / 1 on
    for (int i = 0; i < 2; i++) { float* m = &fp[32 + i * 16]; m[0] = m[5] = m[10] = m[15] = 1.0f; }
    fp[68] = 0.15f;                                               // history blend
    fp[69] = fp[74] = fp[79] = fp[84] = 1.0f;                     // last frame's world->view
    fp[85] = P00; fp[86] = P11; fp[87] = P22; fp[88] = P32;
    fp[92] = 0.5f; fp[93] = 0.8f; fp[94] = f > 0 ? 1.0f : 0.0f;
    for (int t = 0; t < 16; t++) { fp[168 + t * 4] = float(t); fp[168 + t * 4 + 2] = 10.0f; fp[168 + t * 4 + 3] = 1.0f; }
    fp[232] = 1.0f; fp[233] = 0.5f; fp[234] = 100.0f; fp[235] = 4.0f; fp[236] = 128.0f;
    IDirect3DTexture9* ls[12] = { lampMap[0], nullptr, lampMap[1], nullptr };
    CHECK(NFSU_FroxelFog(dev, depth, fogOut, ls, atlas, fp, lights.data(), UINT(lights.size() / 16)), "NFSU_FroxelFog");
    if (f == 0)                                                   // once without the atlas
      CHECK(NFSU_FroxelFog(dev, depth, fogOut, ls, nullptr, fp, lights.data(), UINT(lights.size() / 16)), "NFSU_FroxelFog (no atlas)");

    // ---- FSR (jittered colour + depth + motion -> display-size output) ----
    float sp[32] = { };
    sp[0] = 0.25f * ((f & 1) ? 1.0f : -1.0f); sp[1] = 0.125f * ((f & 2) ? 1.0f : -1.0f);
    sp[2] = float(RW); sp[3] = float(RH);
    sp[4] = P00; sp[5] = P11; sp[6] = P22; sp[7] = P32;
    sp[8] = 16.6f; sp[9] = 0.0f; sp[10] = (variant & 8) ? -1.0f : 0.5f;
    sp[12] = f == frames - 1 && (variant & 16) ? 1.0f : 0.0f;     // debug view on the last frame (variant 16)
    sp[14] = (variant & 2) ? 1.0f : 0.0f; sp[15] = (variant & 1) ? 1.0f : 0.0f;
    CHECK(NFSU_Fsr(dev, color, depth, motion, nullptr, fsrOut, sp, 32), "NFSU_Fsr");

    // ---- back in D3D9: sample every compute output (and the depth) into the back buffer, then present ----
    dev->SetRenderTarget(0, bb); dev->SetDepthStencilSurface(nullptr);
    dev->Clear(0, nullptr, D3DCLEAR_TARGET, 0xff000000, 1.0f, 0);
    dev->SetRenderState(D3DRS_ZENABLE, FALSE);
    IDirect3DTexture9* show[6] = { fsrOut, light, haze, heat, fogOut, depth };
    for (int i = 0; i < 6; i++) {
      dev->SetTexture(0, show[i]);
      quad(dev, float(i % 3) * 213.0f, float(i / 3) * 240.0f, float(i % 3 + 1) * 213.0f, float(i / 3 + 1) * 240.0f, 0.5f, 0xffffffff);
    }
    dev->SetTexture(0, nullptr);
    dev->EndScene();
    CHECK(dev->Present(nullptr, nullptr, nullptr, nullptr), "Present");
    printf("frame %d done\n", f);
    fflush(stdout);
  }

  // outputs must contain something (and finite numbers)
  readback(dev, light, "light", true);
  readback(dev, haze, "haze", false);
  readback(dev, heat, "heat", true);
  readback(dev, fogOut, "fog", true);
  readback(dev, fsrOut, "fsr out", true);

  for (IDirect3DSurface9* s : { bb, colorS, depthS, motionS, lampS[0], lampS[1], lampRtS }) if (s) s->Release();
  for (IDirect3DTexture9* t : { depth, color, motion, atlas, lampMap[0], lampMap[1], lampRt, light, haze, heat, fogOut, fsrOut }) if (t) t->Release();
  ULONG left = dev->Release();
  d3d->Release();
  SDL_DestroyWindow(win);
  SDL_Quit();
  printf("%s (device refs left %lu, %d failures)\n", g_fail ? "TEST FAILED" : "TEST PASSED", (unsigned long)left, g_fail);
  return g_fail ? 1 : 0;
}
