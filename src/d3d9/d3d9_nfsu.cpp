#include "d3d9_nfsu.h"
#include "d3d9_device.h"

#include <version.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../util/util_env.h"
#include "../util/util_string.h"
#include "../util/util_time.h"

namespace dxvk::nfsu {

  static const char* g_version = "DXVK " DXVK_VERSION " (nfsu)";

  static std::mutex    g_logMutex;
  static std::ofstream g_log;
  static bool          g_logOpened = false;

  // Folder of the game executable, with a trailing separator
  static std::string GetExeDir() {
    std::string path = env::getExePath();
    size_t n = path.find_last_of("\\/");
    return n != std::string::npos ? path.substr(0, n + 1) : std::string();
  }


  void Log(const char* fmt, ...) {
    std::lock_guard<std::mutex> lock(g_logMutex);

    if (!g_logOpened) {
      g_logOpened = true;

      // The mod keeps its logs in scripts/logs next to the executable
      std::string dir = GetExeDir();
      env::createDirectory(dir + "scripts");
      env::createDirectory(dir + "scripts/logs");

      g_log.open(str::topath((dir + "scripts/logs/dxvk.log").c_str()).c_str(), std::ios_base::out | std::ios_base::trunc);

      if (!g_log.is_open())
        g_log.open(str::topath((dir + "d3d9_nfsu.log").c_str()).c_str(), std::ios_base::out | std::ios_base::trunc);
    }

    if (!g_log.is_open())
      return;

    char line[1024];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);

    g_log << line << std::endl;
  }


  void OnCreate(bool extended) {
    static bool once = false;

    if (std::exchange(once, true))
      return;

    Log("%s loaded as the D3D9 renderer (Direct3DCreate9%s) for %s", g_version,
      extended ? "Ex" : "", env::getExePath().c_str());

    g_inventory = std::ifstream(str::topath((GetExeDir() + "nfsu_inventory.on").c_str()).c_str()).is_open();

    Log("shader inventory: %s", g_inventory
      ? "on (nfsu_inventory.on found)"
      : "off (create nfsu_inventory.on next to the executable to enable)");
  }


  static uint64_t Fnv1a(const void* data, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint64_t h = 0xcbf29ce484222325ull;

    for (size_t i = 0; i < n; i++) {
      h ^= p[i];
      h *= 0x100000001b3ull;
    }

    return h;
  }

  struct PairStat {
    uint64_t vs, ps;
    uint64_t draws, prims;
  };

  bool g_inventory = false;

  static std::mutex g_invMutex;
  static std::unordered_map<const void*, uint64_t> g_hashOf;   // shader object -> bytecode hash
  static std::unordered_map<uint64_t, PairStat>    g_window;   // per (vs, ps) pair, current window
  static uint64_t g_frame = 0, g_windowStart = 0;


  void OnShaderCreated(bool pixel, const void* object, const void* code, size_t length) {
    if (!g_inventory || !code || length < sizeof(uint32_t))
      return;

    uint64_t hash = Fnv1a(code, length);
    uint32_t version = *static_cast<const uint32_t*>(code);
    bool fresh = true;

    {
      std::lock_guard<std::mutex> lock(g_invMutex);

      for (const auto& e : g_hashOf) {
        if (e.second == hash) {
          fresh = false;
          break;
        }
      }

      g_hashOf[object] = hash;
    }

    const char* type = pixel ? "ps" : "vs";

    Log("create %s %016llx %zu bytes %s_%u_%u obj=%p%s", type, (unsigned long long) hash, length,
      type, (version >> 8) & 0xff, version & 0xff, object, fresh ? "" : " (same bytecode as an earlier shader)");

    if (!fresh)
      return;

    std::string dir = GetExeDir() + "nfsu_shaders";
    env::createDirectory(dir);

    char name[64];
    snprintf(name, sizeof(name), "/%s_%016llx.bin", type, (unsigned long long) hash);

    std::ofstream file(str::topath((dir + name).c_str()).c_str(), std::ios_base::binary | std::ios_base::trunc);
    file.write(static_cast<const char*>(code), length);
  }


  void OnDraw(const void* vs, const void* ps, uint32_t primitives) {
    std::lock_guard<std::mutex> lock(g_invMutex);

    auto hv = vs ? g_hashOf.find(vs) : g_hashOf.end();
    auto hp = ps ? g_hashOf.find(ps) : g_hashOf.end();

    uint64_t v = hv != g_hashOf.end() ? hv->second : 0;
    uint64_t p = hp != g_hashOf.end() ? hp->second : 0;

    PairStat& stat = g_window[v * 0x9e3779b97f4a7c15ull ^ p];
    stat.vs = v;
    stat.ps = p;
    stat.draws += 1;
    stat.prims += primitives;
  }


  // Frame pacing: CPU-side time between presents, which includes any
  // vsync or frame latency wait, plus pipelines compiled per window
  constexpr uint32_t PacingWindow = 600;

  static dxvk::high_resolution_clock::time_point g_lastPresent;
  static bool     g_hasLastPresent = false;
  static float    g_frameTimes[PacingWindow];
  static uint32_t g_frameTimeCount = 0;
  static uint64_t g_pipeG = 0, g_pipeL = 0, g_pipeGPrev = 0, g_pipeLPrev = 0, g_pipeGWin = 0, g_pipeLWin = 0;
  static uint32_t g_slowWithPipes = 0;


  void SetPipelineCounts(uint64_t graphics, uint64_t libraries) {
    g_pipeG = graphics;
    g_pipeL = libraries;
  }


  static void FramePacing() {
    auto now = dxvk::high_resolution_clock::now();

    uint64_t dg = g_pipeGPrev ? g_pipeG - g_pipeGPrev : 0;
    uint64_t dl = g_pipeLPrev ? g_pipeL - g_pipeLPrev : 0;

    g_pipeGPrev = g_pipeG;
    g_pipeLPrev = g_pipeL;
    g_pipeGWin += dg;
    g_pipeLWin += dl;

    if (g_hasLastPresent) {
      float ms = std::chrono::duration<float, std::milli>(now - g_lastPresent).count();
      g_frameTimes[g_frameTimeCount++] = ms;

      if (ms > 33.4f && (dg || dl))
        g_slowWithPipes++;
    }

    g_lastPresent = now;
    g_hasLastPresent = true;

    if (g_frameTimeCount < PacingWindow)
      return;

    float sorted[PacingWindow];
    float sum = 0.0f;
    uint32_t over33 = 0, over50 = 0;

    for (uint32_t i = 0; i < PacingWindow; i++) {
      sorted[i] = g_frameTimes[i];
      sum += g_frameTimes[i];
      over33 += g_frameTimes[i] > 33.4f;
      over50 += g_frameTimes[i] > 50.0f;
    }

    std::sort(sorted, sorted + PacingWindow);

    Log("pacing: %.1f fps, frame ms p50 %.1f p95 %.1f p99 %.1f max %.1f, hitches >33ms %u >50ms %u | "
        "new pipelines %llu, shader libraries %llu, hitches with new pipelines %u (total pipelines %llu)",
      1000.0f * PacingWindow / sum, sorted[PacingWindow / 2], sorted[PacingWindow * 95 / 100],
      sorted[PacingWindow * 99 / 100], sorted[PacingWindow - 1], over33, over50,
      (unsigned long long) g_pipeGWin, (unsigned long long) g_pipeLWin, g_slowWithPipes, (unsigned long long) g_pipeG);

    g_frameTimeCount = 0;
    g_pipeGWin = g_pipeLWin = 0;
    g_slowWithPipes = 0;
  }


  void OnFirstPresent(const char* device, const char* driver, const char* driverInfo, bool gpl,
                      uint32_t gameInterval, uint32_t effInterval, int32_t maxFrameRate) {
    Log("GPU %s, driver %s %s, graphics pipeline library %s", device, driver, driverInfo, gpl
      ? "yes (pipelines link fast, little compile stutter)"
      : "no (pipelines compile at draw time, stutter on new effects)");

    Log("present: game PresentationInterval %u, effective sync interval %u (%s), d3d9.maxFrameRate %d",
      gameInterval, effInterval, effInterval ? "vsync" : "no vsync", maxFrameRate);
  }


  void OnPresent() {
    FramePacing();

    if (!g_inventory)
      return;

    std::vector<PairStat> rows;
    uint64_t frames;

    {
      std::lock_guard<std::mutex> lock(g_invMutex);

      if (++g_frame - g_windowStart < PacingWindow)
        return;

      frames = g_frame - g_windowStart;
      g_windowStart = g_frame;

      for (const auto& e : g_window)
        rows.push_back(e.second);

      g_window.clear();
    }

    std::sort(rows.begin(), rows.end(), [] (const PairStat& a, const PairStat& b) {
      return a.draws > b.draws;
    });

    uint64_t draws = 0, prims = 0;

    for (const auto& r : rows) {
      draws += r.draws;
      prims += r.prims;
    }

    Log("---- frames %llu..%llu: %zu shader pairs, %.1f draws/frame, %.0f prims/frame (vs 0 / ps 0 = fixed function)",
      (unsigned long long) (g_frame - frames), (unsigned long long) g_frame, rows.size(),
      double(draws) / frames, double(prims) / frames);

    for (const auto& r : rows) {
      Log("  vs %016llx ps %016llx  %8.1f draws/frame %9.0f prims/frame", (unsigned long long) r.vs,
        (unsigned long long) r.ps, double(r.draws) / frames, double(r.prims) / frames);
    }
  }

}

extern "C" {

  using dxvk::D3D9DeviceEx;

  // Interface for the game-side mod, which looks these up with GetProcAddress.
  // pDevice must be the device object created by this DLL, not a wrapper.

  DLLEXPORT const char* __stdcall NFSU_GetVersion() {
    return dxvk::nfsu::g_version;
  }

  // Creates a render target texture (RGBA16F) that compute passes can write to
  DLLEXPORT HRESULT __stdcall NFSU_CreateStorageTexture(IDirect3DDevice9* pDevice, UINT Width, UINT Height,
                                                        IDirect3DTexture9** ppTexture) {
    if (!pDevice)
      return D3DERR_INVALIDCALL;

    return static_cast<D3D9DeviceEx*>(pDevice)->NfsuCreateStorageTexture(Width, Height, ppTexture);
  }

  // Clustered lighting, see D3D9DeviceEx::NfsuCluster
  DLLEXPORT HRESULT __stdcall NFSU_ClusterLighting(IDirect3DDevice9* pDevice, IDirect3DTexture9* pDepth,
                                                   IDirect3DTexture9* pHeat, IDirect3DTexture9* pLight,
                                                   IDirect3DTexture9* pHaze, IDirect3DTexture9* pAtlas,
                                                   const float* pParams, const float* pLights, UINT LightCount) {
    if (!pDevice)
      return D3DERR_INVALIDCALL;

    return static_cast<D3D9DeviceEx*>(pDevice)->NfsuCluster(pDepth, pHeat, pLight, pHaze, pAtlas, pParams, pLights, LightCount);
  }

  // Froxel fog, see D3D9DeviceEx::NfsuFog. pShadowMaps points to 12 textures, any of which may be null.
  DLLEXPORT HRESULT __stdcall NFSU_FroxelFog(IDirect3DDevice9* pDevice, IDirect3DTexture9* pDepth,
                                             IDirect3DTexture9* pOut, IDirect3DTexture9* const* pShadowMaps,
                                             IDirect3DTexture9* pAtlas, const float* pParams,
                                             const float* pLights, UINT LightCount) {
    if (!pDevice)
      return D3DERR_INVALIDCALL;

    IDirect3DBaseTexture9* maps[12] = { };

    for (uint32_t i = 0; i < 12; i++)
      maps[i] = pShadowMaps ? pShadowMaps[i] : nullptr;

    return static_cast<D3D9DeviceEx*>(pDevice)->NfsuFog(pDepth, pOut, maps, pAtlas, pParams, pLights, LightCount);
  }

}
