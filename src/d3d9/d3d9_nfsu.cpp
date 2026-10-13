#include "d3d9_nfsu.h"
#include "d3d9_include.h"

#include <version.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <string>
#include <utility>

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
  }

}

extern "C" {

  // Interface for the game-side mod, which looks these up with GetProcAddress

  DLLEXPORT const char* __stdcall NFSU_GetVersion() {
    return dxvk::nfsu::g_version;
  }

}
