#include "d3d9_nfsu.h"
#include "d3d9_include.h"

#include <version.h>

#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <string>
#include <utility>

#include "../util/util_env.h"
#include "../util/util_string.h"

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

}

extern "C" {

  // Interface for the game-side mod, which looks these up with GetProcAddress

  DLLEXPORT const char* __stdcall NFSU_GetVersion() {
    return dxvk::nfsu::g_version;
  }

}
