#pragma once

/**
 * \brief Need for Speed Underground extensions
 *
 * Diagnostics and compute passes for the built-in renderer of
 * Need for Speed Underground. Game state stays in the game-side
 * mod, which drives the compute passes through the exported
 * NFSU_* functions, see d3d9_nfsu.cpp.
 */
namespace dxvk::nfsu {

  /**
   * \brief Writes a line to the NFSU log
   *
   * The log is independent of DXVK_LOG_LEVEL and goes to
   * scripts/logs/dxvk.log next to the game executable, or
   * to d3d9_nfsu.log if that folder cannot be created.
   */
  void Log(const char* fmt, ...);

  /**
   * \brief Called when the application creates a D3D9 interface
   */
  void OnCreate(bool extended);

}
