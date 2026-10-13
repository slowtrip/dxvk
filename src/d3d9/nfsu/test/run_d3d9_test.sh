#!/bin/sh
# Builds a native libdxvk_d3d9.so of this tree and runs nfsu_d3d9_test.cpp on lavapipe with the Khronos
# validation layer. Needs Xvfb, SDL2 development files and lvp_icd.json. Run from the repository root.
# Usage: run_d3d9_test.sh [variant bits] [frames]
set -e
[ -d build.native ] || meson setup build.native --buildtype debugoptimized \
  -Denable_d3d8=false -Denable_d3d10=false -Denable_d3d11=false -Denable_dxgi=false
ninja -C build.native
g++ -std=c++17 -O1 -g -o build.native/nfsu_d3d9_test src/d3d9/nfsu/test/nfsu_d3d9_test.cpp \
  -Iinclude/native/windows -Iinclude/native/directx -Iinclude/native $(pkg-config --cflags sdl2) \
  -Lbuild.native/src/d3d9 -ldxvk_d3d9 $(pkg-config --libs sdl2) -Wl,-rpath,"$PWD/build.native/src/d3d9"
ICD=/usr/share/vulkan/icd.d/lvp_icd.json
cd build.native
xvfb-run -a env DXVK_WSI_DRIVER=SDL2 VK_DRIVER_FILES=$ICD VK_ICD_FILENAMES=$ICD \
  VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation ./nfsu_d3d9_test "${1:-0}" "${2:-5}"
