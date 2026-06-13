#!/usr/bin/env bash
# build_cpu_lib.sh — build librnxa_cpu (the C++ shim over oneDNN that
# the purego CPU backend loads at runtime).
#
# Output:
#   Linux:  internal/compute/cpu_shim/build/librnxa_cpu.so
#   macOS:  internal/compute/cpu_shim/build/librnxa_cpu.dylib
#   Win:    internal/compute/cpu_shim/build/rnxa_cpu.dll
#
# The Go side looks for these via:
#   Linux:  LD_LIBRARY_PATH or /usr/local/lib
#   macOS:  DYLD_LIBRARY_PATH
#   Win:    PATH
# If you want a private install, copy the artifact to /usr/local/lib
# (or wherever your platform searches) and run `sudo ldconfig`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHIM_DIR="${SCRIPT_DIR}/../internal/compute/cpu_shim"

cmake -S "${SHIM_DIR}" -B "${SHIM_DIR}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    "${@}"

cmake --build "${SHIM_DIR}/build" --parallel

echo "Build complete:"
find "${SHIM_DIR}/build" -name "librnxa_cpu*" -o -name "rnxa_cpu.dll" | head -3
