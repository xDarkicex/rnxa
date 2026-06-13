#!/usr/bin/env bash
# build_cuda_lib.sh — build libcuda.so (the CUDA shim that the
# purego CUDA backend loads at runtime).
#
# Output:
#   Linux:  internal/compute/cuda/build/libcuda.so
#
# The Go side looks for libcuda.so via LD_LIBRARY_PATH (or
# /usr/local/lib after `make install`).
#
# nvcc must be on $PATH. On most systems it lives at
# /usr/local/cuda/bin/nvcc, which the CUDA toolkit installer
# adds to .bashrc / .zshrc.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHIM_DIR="${SCRIPT_DIR}/../internal/compute/cuda"

make -C "${SHIM_DIR}" "${@}"

echo "Build complete:"
find "${SHIM_DIR}/build" -name "libcuda*" | head -3
