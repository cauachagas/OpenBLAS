#!/usr/bin/env bash
# Script to build OpenBLAS static library for WebAssembly (TARGET=WASM128_GENERIC)
# Aligned with official OpenBLAS WASM compilation best practices.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WASM_DIR="${ROOT_DIR}/wasm"

# Auto-detect emsdk if not already in PATH
if ! command -v emcc >/dev/null 2>&1; then
  if [ -n "${EMSDK}" ] && [ -f "${EMSDK}/emsdk_env.sh" ]; then
    echo "==> Activating Emscripten from EMSDK (${EMSDK})..."
    # shellcheck disable=SC1091
    source "${EMSDK}/emsdk_env.sh" >/dev/null 2>&1
  else
    echo "Error: emcc not found in PATH and emsdk could not be located." >&2
    echo "Please activate emsdk first:" >&2
    echo "  source /path/to/emsdk/emsdk_env.sh" >&2
    exit 1
  fi
fi

# Verify required tools
for tool in emcc emar emranlib; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "Error: Required Emscripten tool '${tool}' was not found in PATH." >&2
    exit 1
  fi
done

echo "=========================================================="
echo " Building OpenBLAS for WebAssembly (WASM128_GENERIC)"
echo " Compiler: $(emcc --version | head -n 1)"
echo "=========================================================="

cd "${ROOT_DIR}"

NPROC="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

# Clean any previous artifacts
echo ""
echo "[1/3] Cleaning previous build..."
make clean >/dev/null 2>&1 || true

# Build static library with canonical WASM flags
echo ""
echo "[2/3] Compiling libopenblas.a (jobs: ${NPROC})..."
make \
  HOSTCC=gcc \
  CC=emcc \
  AR=emar \
  RANLIB=emranlib \
  TARGET=WASM128_GENERIC \
  USE_THREAD=0 \
  NOFORTRAN=1 \
  NO_LAPACK=1 \
  NO_LAPACKE=1 \
  C_LAPACK=0 \
  BUILD_WITHOUT_LAPACK=1 \
  -j"${NPROC}"

if [ ! -f "libopenblas.a" ]; then
  echo "Error: Failed to generate libopenblas.a" >&2
  exit 1
fi

echo ""
echo "[3/3] Exporting static library and headers to wasm/..."
mkdir -p "${WASM_DIR}/lib" "${WASM_DIR}/include"

cp libopenblas.a "${WASM_DIR}/lib/"
cp cblas.h common.h config.h "${WASM_DIR}/include/"

echo ""
echo "=========================================================="
echo " Build successful!"
echo " Outputs:"
echo "   - ${WASM_DIR}/lib/libopenblas.a"
echo "   - ${WASM_DIR}/include/{cblas.h, common.h, config.h}"
echo " Next: run 'make -C wasm dist' to build WebAssembly bundle"
echo "=========================================================="
