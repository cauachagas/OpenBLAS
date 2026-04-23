#!/bin/bash
# Script to build OpenBLAS for WebAssembly (WASM128_GENERIC) and prepare files for testing

set -e

echo "=========================================="
echo "Building OpenBLAS for WebAssembly (WASM128_GENERIC)"
echo "=========================================="

cd "$(dirname "$0")/.."

# Variation 1: Build static library (.a)
echo ""
echo "[1] Building static library (.a)"
make clean
make \
  CC=emcc \
  FC=emcc \
  HOSTCC=gcc \
  TARGET=WASM128_GENERIC \
  ONLY_CBLAS=1 \
  NOFORTRAN=2 \
  NO_LAPACK=1 \
  NO_LAPACKE=1 \
  C_LAPACK=0 \
  BUILD_WITHOUT_LAPACK=1 \
  USE_THREAD=0 \
  -j4

if [ -f "libopenblas.a" ]; then
  echo "✓ Static library generated: libopenblas.a"
  mkdir -p wasm_build/lib
  cp libopenblas.a wasm_build/lib/
else
  echo "✗ Failed to generate libopenblas.a"
  exit 1
fi

# Variation 2: Generate library.js + library.wasm for linking
echo ""
echo "[2] Preparing files for WebAssembly linking test"
mkdir -p wasm_build/include
cp cblas.h wasm_build/include/
cp common.h wasm_build/include/
cp config.h wasm_build/include/

echo ""
echo "=========================================="
echo "Build complete!"
echo "Generated files:"
echo "  - wasm_build/lib/libopenblas.a"
echo "  - wasm_build/include/cblas.h"
echo "  - wasm_build/include/config.h"
echo ""
echo "Next: Run 'make -C wasm_build wasm' to build the WASM test"
echo "=========================================="
