# OpenBLAS WebAssembly (WASM)

This directory provides the infrastructure, test suite, and interactive web demonstration for running OpenBLAS compiled to WebAssembly with SIMD128 support (`TARGET=WASM128_GENERIC`).

---

## Directory Structure

```text
wasm/
├── Makefile                 # Top-level build orchestration (build, dist, serve, clean)
├── README.md                # Build instructions and architectural guide
├── .gitignore               # Ignores intermediate libraries and distribution output
├── scripts/
│   └── build_openblas.sh    # Compiles OpenBLAS static library (libopenblas.a) with Emscripten
├── src/
│   ├── test.c               # Numerical CBLAS test suite and WASM export bindings
│   ├── openblas-loader.js   # High-level ES module loader and memory allocation API
│   ├── post.js              # Emscripten post-js hook (heap views & memory access)
│   └── cblas-wasm.d.ts      # TypeScript definitions for the WebAssembly module
└── www/
    ├── index.html           # Automated test dashboard & numerical precision verification
    ├── index.js             # Test runner logic and tolerance validation
    ├── interactive.html     # Interactive BLAS playground (real-time vector & matrix operations)
    └── interactive.js       # Dynamic UI handlers for interactive playground
```

---

## Prerequisites

1. **Emscripten SDK (`emsdk`)**:
   Ensure `emsdk` is installed and activated in your environment:
   ```bash
   ./emsdk activate latest
   source ./emsdk_env.sh
   ```

2. **Standard Build Tools**:
   - `make`, `gcc` (host compiler for `getarch`)
   - `python3` (for local HTTP testing via `make serve`)

---

## Quick Start

### 1. Build the OpenBLAS static library for WASM
```bash
make -C wasm lib
```
This runs `wasm/scripts/build_openblas.sh` which executes OpenBLAS's makefile with:
```bash
make HOSTCC=gcc CC=emcc AR=emar RANLIB=emranlib \
     TARGET=WASM128_GENERIC USE_THREAD=0 NOFORTRAN=1 \
     NO_LAPACK=1 NO_LAPACKE=1 C_LAPACK=0 BUILD_WITHOUT_LAPACK=1
```
The resulting `libopenblas.a` is copied to `wasm/lib/`, and essential headers (`cblas.h`, `common.h`, `config.h`) are copied to `wasm/include/`.

### 2. Build the WebAssembly bundle
```bash
make -C wasm dist
```
This compiles `wasm/src/test.c` linked with `libopenblas.a` producing:
- `wasm/dist/test.wasm` (binary module)
- `wasm/dist/test.mjs` (ES6 module factory)
- Ready-to-serve HTML, JS, and loader assets in `wasm/dist/`.

### 3. Run and test locally
```bash
make -C wasm serve
```
Open `http://localhost:8080/` for the automated test dashboard or `http://localhost:8080/interactive.html` for the interactive playground.

---

## GitHub Actions & GitHub Pages

The workflow `.github/workflows/wasm-pages.yml` triggers on changes to `wasm/**` or core build files and automatically builds and deploys `wasm/dist` to GitHub Pages.
