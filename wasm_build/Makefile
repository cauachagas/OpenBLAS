# Makefile for building OpenBLAS WebAssembly module and native test executable

ROOT := $(abspath ..)
CC := emcc
NATIVE_CC := gcc
INCLUDE_FLAGS := -I$(ROOT)/wasm_build/include -I$(ROOT)
WASM_CFLAGS := -O2 -ffast-math -msimd128 $(INCLUDE_FLAGS)
NATIVE_CFLAGS := -O2 -ffast-math $(INCLUDE_FLAGS)
LDFLAGS := -L$(ROOT)/wasm_build/lib -lopenblas
POST_JS := openblas-module-post.js
WASM_FLAGS := -s WASM=1 \
	-s MODULARIZE=1 \
	-s EXPORT_ES6=1 \
	-s ENVIRONMENT='web,worker' \
	-s EXPORTED_FUNCTIONS='["_test_daxpy","_test_ddot","_test_dgemm","_test_dscal","_test_dnrm2","_test_dgemv","_test_dger","_test_dtrsm","_test_dsymm","_test_saxpy","_test_sdot","_test_snrm2","_test_sscal","_test_scopy","_test_sswap","_test_dcopy","_test_dswap","_test_caxpy","_test_cdotc","_test_cdotu","_test_cscal","_test_ccopy","_test_cswap","_test_zaxpy","_test_zdotc","_test_zdotu","_test_zscal","_test_zcopy","_test_zswap","_test_zgemv","_test_zgemm","_wasm_daxpy","_wasm_ddot","_wasm_dnrm2","_wasm_set_f64","_wasm_get_f64","_malloc","_free"]' \
	-s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAP8","HEAPU8","HEAPF64"]' \
	-s ALLOW_MEMORY_GROWTH=1 \
	-s INITIAL_MEMORY=67108864

.PHONY: all clean test wasm native

all: wasm

# Build WebAssembly version
wasm: test.wasm index.html interactive_inputs.html openblas-loader.js index-page.js interactive_inputs.js
	@echo ""
	@echo "✓ WebAssembly build complete!"
	@echo "  - test.wasm (binary module)"
	@echo "  - test.mjs (ES module factory)"
	@echo "  - openblas-loader.js (stable browser wrapper)"
	@echo "  - index.html (fixed test page)"
	@echo "  - interactive_inputs.html (interactive page)"
	@echo ""
	@echo "To test: Serve this directory and open index.html or interactive_inputs.html"

test.wasm: test.c lib/libopenblas.a include/config.h include/cblas.h include/common.h
	$(CC) $(WASM_CFLAGS) $(WASM_FLAGS) --post-js $(POST_JS) -o test.mjs test.c $(LDFLAGS)
	@echo "Generated: test.mjs and test.wasm"

# Build native version (for debugging)
native: test_native
	@echo ""
	@echo "✓ Native build complete: ./test_native"
	@echo "Run: ./test_native"

test_native: test.c lib/libopenblas.a
	$(NATIVE_CC) $(NATIVE_CFLAGS) -o test_native test.c $(LDFLAGS) -lm

# Clean build artifacts
clean:
	rm -f test.js test.mjs test.wasm test_native test.c.o
	@echo "Clean complete"

# Help
help:
	@echo "OpenBLAS WASM Build Targets"
	@echo ""
	@echo "  make              - Build WebAssembly (wasm)"
	@echo "  make wasm         - Build WASM version"
	@echo "  make native       - Build native version for debugging"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make help         - Show this help"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - libopenblas.a should be in ./lib/"
	@echo "  - cblas.h should be in ./include/"
