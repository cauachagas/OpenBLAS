# Makefile for building OpenBLAS WebAssembly module, web distribution, and native test harness

ROOT_DIR    := $(abspath ..)
WASM_DIR    := $(ROOT_DIR)/wasm
SRC_DIR     := $(WASM_DIR)/src
WWW_DIR     := $(WASM_DIR)/www
LIB_DIR     := $(WASM_DIR)/lib
INC_DIR     := $(WASM_DIR)/include
DIST_DIR    := $(WASM_DIR)/dist

CC          := emcc
NATIVE_CC   := gcc

INCLUDE_FLAGS := -I$(INC_DIR) -I$(ROOT_DIR)
WASM_CFLAGS   := -O3 -ffast-math -msimd128 $(INCLUDE_FLAGS)
NATIVE_CFLAGS := -O3 -ffast-math $(INCLUDE_FLAGS)
LDFLAGS       := -L$(LIB_DIR) -lopenblas

POST_JS := $(SRC_DIR)/post.js

EXPORTED_FUNCTIONS := '[\
	"_test_daxpy","_test_ddot","_test_dgemm","_test_dscal","_test_dnrm2","_test_dgemv","_test_dger","_test_dtrsm","_test_dsymm",\
	"_test_saxpy","_test_sdot","_test_snrm2","_test_sscal","_test_scopy","_test_sswap",\
	"_test_dcopy","_test_dswap",\
	"_test_caxpy","_test_cdotc","_test_cdotu","_test_cscal","_test_ccopy","_test_cswap",\
	"_test_zaxpy","_test_zdotc","_test_zdotu","_test_zscal","_test_zcopy","_test_zswap","_test_zgemv","_test_zgemm",\
	"_test_sgemm","_test_sgemv","_test_sger","_test_ssymm","_test_idamax","_test_isamax",\
	"_wasm_daxpy","_wasm_ddot","_wasm_dnrm2","_wasm_set_f64","_wasm_get_f64",\
	"_wasm_saxpy","_wasm_sdot","_wasm_set_f32","_wasm_get_f32",\
	"_malloc","_free"\
]'

EXPORTED_RUNTIME_METHODS := '["ccall","cwrap","HEAP8","HEAPU8","HEAPF32","HEAPF64"]'

WASM_FLAGS := -s WASM=1 \
	-s MODULARIZE=1 \
	-s EXPORT_ES6=1 \
	-s ENVIRONMENT='web,worker,node' \
	-s EXPORTED_FUNCTIONS=$(EXPORTED_FUNCTIONS) \
	-s EXPORTED_RUNTIME_METHODS=$(EXPORTED_RUNTIME_METHODS) \
	-s ALLOW_MEMORY_GROWTH=1 \
	-s INITIAL_MEMORY=67108864

.PHONY: all lib wasm dist native serve clean distclean help

all: dist

# Compile OpenBLAS static library for WASM128_GENERIC
lib: $(LIB_DIR)/libopenblas.a

$(LIB_DIR)/libopenblas.a: scripts/build_openblas.sh
	@bash scripts/build_openblas.sh

# Compile WebAssembly binary and ES module factory
wasm: $(DIST_DIR)/test.wasm

$(DIST_DIR)/test.wasm: $(SRC_DIR)/test.c $(LIB_DIR)/libopenblas.a $(POST_JS)
	@mkdir -p $(DIST_DIR)
	$(CC) $(WASM_CFLAGS) $(WASM_FLAGS) --post-js $(POST_JS) -o $(DIST_DIR)/test.mjs $(SRC_DIR)/test.c $(LDFLAGS)
	@echo "✓ Generated $(DIST_DIR)/test.mjs and $(DIST_DIR)/test.wasm"

# Prepare full distributable bundle for GitHub Pages or local serving
dist: wasm
	@mkdir -p $(DIST_DIR)
	@cp -f $(SRC_DIR)/openblas-loader.js $(DIST_DIR)/
	@cp -f $(WWW_DIR)/index.html $(DIST_DIR)/
	@cp -f $(WWW_DIR)/index.js $(DIST_DIR)/
	@cp -f $(WWW_DIR)/interactive.html $(DIST_DIR)/
	@cp -f $(WWW_DIR)/interactive.js $(DIST_DIR)/
	@touch $(DIST_DIR)/.nojekyll
	@echo ""
	@echo "=========================================================="
	@echo "✓ WebAssembly bundle ready in $(DIST_DIR)!"
	@echo "  Run 'make serve' to launch local web server"
	@echo "=========================================================="

# Build native binary for local debugging without Emscripten runtime
native: $(WASM_DIR)/test_native

$(WASM_DIR)/test_native: $(SRC_DIR)/test.c $(LIB_DIR)/libopenblas.a
	$(NATIVE_CC) $(NATIVE_CFLAGS) -o $@ $(SRC_DIR)/test.c $(LDFLAGS) -lm
	@echo "✓ Native test executable ready: $(WASM_DIR)/test_native"

# Serve distribution folder locally
serve: dist
	@echo "Serving $(DIST_DIR) on http://localhost:8080..."
	@python3 -m http.server 8080 -d $(DIST_DIR)

# Clean compiled WebAssembly artifacts
clean:
	rm -rf $(DIST_DIR) $(WASM_DIR)/test_native
	@echo "Clean complete."

# Deep clean including compiled static library and extracted headers
distclean: clean
	rm -rf $(LIB_DIR) $(INC_DIR)
	@echo "Distclean complete."

# Help
help:
	@echo "OpenBLAS WebAssembly Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make lib        - Build libopenblas.a for WASM128_GENERIC"
	@echo "  make wasm       - Compile test.wasm and test.mjs into dist/"
	@echo "  make dist       - Build complete web distribution bundle (default)"
	@echo "  make native     - Compile native test executable for debugging"
	@echo "  make serve      - Serve dist/ locally on port 8080"
	@echo "  make clean      - Remove dist/ and test binaries"
	@echo "  make distclean  - Remove dist/, lib/, and include/ directories"
	@echo "  make help       - Show this help message"
