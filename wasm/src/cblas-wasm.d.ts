/**
 * TypeScript definitions for OpenBLAS WebAssembly Module and Loader
 */

export interface OpenBlasWasmModule {
    HEAP8: Int8Array;
    HEAPU8: Uint8Array;
    HEAPF32: Float32Array;
    HEAPF64: Float64Array;
    wasmMemory: WebAssembly.Memory;

    _malloc(size: number): number;
    _free(ptr: number): number;

    // Double Precision (Level 1, 2, 3)
    _test_daxpy(): number;
    _test_ddot(): number;
    _test_dgemm(): number;
    _test_dscal(): number;
    _test_dnrm2(): number;
    _test_dgemv(): number;
    _test_dger(): number;
    _test_dtrsm(): number;
    _test_dsymm(): number;
    _test_dcopy(): number;
    _test_dswap(): number;
    _test_idamax(): number;

    // Single Precision (Level 1, 2, 3)
    _test_saxpy(): number;
    _test_sdot(): number;
    _test_snrm2(): number;
    _test_sscal(): number;
    _test_scopy(): number;
    _test_sswap(): number;
    _test_isamax(): number;
    _test_sgemm(): number;
    _test_sgemv(): number;
    _test_sger(): number;
    _test_ssymm(): number;

    // Complex Single Precision (C)
    _test_caxpy(): number;
    _test_cdotc(): number;
    _test_cdotu(): number;
    _test_cscal(): number;
    _test_ccopy(): number;
    _test_cswap(): number;

    // Complex Double Precision (Z)
    _test_zaxpy(): number;
    _test_zdotc(): number;
    _test_zdotu(): number;
    _test_zscal(): number;
    _test_zcopy(): number;
    _test_zswap(): number;
    _test_zgemv(): number;
    _test_zgemm(): number;

    // Interactive Wrappers
    _wasm_daxpy(n: number, alpha: number, x: number, incx: number, y: number, incy: number): number;
    _wasm_ddot(n: number, x: number, incx: number, y: number, incy: number, resultPtr: number): number;
    _wasm_dnrm2(n: number, x: number, incx: number, resultPtr: number): number;
    _wasm_set_f64(basePtr: number, index: number, value: number): number;
    _wasm_get_f64(basePtr: number, index: number): number;

    _wasm_saxpy(n: number, alpha: number, x: number, incx: number, y: number, incy: number): number;
    _wasm_sdot(n: number, x: number, incx: number, y: number, incy: number, resultPtr: number): number;
    _wasm_set_f32(basePtr: number, index: number, value: number): number;
    _wasm_get_f32(basePtr: number, index: number): number;
}

export interface VectorAllocation<T> {
    ptr: number;
    length: number;
    free(): void;
}

export interface OpenBlasApi {
    module: OpenBlasWasmModule;
    callTest(testName: string): number;
    daxpy(args: { n: number; alpha: number; x: number[]; incx?: number; y: number[]; incy?: number }): number[];
    ddot(args: { n: number; x: number[]; incx?: number; y: number[]; incy?: number }): number;
    dnrm2(args: { n: number; x: number[]; incx?: number }): number;
    saxpy(args: { n: number; alpha: number; x: number[]; incx?: number; y: number[]; incy?: number }): number[];
    sdot(args: { n: number; x: number[]; incx?: number; y: number[]; incy?: number }): number;
    allocFloat64(length: number, initialValues?: number[]): VectorAllocation<Float64Array>;
    allocFloat32(length: number, initialValues?: number[]): VectorAllocation<Float32Array>;
    hasExport(name: string): boolean;
}

export interface LoadOpenBlasOptions {
    print?: (text: string) => void;
    printErr?: (text: string) => void;
    onAbort?: (message: string) => void;
}

export function loadOpenBlas(options?: LoadOpenBlasOptions): Promise<OpenBlasApi>;
export default loadOpenBlas;
