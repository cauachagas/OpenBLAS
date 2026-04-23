/**
 * OpenBLAS CBLAS WebAssembly Module
 * TypeScript type definitions
 */

/**
 * CBLAS Function Enumerations
 */
export enum CBLASOrder {
    RowMajor = 101,
    ColMajor = 102
}

export enum CBLASTranspose {
    NoTrans = 111,
    Trans = 112,
    ConjTrans = 113
}

export enum CBLASUplo {
    Upper = 121,
    Lower = 122
}

export enum CBLASDiag {
    NonUnit = 131,
    Unit = 132
}

export enum CBLASSide {
    Left = 141,
    Right = 142
}

/**
 * WebAssembly Module Interface
 */
export interface CBlasWasmModule {
    /**
     * Test function: DAXPY
     * y = alpha * x + y
     * 
     * Tests with:
     * - alpha = 2.0
     * - x = [1.0, 2.0, 3.0, 4.0, 5.0]
     * - y = [1.0, 1.0, 1.0, 1.0, 1.0]
     * 
     * @returns Sum of result vector (expected: 35.0)
     */
    test_daxpy(): number;
    test_saxpy(): number;
    test_caxpy(): number;
    test_zaxpy(): number;

    /**
     * Test function: DDOT
     * Dot product: x · y
     * 
     * Tests with:
     * - x = [1.0, 2.0, 3.0, 4.0]
     * - y = [5.0, 6.0, 7.0, 8.0]
     * 
     * @returns Dot product (expected: 70.0)
     */
    test_ddot(): number;
    test_sdot(): number;
    test_cdotc(): number;
    test_cdotu(): number;
    test_zdotc(): number;
    test_zdotu(): number;

    /**
     * Test functions: NRM2
     */
    test_snrm2(): number;
    test_dnrm2(): number;

    /**
     * Test functions: SCAL
     */
    test_sscal(): number;
    test_dscal(): number;
    test_cscal(): number;
    test_zscal(): number;

    /**
     * Test functions: COPY
     */
    test_scopy(): number;
    test_dcopy(): number;
    test_ccopy(): number;
    test_zcopy(): number;

    /**
     * Test functions: SWAP
     */
    test_sswap(): number;
    test_dswap(): number;
    test_cswap(): number;
    test_zswap(): number;

    /**
     * Test functions: Level 2 and 3 double/complex-double coverage
     */
    test_dgemv(): number;
    test_dger(): number;
    test_dtrsm(): number;
    test_dsymm(): number;
    test_zgemv(): number;

    /**
     * Test function: DGEMM
     * General matrix multiply: C = alpha * A * B + beta * C
     * 
     * Tests with 2x2 matrices
     * 
     * @returns Sum of result matrix (expected: 134.0)
     */
    test_dgemm(): number;

    /**
     * Test function: ZGEMM  (Level 3 · Complex)
     * C = alpha * A * B + beta * C  (complex matrix multiply)
     *
     * A=2×2 complex, B=2×2 complex identity
     * @returns Sum of all Re+Im components of C (expected: 36.0)
     */
    test_zgemm(): number;

    /**
     * Memory allocator (from libc)
     * @param size Number of bytes to allocate
     * @returns Pointer to allocated memory
     */
    malloc(size: number): number;

    /**
     * Parameterized DAXPY wrapper for interactive WASM usage.
     * Returns 1 on success and 0 on invalid arguments.
     */
    wasm_daxpy(
        n: number,
        alpha: number,
        xPtr: number,
        incx: number,
        yPtr: number,
        incy: number
    ): number;

    /**
     * Parameterized DDOT wrapper for interactive WASM usage.
     */
    wasm_ddot(
        n: number,
        xPtr: number,
        incx: number,
        yPtr: number,
        incy: number,
        resultPtr: number
    ): number;

    /**
     * Parameterized DNRM2 wrapper for interactive WASM usage.
     */
    wasm_dnrm2(
        n: number,
        xPtr: number,
        incx: number,
        resultPtr: number
    ): number;

    /**
     * Memory deallocator (from libc)
     * @param ptr Pointer to memory to free
     */
    free(ptr: number): void;

    /**
     * Shared memory buffer
     */
    memory: WebAssembly.Memory;

    /**
     * Raw WebAssembly exports
     */
    [key: string]: any;
}

/**
 * CBLAS Wrapper for easier usage
 */
export interface CBlasLibrary {
    /**
     * Vector scaling and addition: y = alpha*x + y
     */
    daxpy(
        n: number,
        alpha: number,
        x: Float64Array,
        incx: number,
        y: Float64Array,
        incy: number
    ): void;

    /**
     * Dot product: x · y
     */
    ddot(
        n: number,
        x: Float64Array,
        incx: number,
        y: Float64Array,
        incy: number
    ): number;

    /**
     * General matrix multiply: C = alpha*A*B + beta*C
     */
    dgemm(
        order: CBLASOrder,
        transA: CBLASTranspose,
        transB: CBLASTranspose,
        m: number,
        n: number,
        k: number,
        alpha: number,
        A: Float64Array,
        lda: number,
        B: Float64Array,
        ldb: number,
        beta: number,
        C: Float64Array,
        ldc: number
    ): void;
}

/**
 * Load and initialize the CBLAS WASM module
 * 
 * @example
 * ```typescript
 * const blas = await loadCBLAS();
 * const result = blas.test_daxpy(); // 35.0
 * ```
 */
export function loadCBLAS(): Promise<CBlasWasmModule>;

/**
 * Create a typed wrapper around WASM module
 * Provides easier TypeScript interface
 * 
 * @param wasmModule The raw WASM module from loadCBLAS()
 * @returns Typed CBLAS interface
 */
export function createCBlasLibrary(
    wasmModule: CBlasWasmModule
): CBlasLibrary;

/**
 * Run test suite and return results
 */
export interface TestResults {
    daxpy: number;
    ddot: number;
    dgemm: number;
    passed: boolean;
    errors: string[];
}

export function runTestSuite(): Promise<TestResults>;

/**
 * Configuration options
 */
export interface CBlasConfig {
    wasmUrl?: string;
    workerUrl?: string;
    enableLogging?: boolean;
    timeoutMs?: number;
}

/**
 * Initialize with custom config
 */
export function initialize(config?: CBlasConfig): Promise<void>;
