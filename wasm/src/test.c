#include <stdio.h>
#include <math.h>
#include <string.h>
#include "cblas.h"

/**
 * OpenBLAS CBLAS WebAssembly Numerical Test Suite & WASM Bindings
 *
 * Covers:
 * - Level 1: DAXPY, DDOT, DSCAL, DNRM2, DCOPY, DSWAP, IDAMAX
 *            SAXPY, SDOT, SNRM2, SSCAL, SCOPY, SSWAP, ISAMAX
 *            CAXPY, CDOTC, CDOTU, CSCAL, CCOPY, CSWAP
 *            ZAXPY, ZDOTC, ZDOTU, ZSCAL, ZCOPY, ZSWAP
 * - Level 2: DGEMV, DGER, SGEMV, SGER, ZGEMV
 * - Level 3: DGEMM, DTRSM, DSYMM, SGEMM, SSYMM, ZGEMM
 *
 * Parameterized Interactive Helpers:
 * - wasm_daxpy, wasm_ddot, wasm_dnrm2, wasm_set_f64, wasm_get_f64
 * - wasm_saxpy, wasm_sdot, wasm_set_f32, wasm_get_f32
 */

/* Helper: sum elements of array */
static double sum_doubles(const double *a, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += a[i];
    return s;
}

static double sum_floats(const float *a, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)a[i];
    return s;
}

static double relative_error_scalar(double expected, double actual) {
    double scale = fabs(expected);
    if (scale < 1e-12) scale = 1.0;
    return fabs(actual - expected) / scale;
}

static double relative_error_array_double(const double *expected, const double *actual, int n) {
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = actual[i] - expected[i];
        num += diff * diff;
        den += expected[i] * expected[i];
    }
    if (den < 1e-24) den = 1.0;
    return sqrt(num / den);
}

static double relative_error_array_float(const float *expected, const float *actual, int n) {
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = (double)actual[i] - (double)expected[i];
        num += diff * diff;
        den += (double)expected[i] * (double)expected[i];
    }
    if (den < 1e-24) den = 1.0;
    return sqrt(num / den);
}

static int scalar_within(double expected, double actual, double tol) {
    return relative_error_scalar(expected, actual) <= tol;
}

static int validate_blas_vector_args(int n, int inc) {
    return n > 0 && inc > 0;
}

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define EXPORTED EMSCRIPTEN_KEEPALIVE
#else
#define EXPORTED
#endif

/* ========================================================================= */
/* Double Precision (Level 1, 2, 3)                                          */
/* ========================================================================= */

/* Test 1: DAXPY - y := alpha*x + y */
EXPORTED
double test_daxpy(void) {
    int n = 5;
    double alpha = 2.0;
    double x[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    
    cblas_daxpy(n, alpha, x, 1, y, 1);
    
    double expected_sum = 35.0;  /* [3, 5, 7, 9, 11] => 35 */
    double actual_sum = sum_doubles(y, n);
    
    printf("DAXPY Test: Expected sum=%.1f, Got sum=%.1f\n", expected_sum, actual_sum);
    return actual_sum;
}

/* Test 2: DDOT - dot product */
EXPORTED
double test_ddot(void) {
    int n = 4;
    double x[4] = {1.0, 2.0, 3.0, 4.0};
    double y[4] = {5.0, 6.0, 7.0, 8.0};
    
    double result = cblas_ddot(n, x, 1, y, 1);
    double expected = 70.0;
    
    printf("DDOT Test: Expected=%.1f, Got=%.1f\n", expected, result);
    return result;
}

/* Test 3: DGEMM - matrix multiplication C := alpha*A*B + beta*C */
EXPORTED
double test_dgemm(void) {
    int m = 2, n = 2, k = 2;
    double A[4] = {1.0, 2.0,
                   3.0, 4.0};
    double B[4] = {5.0, 6.0,
                   7.0, 8.0};
    double C[4] = {0.0, 0.0, 0.0, 0.0};
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, n, k, 1.0, A, k, B, n, 0.0, C, n);
    
    double expected_sum = 134.0;
    double actual_sum = sum_doubles(C, 4);
    
    printf("DGEMM Test: Expected sum=%.1f, Got sum=%.1f\n", expected_sum, actual_sum);
    return actual_sum;
}

/* Test 4: DSCAL - x := alpha*x */
EXPORTED
double test_dscal(void) {
    int n = 5;
    double alpha = 3.0;
    double x[5] = {1.0, 2.0, 3.0, 4.0, 5.0};

    cblas_dscal(n, alpha, x, 1);

    double expected_sum = 45.0;
    double actual_sum = sum_doubles(x, n);

    printf("DSCAL Test: Expected sum=%.1f, Got sum=%.1f\n", expected_sum, actual_sum);
    return actual_sum;
}

/* Test 5: DNRM2 - Euclidean norm */
EXPORTED
double test_dnrm2(void) {
    int n = 3;
    double x[3] = {2.0, 6.0, 3.0};

    double result = cblas_dnrm2(n, x, 1);
    double expected = 7.0;

    printf("DNRM2 Test: Expected=%.1f, Got=%.1f\n", expected, result);
    return result;
}

/* Test 6: DGEMV - y := alpha*A*x + beta*y */
EXPORTED
double test_dgemv(void) {
    int m = 3, n = 2;
    double A[6] = {1.0, 2.0,
                   3.0, 4.0,
                   5.0, 6.0};
    double x[2] = {1.0, 1.0};
    double y[3] = {0.0, 0.0, 0.0};

    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, A, n, x, 1, 0.0, y, 1);

    double expected_sum = 21.0;
    double actual_sum = sum_doubles(y, 3);

    printf("DGEMV Test: Expected sum=%.1f, Got sum=%.1f\n", expected_sum, actual_sum);
    return actual_sum;
}

/* Test 7: DGER - A := alpha*x*y^T + A */
EXPORTED
double test_dger(void) {
    int m = 2, n = 3;
    double x[2] = {1.0, 2.0};
    double y[3] = {1.0, 2.0, 3.0};
    double A[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    cblas_dger(CblasRowMajor, m, n, 1.0, x, 1, y, 1, A, n);

    double expected_sum = 18.0;
    double actual_sum = sum_doubles(A, 6);

    printf("DGER Test: Expected sum=%.1f, Got sum=%.1f\n", expected_sum, actual_sum);
    return actual_sum;
}

/* Test 8: DTRSM - solve A*X = alpha*B */
EXPORTED
double test_dtrsm(void) {
    double A[4] = {1.0, 0.0,
                   2.0, 1.0};
    double B[2] = {3.0, 8.0};

    cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                2, 1, 1.0, A, 2, B, 1);

    double expected_sum = 5.0;
    double actual_sum = sum_doubles(B, 2);

    printf("DTRSM Test: Expected sum=%.1f, Got sum=%.1f\n", expected_sum, actual_sum);
    return actual_sum;
}

/* Test 9: DSYMM - C := alpha*A*B + beta*C */
EXPORTED
double test_dsymm(void) {
    double A[4] = {4.0, 2.0,
                   2.0, 3.0};
    double B[4] = {1.0, 0.0,
                   0.0, 1.0};
    double C[4] = {0.0, 0.0, 0.0, 0.0};

    cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper,
                2, 2, 1.0, A, 2, B, 2, 0.0, C, 2);

    double expected_sum = 11.0;
    double actual_sum = sum_doubles(C, 4);

    printf("DSYMM Test: Expected sum=%.1f, Got sum=%.1f\n", expected_sum, actual_sum);
    return actual_sum;
}

/* Test 10: DCOPY */
EXPORTED
double test_dcopy(void) {
    int n = 3;
    double x[5] = {1.0, 99.0, 2.0, 99.0, 3.0};
    double y[5] = {0.0, 88.0, 0.0, 88.0, 0.0};
    double expected[3] = {1.0, 2.0, 3.0};
    double actual[3];

    cblas_dcopy(n, x, 2, y, 2);

    actual[0] = y[0];
    actual[1] = y[2];
    actual[2] = y[4];

    printf("DCOPY Test: Expected sum=6.0, Got sum=%.1f, RelErr=%.3e\n",
           sum_doubles(actual, 3), relative_error_array_double(expected, actual, 3));
    return sum_doubles(actual, 3);
}

/* Test 11: DSWAP */
EXPORTED
double test_dswap(void) {
    int n = 3;
    double x[5] = {1.0, 99.0, 2.0, 99.0, 3.0};
    double y[5] = {4.0, 77.0, 5.0, 77.0, 6.0};
    double expected[3] = {4.0, 5.0, 6.0};
    double actual[3];

    cblas_dswap(n, x, 2, y, 2);

    actual[0] = x[0];
    actual[1] = x[2];
    actual[2] = x[4];

    printf("DSWAP Test: Expected sum=15.0, Got sum=%.1f, RelErr=%.3e\n",
           sum_doubles(actual, 3), relative_error_array_double(expected, actual, 3));
    return sum_doubles(actual, 3);
}

/* Test 12: IDAMAX - index of max absolute element (0-indexed in CBLAS) */
EXPORTED
double test_idamax(void) {
    int n = 4;
    double x[4] = {1.0, -7.5, 3.2, -2.1};
    CBLAS_INDEX idx = cblas_idamax(n, x, 1);
    /* Index of -7.5 is 1 */
    printf("IDAMAX Test: Expected=1.0, Got=%lu\n", (unsigned long)idx);
    return (double)idx;
}

/* ========================================================================= */
/* Single Precision (Level 1, 2, 3)                                          */
/* ========================================================================= */

/* Test 13: SAXPY */
EXPORTED
double test_saxpy(void) {
    int n = 3;
    float alpha = 2.0f;
    float x[5] = {1.0f, 99.0f, 2.0f, 99.0f, 3.0f};
    float y[5] = {1.0f, 88.0f, 1.0f, 88.0f, 1.0f};
    float expected[3] = {3.0f, 5.0f, 7.0f};
    float actual[3];

    cblas_saxpy(n, alpha, x, 2, y, 2);

    actual[0] = y[0];
    actual[1] = y[2];
    actual[2] = y[4];

    printf("SAXPY Test: Expected sum=15.0, Got sum=%.1f, RelErr=%.3e\n",
           sum_floats(actual, 3), relative_error_array_float(expected, actual, 3));
    return sum_floats(actual, 3);
}

/* Test 14: SDOT */
EXPORTED
double test_sdot(void) {
    int n = 3;
    float x[5] = {1.0f, 99.0f, 2.0f, 99.0f, 3.0f};
    float y[5] = {4.0f, 77.0f, 5.0f, 77.0f, 6.0f};
    float result = cblas_sdot(n, x, 2, y, 2);

    printf("SDOT Test: Expected=32.0, Got=%.1f, RelErr=%.3e\n",
           result, relative_error_scalar(32.0, result));
    return result;
}

/* Test 15: SNRM2 */
EXPORTED
double test_snrm2(void) {
    int n = 2;
    float x[3] = {3.0f, 99.0f, 4.0f};
    float result = cblas_snrm2(n, x, 2);

    printf("SNRM2 Test: Expected=5.0, Got=%.1f, RelErr=%.3e\n",
           result, relative_error_scalar(5.0, result));
    return result;
}

/* Test 16: SSCAL */
EXPORTED
double test_sscal(void) {
    int n = 3;
    float alpha = 0.5f;
    float x[5] = {1.0f, 99.0f, 2.0f, 99.0f, 3.0f};
    float expected[3] = {0.5f, 1.0f, 1.5f};
    float actual[3];

    cblas_sscal(n, alpha, x, 2);

    actual[0] = x[0];
    actual[1] = x[2];
    actual[2] = x[4];

    printf("SSCAL Test: Expected sum=3.0, Got sum=%.1f, RelErr=%.3e\n",
           sum_floats(actual, 3), relative_error_array_float(expected, actual, 3));
    return sum_floats(actual, 3);
}

/* Test 17: SCOPY */
EXPORTED
double test_scopy(void) {
    int n = 3;
    float x[5] = {1.0f, 99.0f, 2.0f, 99.0f, 3.0f};
    float y[5] = {0.0f, 88.0f, 0.0f, 88.0f, 0.0f};
    float expected[3] = {1.0f, 2.0f, 3.0f};
    float actual[3];

    cblas_scopy(n, x, 2, y, 2);

    actual[0] = y[0];
    actual[1] = y[2];
    actual[2] = y[4];

    printf("SCOPY Test: Expected sum=6.0, Got sum=%.1f, RelErr=%.3e\n",
           sum_floats(actual, 3), relative_error_array_float(expected, actual, 3));
    return sum_floats(actual, 3);
}

/* Test 18: SSWAP */
EXPORTED
double test_sswap(void) {
    int n = 3;
    float x[5] = {1.0f, 99.0f, 2.0f, 99.0f, 3.0f};
    float y[5] = {4.0f, 77.0f, 5.0f, 77.0f, 6.0f};
    float expected[3] = {4.0f, 5.0f, 6.0f};
    float actual[3];

    cblas_sswap(n, x, 2, y, 2);

    actual[0] = x[0];
    actual[1] = x[2];
    actual[2] = x[4];

    printf("SSWAP Test: Expected sum=15.0, Got sum=%.1f, RelErr=%.3e\n",
           sum_floats(actual, 3), relative_error_array_float(expected, actual, 3));
    return sum_floats(actual, 3);
}

/* Test 19: ISAMAX */
EXPORTED
double test_isamax(void) {
    int n = 4;
    float x[4] = {1.0f, 2.0f, -8.4f, 3.1f};
    CBLAS_INDEX idx = cblas_isamax(n, x, 1);
    /* Index of -8.4f is 2 */
    printf("ISAMAX Test: Expected=2.0, Got=%lu\n", (unsigned long)idx);
    return (double)idx;
}

/* Test 20: SGEMM */
EXPORTED
double test_sgemm(void) {
    int m = 2, n = 2, k = 2;
    float A[4] = {1.0f, 2.0f,
                  3.0f, 4.0f};
    float B[4] = {5.0f, 6.0f,
                  7.0f, 8.0f};
    float C[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0f, A, k, B, n, 0.0f, C, n);

    double expected_sum = 134.0;
    double actual_sum = sum_floats(C, 4);

    printf("SGEMM Test: Expected sum=%.1f, Got sum=%.1f\n", expected_sum, actual_sum);
    return actual_sum;
}

/* Test 21: SGEMV */
EXPORTED
double test_sgemv(void) {
    int m = 3, n = 2;
    float A[6] = {1.0f, 2.0f,
                  3.0f, 4.0f,
                  5.0f, 6.0f};
    float x[2] = {1.0f, 1.0f};
    float y[3] = {0.0f, 0.0f, 0.0f};

    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0f, A, n, x, 1, 0.0f, y, 1);

    double expected_sum = 21.0;
    double actual_sum = sum_floats(y, 3);

    printf("SGEMV Test: Expected sum=%.1f, Got sum=%.1f\n", expected_sum, actual_sum);
    return actual_sum;
}

/* Test 22: SGER */
EXPORTED
double test_sger(void) {
    int m = 2, n = 3;
    float x[2] = {1.0f, 2.0f};
    float y[3] = {1.0f, 2.0f, 3.0f};
    float A[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    cblas_sger(CblasRowMajor, m, n, 1.0f, x, 1, y, 1, A, n);

    double expected_sum = 18.0;
    double actual_sum = sum_floats(A, 6);

    printf("SGER Test: Expected sum=%.1f, Got sum=%.1f\n", expected_sum, actual_sum);
    return actual_sum;
}

/* Test 23: SSYMM */
EXPORTED
double test_ssymm(void) {
    float A[4] = {4.0f, 2.0f,
                  2.0f, 3.0f};
    float B[4] = {1.0f, 0.0f,
                  0.0f, 1.0f};
    float C[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    cblas_ssymm(CblasRowMajor, CblasLeft, CblasUpper,
                2, 2, 1.0f, A, 2, B, 2, 0.0f, C, 2);

    double expected_sum = 11.0;
    double actual_sum = sum_floats(C, 4);

    printf("SSYMM Test: Expected sum=%.1f, Got sum=%.1f\n", expected_sum, actual_sum);
    return actual_sum;
}

/* ========================================================================= */
/* Complex (Single Precision - C)                                            */
/* ========================================================================= */

/* Test 24: CAXPY */
EXPORTED
double test_caxpy(void) {
    int n = 2;
    float alpha[2] = {1.0f, 1.0f};
    float x[6] = {1.0f, 0.0f, 99.0f, 99.0f, 0.0f, 1.0f};
    float y[6] = {1.0f, 0.0f, 88.0f, 88.0f, 1.0f, 0.0f};
    float expected[4] = {2.0f, 1.0f, 0.0f, 1.0f};
    float actual[4];

    cblas_caxpy(n, alpha, x, 2, y, 2);

    actual[0] = y[0];
    actual[1] = y[1];
    actual[2] = y[4];
    actual[3] = y[5];

    printf("CAXPY Test: Expected sum=4.0, Got sum=%.1f, RelErr=%.3e\n",
           sum_floats(actual, 4), relative_error_array_float(expected, actual, 4));
    return sum_floats(actual, 4);
}

/* Test 25: CDOTC */
EXPORTED
double test_cdotc(void) {
    int n = 2;
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float result[2] = {0.0f, 0.0f};

    cblas_cdotc_sub(n, x, 1, y, 1, result);

    printf("CDOTC Test: Expected Re=70.0, Got Re=%.1f, RelErr=%.3e\n",
           result[0], relative_error_scalar(70.0, result[0]));
    return result[0];
}

/* Test 26: CDOTU */
EXPORTED
double test_cdotu(void) {
    int n = 2;
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float result[2] = {0.0f, 0.0f};

    cblas_cdotu_sub(n, x, 1, y, 1, result);

    printf("CDOTU Test: Expected Im=68.0, Got Im=%.1f, RelErr=%.3e\n",
           result[1], relative_error_scalar(68.0, result[1]));
    return result[1];
}

/* Test 27: CSCAL */
EXPORTED
double test_cscal(void) {
    int n = 2;
    float alpha[2] = {2.0f, 0.0f};
    float x[4] = {1.0f, 1.0f, 2.0f, 3.0f};
    float expected[4] = {2.0f, 2.0f, 4.0f, 6.0f};

    cblas_cscal(n, alpha, x, 1);

    printf("CSCAL Test: Expected sum=14.0, Got sum=%.1f, RelErr=%.3e\n",
           sum_floats(x, 4), relative_error_array_float(expected, x, 4));
    return sum_floats(x, 4);
}

/* Test 28: CCOPY */
EXPORTED
double test_ccopy(void) {
    int n = 2;
    float x[6] = {1.0f, 1.0f, 99.0f, 99.0f, 2.0f, 0.0f};
    float y[6] = {0.0f, 0.0f, 88.0f, 88.0f, 0.0f, 0.0f};
    float expected[4] = {1.0f, 1.0f, 2.0f, 0.0f};
    float actual[4];

    cblas_ccopy(n, x, 2, y, 2);

    actual[0] = y[0];
    actual[1] = y[1];
    actual[2] = y[4];
    actual[3] = y[5];

    printf("CCOPY Test: Expected sum=4.0, Got sum=%.1f, RelErr=%.3e\n",
           sum_floats(actual, 4), relative_error_array_float(expected, actual, 4));
    return sum_floats(actual, 4);
}

/* Test 29: CSWAP */
EXPORTED
double test_cswap(void) {
    int n = 2;
    float x[6] = {1.0f, 0.0f, 99.0f, 99.0f, 2.0f, 0.0f};
    float y[6] = {3.0f, 1.0f, 77.0f, 77.0f, 4.0f, 1.0f};
    float expected[4] = {3.0f, 1.0f, 4.0f, 1.0f};
    float actual[4];

    cblas_cswap(n, x, 2, y, 2);

    actual[0] = x[0];
    actual[1] = x[1];
    actual[2] = x[4];
    actual[3] = x[5];

    printf("CSWAP Test: Expected sum=9.0, Got sum=%.1f, RelErr=%.3e\n",
           sum_floats(actual, 4), relative_error_array_float(expected, actual, 4));
    return sum_floats(actual, 4);
}

/* ========================================================================= */
/* Complex (Double Precision - Z)                                            */
/* ========================================================================= */

/* Test 30: ZAXPY */
EXPORTED
double test_zaxpy(void) {
    int n = 3;
    double alpha[2] = {1.0, 1.0};
    double x[6] = {1.0, 0.0,  0.0, 1.0,  1.0, 1.0};
    double y[6] = {1.0, 0.0,  1.0, 0.0,  1.0, 0.0};

    cblas_zaxpy(n, alpha, x, 1, y, 1);

    double expected_sum = 7.0;
    double actual_sum = sum_doubles(y, 6);

    printf("ZAXPY Test: Expected sum=%.1f, Got sum=%.1f\n", expected_sum, actual_sum);
    return actual_sum;
}

/* Test 31: ZDOTC */
EXPORTED
double test_zdotc(void) {
    int n = 2;
    double x[4] = {1.0, 2.0,  3.0, 4.0};
    double y[4] = {5.0, 6.0,  7.0, 8.0};
    double result[2] = {0.0, 0.0};

    cblas_zdotc_sub(n, x, 1, y, 1, result);

    double expected = 70.0;
    printf("ZDOTC Test: Expected Re=%.1f, Got Re=%.1f\n", expected, result[0]);
    return result[0];
}

/* Test 32: ZDOTU */
EXPORTED
double test_zdotu(void) {
    int n = 2;
    double x[4] = {1.0, 2.0,  3.0, 4.0};
    double y[4] = {5.0, 6.0,  7.0, 8.0};
    double result[2] = {0.0, 0.0};

    cblas_zdotu_sub(n, x, 1, y, 1, result);

    double expected = 68.0;
    printf("ZDOTU Test: Expected Im=%.1f, Got Im=%.1f\n", expected, result[1]);
    return result[1];
}

/* Test 33: ZSCAL */
EXPORTED
double test_zscal(void) {
    int n = 3;
    double alpha[2] = {2.0, 0.0};
    double x[6] = {1.0, 1.0,  2.0, 3.0,  0.0, 1.0};

    cblas_zscal(n, alpha, x, 1);

    double expected_sum = 16.0;
    double actual_sum = sum_doubles(x, 6);

    printf("ZSCAL Test: Expected sum=%.1f, Got sum=%.1f\n", expected_sum, actual_sum);
    return actual_sum;
}

/* Test 34: ZCOPY */
EXPORTED
double test_zcopy(void) {
    int n = 2;
    double x[6] = {1.0, 1.0, 99.0, 99.0, 2.0, 0.0};
    double y[6] = {0.0, 0.0, 88.0, 88.0, 0.0, 0.0};
    double expected[4] = {1.0, 1.0, 2.0, 0.0};
    double actual[4];

    cblas_zcopy(n, x, 2, y, 2);

    actual[0] = y[0];
    actual[1] = y[1];
    actual[2] = y[4];
    actual[3] = y[5];

    printf("ZCOPY Test: Expected sum=4.0, Got sum=%.1f, RelErr=%.3e\n",
           sum_doubles(actual, 4), relative_error_array_double(expected, actual, 4));
    return sum_doubles(actual, 4);
}

/* Test 35: ZSWAP */
EXPORTED
double test_zswap(void) {
    int n = 2;
    double x[6] = {1.0, 0.0, 99.0, 99.0, 2.0, 0.0};
    double y[6] = {3.0, 1.0, 77.0, 77.0, 4.0, 1.0};
    double expected[4] = {3.0, 1.0, 4.0, 1.0};
    double actual[4];

    cblas_zswap(n, x, 2, y, 2);

    actual[0] = x[0];
    actual[1] = x[1];
    actual[2] = x[4];
    actual[3] = x[5];

    printf("ZSWAP Test: Expected sum=9.0, Got sum=%.1f, RelErr=%.3e\n",
           sum_doubles(actual, 4), relative_error_array_double(expected, actual, 4));
    return sum_doubles(actual, 4);
}

/* Test 36: ZGEMV */
EXPORTED
double test_zgemv(void) {
    int m = 2, n = 2;
    double A[8] = {1.0, 0.0,  0.0, 1.0,
                   0.0,-1.0,  1.0, 0.0};
    double x[4] = {1.0, 0.0,  1.0, 0.0};
    double y[4] = {0.0, 0.0,  0.0, 0.0};
    double alpha[2] = {1.0, 0.0};
    double beta[2]  = {0.0, 0.0};

    cblas_zgemv(CblasRowMajor, CblasNoTrans, m, n,
                alpha, A, n, x, 1, beta, y, 1);

    double expected_sum = 2.0;
    double actual_sum = sum_doubles(y, 4);

    printf("ZGEMV Test: Expected sum=%.1f, Got sum=%.1f\n", expected_sum, actual_sum);
    return actual_sum;
}

/* Test 37: ZGEMM */
EXPORTED
double test_zgemm(void) {
    int m = 2, n_mat = 2, k = 2;
    double A[8] = {1.0, 2.0,  3.0, 4.0,
                   5.0, 6.0,  7.0, 8.0};
    double B[8] = {1.0, 0.0,  0.0, 0.0,
                   0.0, 0.0,  1.0, 0.0};
    double C[8] = {0.0, 0.0,  0.0, 0.0,
                   0.0, 0.0,  0.0, 0.0};
    double alpha[2] = {1.0, 0.0};
    double beta[2]  = {0.0, 0.0};

    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n_mat, k, alpha, A, k, B, n_mat, beta, C, n_mat);

    double expected_sum = 36.0;
    double actual_sum = sum_doubles(C, 8);

    printf("ZGEMM Test: Expected sum=%.1f, Got sum=%.1f\n", expected_sum, actual_sum);
    return actual_sum;
}

/* ========================================================================= */
/* Parameterized Interactive Helpers (f64 and f32)                           */
/* ========================================================================= */

EXPORTED
int wasm_daxpy(int n, double alpha, double *x, int incx, double *y, int incy) {
    if (!x || !y || !validate_blas_vector_args(n, incx) || !validate_blas_vector_args(n, incy)) {
        return 0;
    }
    cblas_daxpy(n, alpha, x, incx, y, incy);
    return 1;
}

EXPORTED
int wasm_set_f64(double *base, int index, double value) {
    if (!base || index < 0) return 0;
    base[index] = value;
    return 1;
}

EXPORTED
double wasm_get_f64(const double *base, int index) {
    if (!base || index < 0) return 0.0;
    return base[index];
}

EXPORTED
int wasm_ddot(int n, const double *x, int incx, const double *y, int incy, double *result) {
    if (!x || !y || !result || !validate_blas_vector_args(n, incx) || !validate_blas_vector_args(n, incy)) {
        return 0;
    }
    *result = cblas_ddot(n, x, incx, y, incy);
    return 1;
}

EXPORTED
int wasm_dnrm2(int n, const double *x, int incx, double *result) {
    if (!x || !result || !validate_blas_vector_args(n, incx)) return 0;
    *result = cblas_dnrm2(n, x, incx);
    return 1;
}

EXPORTED
int wasm_saxpy(int n, float alpha, float *x, int incx, float *y, int incy) {
    if (!x || !y || !validate_blas_vector_args(n, incx) || !validate_blas_vector_args(n, incy)) {
        return 0;
    }
    cblas_saxpy(n, alpha, x, incx, y, incy);
    return 1;
}

EXPORTED
int wasm_sdot(int n, const float *x, int incx, const float *y, int incy, float *result) {
    if (!x || !y || !result || !validate_blas_vector_args(n, incx) || !validate_blas_vector_args(n, incy)) {
        return 0;
    }
    *result = cblas_sdot(n, x, incx, y, incy);
    return 1;
}

EXPORTED
int wasm_set_f32(float *base, int index, float value) {
    if (!base || index < 0) return 0;
    base[index] = value;
    return 1;
}

EXPORTED
float wasm_get_f32(const float *base, int index) {
    if (!base || index < 0) return 0.0f;
    return base[index];
}

/* ========================================================================= */
/* Native runner entrypoint                                                  */
/* ========================================================================= */

int main(void) {
    printf("========== OpenBLAS CBLAS Test Suite ==========\n\n");

    double r1 = test_daxpy();
    double r2 = test_ddot();
    double r3 = test_dgemm();
    double r4 = test_dscal();
    double r5 = test_dnrm2();
    double r6 = test_dgemv();
    double r7 = test_dger();
    double r8 = test_dtrsm();
    double r9 = test_dsymm();
    double r10 = test_dcopy();
    double r11 = test_dswap();
    double r12 = test_idamax();

    printf("\n----- Single Precision (S) -----\n");
    double r13 = test_saxpy();
    double r14 = test_sdot();
    double r15 = test_snrm2();
    double r16 = test_sscal();
    double r17 = test_scopy();
    double r18 = test_sswap();
    double r19 = test_isamax();
    double r20 = test_sgemm();
    double r21 = test_sgemv();
    double r22 = test_sger();
    double r23 = test_ssymm();

    printf("\n----- Complex Single Precision (C) -----\n");
    double r24 = test_caxpy();
    double r25 = test_cdotc();
    double r26 = test_cdotu();
    double r27 = test_cscal();
    double r28 = test_ccopy();
    double r29 = test_cswap();

    printf("\n----- Complex Double Precision (Z) -----\n");
    double r30 = test_zaxpy();
    double r31 = test_zdotc();
    double r32 = test_zdotu();
    double r33 = test_zscal();
    double r34 = test_zcopy();
    double r35 = test_zswap();
    double r36 = test_zgemv();
    double r37 = test_zgemm();

    printf("\n============================================\n");
    printf("All tests completed!\n");

    return (scalar_within(35.0, r1, 1e-12) && scalar_within(70.0, r2, 1e-12) &&
            scalar_within(134.0, r3, 1e-12) && scalar_within(45.0, r4, 1e-12) &&
            scalar_within(7.0, r5, 1e-12) && scalar_within(21.0, r6, 1e-12) &&
            scalar_within(18.0, r7, 1e-12) && scalar_within(5.0, r8, 1e-12) &&
            scalar_within(11.0, r9, 1e-12) && scalar_within(6.0, r10, 1e-12) &&
            scalar_within(15.0, r11, 1e-12) && scalar_within(1.0, r12, 1e-12) &&
            scalar_within(15.0, r13, 1e-5) && scalar_within(32.0, r14, 1e-5) &&
            scalar_within(5.0, r15, 1e-5) && scalar_within(3.0, r16, 1e-5) &&
            scalar_within(6.0, r17, 1e-5) && scalar_within(15.0, r18, 1e-5) &&
            scalar_within(2.0, r19, 1e-5) && scalar_within(134.0, r20, 1e-5) &&
            scalar_within(21.0, r21, 1e-5) && scalar_within(18.0, r22, 1e-5) &&
            scalar_within(11.0, r23, 1e-5) && scalar_within(4.0, r24, 1e-5) &&
            scalar_within(70.0, r25, 1e-5) && scalar_within(68.0, r26, 1e-5) &&
            scalar_within(14.0, r27, 1e-5) && scalar_within(4.0, r28, 1e-5) &&
            scalar_within(9.0, r29, 1e-5) && scalar_within(7.0, r30, 1e-12) &&
            scalar_within(70.0, r31, 1e-12) && scalar_within(68.0, r32, 1e-12) &&
            scalar_within(16.0, r33, 1e-12) && scalar_within(4.0, r34, 1e-12) &&
            scalar_within(9.0, r35, 1e-12) && scalar_within(2.0, r36, 1e-12) &&
            scalar_within(36.0, r37, 1e-12)) ? 0 : 1;
}
