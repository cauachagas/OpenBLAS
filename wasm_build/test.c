#include <stdio.h>
#include <math.h>
#include <string.h>
#include "cblas.h"

/**
 * Minimal CBLAS test for WebAssembly
 * Tests: DAXPY, DDOT, DGEMM, DSCAL, DNRM2, DGEMV, DGER, DTRSM, DSYMM,
 *        ZAXPY, ZDOTC, ZDOTU, ZSCAL, ZGEMV, ZGEMM
 */

/* Helper: sum all elements of a double array */
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

/* Test 1: DAXPY - y := alpha*x + y */
EXPORTED
double test_daxpy(void) {
    int n = 5;
    double alpha = 2.0;
    double x[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    
    cblas_daxpy(n, alpha, x, 1, y, 1);
    
    /* Expected: y = [3.0, 5.0, 7.0, 9.0, 11.0] */
    double expected_sum = 35.0;  // 3+5+7+9+11
    double actual_sum = 0.0;
    for (int i = 0; i < n; i++) {
        actual_sum += y[i];
    }
    
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
    
    /* Expected: 1*5 + 2*6 + 3*7 + 4*8 = 5+12+21+32 = 70.0 */
    double expected = 70.0;
    
    printf("DDOT Test: Expected=%.1f, Got=%.1f\n", expected, result);
    return result;
}

/* Test 3: DGEMM - matrix multiplication C := alpha*A*B + beta*C */
EXPORTED
double test_dgemm(void) {
    int m = 2, n = 2, k = 2;
    
    /* A: 2x2 matrix */
    double A[4] = {1.0, 2.0,   /* row 0 */
                   3.0, 4.0};  /* row 1 */
    
    /* B: 2x2 matrix */
    double B[4] = {5.0, 6.0,   /* row 0 */
                   7.0, 8.0};  /* row 1 */
    
    /* C: 2x2 matrix (result) */
    double C[4] = {0.0, 0.0, 0.0, 0.0};
    
    double alpha = 1.0, beta = 0.0;
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, n, k, alpha, A, k, B, n, beta, C, n);
    
    /*
     * Expected result:
     * C[0,0] = 1*5 + 2*7 = 19
     * C[0,1] = 1*6 + 2*8 = 22
     * C[1,0] = 3*5 + 4*7 = 43
     * C[1,1] = 3*6 + 4*8 = 50
     */
    double expected_sum = 19.0 + 22.0 + 43.0 + 50.0;  /* 134.0 */
    double actual_sum = C[0] + C[1] + C[2] + C[3];
    
    printf("DGEMM Test: Expected sum=%.1f, Got sum=%.1f\n", expected_sum, actual_sum);
    printf("  C[0,0]=%.1f, C[0,1]=%.1f\n", C[0], C[1]);
    printf("  C[1,0]=%.1f, C[1,1]=%.1f\n", C[2], C[3]);
    
    return actual_sum;
}

/* Test 4: DSCAL - x := alpha*x */
EXPORTED
double test_dscal(void) {
    int n = 5;
    double alpha = 3.0;
    double x[5] = {1.0, 2.0, 3.0, 4.0, 5.0};

    cblas_dscal(n, alpha, x, 1);

    /* Expected: x = [3,6,9,12,15], sum = 45.0 */
    double expected_sum = 45.0;
    double actual_sum = 0.0;
    for (int i = 0; i < n; i++) actual_sum += x[i];

    printf("DSCAL Test: Expected sum=%.1f, Got sum=%.1f\n", expected_sum, actual_sum);
    return actual_sum;
}

/* Test 5: DNRM2 - Euclidean norm ||x||_2 */
EXPORTED
double test_dnrm2(void) {
    int n = 3;
    double x[3] = {2.0, 6.0, 3.0};

    /* sqrt(4 + 36 + 9) = sqrt(49) = 7.0 */
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

    /* Expected: y = [3, 7, 11], sum = 21.0 */
    double expected_sum = 21.0;
    double actual_sum = y[0] + y[1] + y[2];

    printf("DGEMV Test: Expected sum=%.1f, Got sum=%.1f\n", expected_sum, actual_sum);
    return actual_sum;
}

/* Test 7: DGER - A := alpha*x*y^T + A (rank-1 update) */
EXPORTED
double test_dger(void) {
    int m = 2, n = 3;
    double x[2] = {1.0, 2.0};
    double y[3] = {1.0, 2.0, 3.0};
    double A[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    cblas_dger(CblasRowMajor, m, n, 1.0, x, 1, y, 1, A, n);

    /*
     * Expected A:
     *   row 0: [1, 2, 3]
     *   row 1: [2, 4, 6]
     *   sum = 18.0
     */
    double expected_sum = 18.0;
    double actual_sum = 0.0;
    for (int i = 0; i < m * n; i++) actual_sum += A[i];

    printf("DGER Test: Expected sum=%.1f, Got sum=%.1f\n", expected_sum, actual_sum);
    return actual_sum;
}

/* Test 8: DTRSM - solve A*X = alpha*B, A lower triangular */
EXPORTED
double test_dtrsm(void) {
    /*
     * A = [[1, 0],     B = [3]
     *      [2, 1]]         [8]
     *
     * Solve A*X = B:
     *   x[0] = 3
     *   2*3 + x[1] = 8  =>  x[1] = 2
     * Expected sum = 5.0
     */
    double A[4] = {1.0, 0.0,
                   2.0, 1.0};
    double B[2] = {3.0, 8.0};

    cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                2, 1, 1.0, A, 2, B, 1);

    double expected_sum = 5.0;
    double actual_sum = B[0] + B[1];

    printf("DTRSM Test: Expected sum=%.1f, Got sum=%.1f\n", expected_sum, actual_sum);
    return actual_sum;
}

/* Test 9: DSYMM - C := alpha*A*B + beta*C, A symmetric */
EXPORTED
double test_dsymm(void) {
    /*
     * A = [[4, 2],   B = I_2x2
     *      [2, 3]]
     * C = A * I = A
     * sum = 4+2+2+3 = 11.0
     */
    double A[4] = {4.0, 2.0,
                   2.0, 3.0};
    double B[4] = {1.0, 0.0,
                   0.0, 1.0};
    double C[4] = {0.0, 0.0, 0.0, 0.0};

    cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper,
                2, 2, 1.0, A, 2, B, 2, 0.0, C, 2);

    double expected_sum = 11.0;
    double actual_sum = C[0] + C[1] + C[2] + C[3];

    printf("DSYMM Test: Expected sum=%.1f, Got sum=%.1f\n", expected_sum, actual_sum);
    return actual_sum;
}

/* Parameterized wrappers for interactive WebAssembly usage */
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
    if (!base || index < 0) {
        return 0;
    }

    base[index] = value;
    return 1;
}

EXPORTED
double wasm_get_f64(const double *base, int index) {
    if (!base || index < 0) {
        return 0.0;
    }

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
    if (!x || !result || !validate_blas_vector_args(n, incx)) {
        return 0;
    }

    *result = cblas_dnrm2(n, x, incx);
    return 1;
}

/* Test 10: SAXPY - y := alpha*x + y with strides */
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

/* Test 11: SDOT - dot product with strides */
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

/* Test 12: SNRM2 - Euclidean norm with strides */
EXPORTED
double test_snrm2(void) {
    int n = 2;
    float x[3] = {3.0f, 99.0f, 4.0f};
    float result = cblas_snrm2(n, x, 2);

    printf("SNRM2 Test: Expected=5.0, Got=%.1f, RelErr=%.3e\n",
           result, relative_error_scalar(5.0, result));
    return result;
}

/* Test 13: SSCAL - x := alpha*x with strides */
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

/* Test 14: SCOPY - y := x with strides */
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

/* Test 15: SSWAP - swap x and y with strides */
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

/* Test 16: DCOPY - y := x with strides */
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

/* Test 17: DSWAP - swap x and y with strides */
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

/* ===== Complex (single-precision) BLAS tests ===== */

/* Test 18: CAXPY - y := alpha*x + y with strides */
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

/* Test 19: CDOTC_SUB - conjugate dot product */
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

/* Test 20: CDOTU_SUB - unconjugated dot product */
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

/* Test 21: CSCAL - x := alpha*x */
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

/* Test 22: CCOPY - y := x with strides */
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

/* Test 23: CSWAP - swap x and y with strides */
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

/* ===== Complex (double-precision) BLAS tests ===== */
/*
 * Complex numbers are stored as interleaved real/imaginary pairs:
 *   double z[2*n] where z[2*i] = Re(z_i), z[2*i+1] = Im(z_i)
 */

/* Test 10: ZAXPY - y := alpha*x + y (complex Level 1) */
EXPORTED
double test_zaxpy(void) {
    int n = 3;
    /* alpha = 1+1i */
    double alpha[2] = {1.0, 1.0};
    /* x = [(1+0i), (0+1i), (1+1i)] */
    double x[6] = {1.0, 0.0,  0.0, 1.0,  1.0, 1.0};
    /* y = [(1+0i), (1+0i), (1+0i)] */
    double y[6] = {1.0, 0.0,  1.0, 0.0,  1.0, 0.0};

    cblas_zaxpy(n, alpha, x, 1, y, 1);

    /*
     * alpha*x[0] = (1+1i)*(1+0i) = 1+1i  => y[0] = 2+1i
     * alpha*x[1] = (1+1i)*(0+1i) = -1+1i => y[1] = 0+1i
     * alpha*x[2] = (1+1i)*(1+1i) = 0+2i  => y[2] = 1+2i
     * Sum of all Re/Im: 2+1 + 0+1 + 1+2 = 7.0
     */
    double expected_sum = 7.0;
    double actual_sum = sum_doubles(y, 6);

    printf("ZAXPY Test: Expected sum=%.1f, Got sum=%.1f\n", expected_sum, actual_sum);
    return actual_sum;
}

/* Test 11: ZDOTC_SUB - conjugate dot product: result = conj(x)·y (complex Level 1) */
EXPORTED
double test_zdotc(void) {
    int n = 2;
    /* x = [(1+2i), (3+4i)] */
    double x[4] = {1.0, 2.0,  3.0, 4.0};
    /* y = [(5+6i), (7+8i)] */
    double y[4] = {5.0, 6.0,  7.0, 8.0};
    double result[2] = {0.0, 0.0};

    cblas_zdotc_sub(n, x, 1, y, 1, result);

    /*
     * conj(x[0])*y[0] = (1-2i)*(5+6i) = 5+6i-10i+12 = 17-4i
     * conj(x[1])*y[1] = (3-4i)*(7+8i) = 21+24i-28i+32 = 53-4i
     * result = 70-8i  =>  Re(result) = 70.0
     */
    double expected = 70.0;
    printf("ZDOTC Test: Expected Re=%.1f, Got Re=%.1f\n", expected, result[0]);
    return result[0];
}

/* Test 12: ZDOTU_SUB - unconjugated dot product: result = x·y (complex Level 1) */
EXPORTED
double test_zdotu(void) {
    int n = 2;
    /* x = [(1+2i), (3+4i)] */
    double x[4] = {1.0, 2.0,  3.0, 4.0};
    /* y = [(5+6i), (7+8i)] */
    double y[4] = {5.0, 6.0,  7.0, 8.0};
    double result[2] = {0.0, 0.0};

    cblas_zdotu_sub(n, x, 1, y, 1, result);

    /*
     * x[0]*y[0] = (1+2i)*(5+6i) = 5+6i+10i-12 = -7+16i
     * x[1]*y[1] = (3+4i)*(7+8i) = 21+24i+28i-32 = -11+52i
     * result = -18+68i  =>  Im(result) = 68.0
     */
    double expected = 68.0;
    printf("ZDOTU Test: Expected Im=%.1f, Got Im=%.1f\n", expected, result[1]);
    return result[1];
}

/* Test 13: ZSCAL - x := alpha*x (complex Level 1) */
EXPORTED
double test_zscal(void) {
    int n = 3;
    /* alpha = 2+0i (pure real scalar) */
    double alpha[2] = {2.0, 0.0};
    /* x = [(1+1i), (2+3i), (0+1i)] */
    double x[6] = {1.0, 1.0,  2.0, 3.0,  0.0, 1.0};

    cblas_zscal(n, alpha, x, 1);

    /*
     * 2*(1+1i) = 2+2i
     * 2*(2+3i) = 4+6i
     * 2*(0+1i) = 0+2i
     * Sum of all Re/Im: 2+2+4+6+0+2 = 16.0
     */
    double expected_sum = 16.0;
    double actual_sum = sum_doubles(x, 6);

    printf("ZSCAL Test: Expected sum=%.1f, Got sum=%.1f\n", expected_sum, actual_sum);
    return actual_sum;
}

/* Test 24: ZCOPY - y := x with strides */
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

/* Test 25: ZSWAP - swap x and y with strides */
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

/* Test 14: ZGEMV - y := alpha*A*x + beta*y (complex Level 2) */
EXPORTED
double test_zgemv(void) {
    int m = 2, n = 2;
    /*
     * A (2x2 unitary, row-major interleaved):
     *   A[0,0]=1+0i, A[0,1]=0+1i
     *   A[1,0]=0-1i, A[1,1]=1+0i
     */
    double A[8] = {1.0, 0.0,  0.0, 1.0,
                   0.0,-1.0,  1.0, 0.0};
    /* x = [(1+0i), (1+0i)] */
    double x[4] = {1.0, 0.0,  1.0, 0.0};
    /* y = [(0+0i), (0+0i)] */
    double y[4] = {0.0, 0.0,  0.0, 0.0};
    /* alpha = 1+0i, beta = 0+0i */
    double alpha[2] = {1.0, 0.0};
    double beta[2]  = {0.0, 0.0};

    cblas_zgemv(CblasRowMajor, CblasNoTrans, m, n,
                alpha, A, n, x, 1, beta, y, 1);

    /*
     * y[0] = (1+0i)*(1+0i) + (0+1i)*(1+0i) = 1 + (0+1i) = 1+1i
     * y[1] = (0-1i)*(1+0i) + (1+0i)*(1+0i) = (0-1i) + 1 = 1-1i
     * Sum of all Re/Im: 1+1 + 1+(-1) = 2.0
     */
    double expected_sum = 2.0;
    double actual_sum = sum_doubles(y, 4);

    printf("ZGEMV Test: Expected sum=%.1f, Got sum=%.1f\n", expected_sum, actual_sum);
    return actual_sum;
}

/* Test 15: ZGEMM - C := alpha*A*B + beta*C (complex Level 3) */
EXPORTED
double test_zgemm(void) {
    int m = 2, n_mat = 2, k = 2;
    /*
     * A (2x2, row-major interleaved):
     *   A[0,0]=1+2i, A[0,1]=3+4i
     *   A[1,0]=5+6i, A[1,1]=7+8i
     */
    double A[8] = {1.0, 2.0,  3.0, 4.0,
                   5.0, 6.0,  7.0, 8.0};
    /* B = 2x2 identity (complex) */
    double B[8] = {1.0, 0.0,  0.0, 0.0,
                   0.0, 0.0,  1.0, 0.0};
    double C[8] = {0.0, 0.0,  0.0, 0.0,
                   0.0, 0.0,  0.0, 0.0};
    double alpha[2] = {1.0, 0.0};
    double beta[2]  = {0.0, 0.0};

    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n_mat, k, alpha, A, k, B, n_mat, beta, C, n_mat);

    /*
     * C = A * I = A
     * Sum of all Re/Im: 1+2+3+4+5+6+7+8 = 36.0
     */
    double expected_sum = 36.0;
    double actual_sum = sum_doubles(C, 8);

    printf("ZGEMM Test: Expected sum=%.1f, Got sum=%.1f\n", expected_sum, actual_sum);
    return actual_sum;
}

/* Main entry point for native builds */
int main(void) {
    printf("========== OpenBLAS CBLAS Test Suite ==========\n\n");

    double r1 = test_daxpy();  printf("  Result: %.1f\n\n", r1);
    double r2 = test_ddot();   printf("  Result: %.1f\n\n", r2);
    double r3 = test_dgemm();  printf("  Result: %.1f\n\n", r3);
    double r4 = test_dscal();  printf("  Result: %.1f\n\n", r4);
    double r5 = test_dnrm2();  printf("  Result: %.1f\n\n", r5);
    double r6 = test_dgemv();  printf("  Result: %.1f\n\n", r6);
    double r7 = test_dger();   printf("  Result: %.1f\n\n", r7);
    double r8 = test_dtrsm();  printf("  Result: %.1f\n\n", r8);
    double r9 = test_dsymm();  printf("  Result: %.1f\n\n", r9);

        printf("----- Testes Reais Nivel 1 (S) -----\n\n");
        double r10 = test_saxpy();  printf("  Result: %.1f\n\n", r10);
        double r11 = test_sdot();   printf("  Result: %.1f\n\n", r11);
        double r12 = test_snrm2();  printf("  Result: %.1f\n\n", r12);
        double r13 = test_sscal();  printf("  Result: %.1f\n\n", r13);
        double r14 = test_scopy();  printf("  Result: %.1f\n\n", r14);
        double r15 = test_sswap();  printf("  Result: %.1f\n\n", r15);

        printf("----- Testes Copy/Swap (D) -----\n\n");
        double r16 = test_dcopy();  printf("  Result: %.1f\n\n", r16);
        double r17 = test_dswap();  printf("  Result: %.1f\n\n", r17);

        printf("----- Testes Complexos Nivel 1 (C) -----\n\n");
        double r18 = test_caxpy();  printf("  Result: %.1f\n\n", r18);
        double r19 = test_cdotc();  printf("  Result: %.1f\n\n", r19);
        double r20 = test_cdotu();  printf("  Result: %.1f\n\n", r20);
        double r21 = test_cscal();  printf("  Result: %.1f\n\n", r21);
        double r22 = test_ccopy();  printf("  Result: %.1f\n\n", r22);
        double r23 = test_cswap();  printf("  Result: %.1f\n\n", r23);

    printf("----- Testes Complexos (Z) -----\n\n");
        double r24 = test_zaxpy();  printf("  Result: %.1f\n\n", r24);
        double r25 = test_zdotc();  printf("  Result: %.1f\n\n", r25);
        double r26 = test_zdotu();  printf("  Result: %.1f\n\n", r26);
        double r27 = test_zscal();  printf("  Result: %.1f\n\n", r27);
        double r28 = test_zcopy();  printf("  Result: %.1f\n\n", r28);
        double r29 = test_zswap();  printf("  Result: %.1f\n\n", r29);
        double r30 = test_zgemv();  printf("  Result: %.1f\n\n", r30);
        double r31 = test_zgemm();  printf("  Result: %.1f\n\n", r31);

    printf("============================================\n");
    printf("All tests completed!\n");

        return (scalar_within(35.0, r1, 1e-12)  && scalar_within(70.0, r2, 1e-12)  &&
            scalar_within(134.0, r3, 1e-12) && scalar_within(45.0, r4, 1e-12)  &&
            scalar_within(7.0, r5, 1e-12)   && scalar_within(21.0, r6, 1e-12)  &&
            scalar_within(18.0, r7, 1e-12)  && scalar_within(5.0, r8, 1e-12)   &&
            scalar_within(11.0, r9, 1e-12)  && scalar_within(15.0, r10, 1e-5)  &&
            scalar_within(32.0, r11, 1e-5)  && scalar_within(5.0, r12, 1e-5)   &&
            scalar_within(3.0, r13, 1e-5)   && scalar_within(6.0, r14, 1e-5)   &&
            scalar_within(15.0, r15, 1e-5)  && scalar_within(6.0, r16, 1e-12)  &&
            scalar_within(15.0, r17, 1e-12) && scalar_within(4.0, r18, 1e-5)   &&
            scalar_within(70.0, r19, 1e-5)  && scalar_within(68.0, r20, 1e-5)  &&
            scalar_within(14.0, r21, 1e-5)  && scalar_within(4.0, r22, 1e-5)   &&
            scalar_within(9.0, r23, 1e-5)   && scalar_within(7.0, r24, 1e-12)  &&
            scalar_within(70.0, r25, 1e-12) && scalar_within(68.0, r26, 1e-12) &&
            scalar_within(16.0, r27, 1e-12) && scalar_within(4.0, r28, 1e-12)  &&
            scalar_within(9.0, r29, 1e-12)  && scalar_within(2.0, r30, 1e-12)  &&
            scalar_within(36.0, r31, 1e-12)) ? 0 : 1;
}
