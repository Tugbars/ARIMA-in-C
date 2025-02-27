#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

/*==================== Defined Constants ====================*/
#define CONVERGENCE_TOLERANCE 0.01
#define MIN_ITERATIONS 30
#define UNIT_TOLERANCE 1.001

#define MAX_ITERATIONS 300

// Numerical threshold for detecting near-zero pivot elements in LU decomposition.
#define SINGULARITY_THRESHOLD 1e-12

// Forecasting horizon: number of forecast steps (excluding additional summary metrics).
#define FORECAST_HORIZON 16

// Total size for forecast arrays: forecast horizon plus extra space for summary metrics (e.g., MAPE, error variance).
#define FORECAST_ARRAY_SIZE (FORECAST_HORIZON + 2)

// Critical value for the augmented Dickey–Fuller (ADF) test.
// In this simplified test, if the estimated coefficient on the lagged level is below this value, we reject the unit root.
#define ADF_CRITICAL_VALUE -3.5

// Exponent used in computing the lag order for the ADF test (e.g., p = floor((n-1)^(1/3)) ).
#define ADF_LAG_EXPONENT (1.0 / 3.0)

#define INITIAL_MA_LEARNING_RATE 0.001
#define MIN_MA_LEARNING_RATE 1e-6


#ifdef DEBUG
#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define DEBUG_PRINT(...)
#endif

/*==================== Data Structures =====================*/
/**
 * @brief Structure to hold ARMA parameters.
 */
typedef struct
{
  double ar[3];     // AR coefficients (if needed)
  double ma[3];     // MA coefficients (for example, theta values)
  double intercept; // Constant term (for AR or MA part)
} ARMA_Params;

/*========================================================================
  Utility Functions: Array and Matrix Operations
========================================================================*/

/**
 * @brief Calculates the mean (average) of an array.
 *
 * @param array The input array of doubles.
 * @param length The number of elements in the array.
 * @return The computed mean.
 */
double calculateMean(const double array[], int length)
{
  double sum = 0.0;
  for (int i = 0; i < length; i++)
  {
    sum += array[i];
  }
  return sum / length;
}

/**
 * @brief Finds the index of the minimum value in an integer array.
 *
 * @param array The input array.
 * @param size The number of elements.
 * @return The index of the smallest element.
 */
int findIndexOfMin(const int array[], int size)
{
  int minValue = array[0];
  int minIndex = 0;
  for (int i = 1; i < size; i++)
  {
    if (array[i] < minValue)
    {
      minValue = array[i];
      minIndex = i;
    }
  }
  return minIndex;
}

/**
 * @brief Copies the contents of one array into another.
 *
 * @param source The array to copy from.
 * @param destination The array to copy to.
 * @param length The number of elements to copy.
 */
void copyArray(const double source[], double destination[], int length)
{
  for (int i = 0; i < length; i++)
  {
    destination[i] = source[i];
  }
}

/**
 * @brief Returns a new array where each element is the square of the corresponding element in the input.
 *
 * @param input The input array.
 * @param size The number of elements.
 * @return Pointer to a dynamically allocated array with squared values.
 *
 * @note Caller is responsible for freeing the returned memory.
 */
double *squareArray(const double input[], int size)
{
  double *squared = malloc(sizeof(double) * size);
  if (!squared)
  {
    fprintf(stderr, "Error allocating memory in squareArray.\n");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < size; i++)
  {
    squared[i] = input[i] * input[i];
  }
  return squared;
}

/**
 * @brief Computes the sum of all elements in an array.
 *
 * @param array The input array.
 * @param length The number of elements.
 * @return The sum.
 */
double calculateArraySum(const double array[], int length)
{
  double sum = 0.0;
  for (int i = 0; i < length; i++)
  {
    sum += array[i];
  }
  return sum;
}

/**
 * @brief Computes the element‐wise difference between two arrays.
 *
 * @param array1 The first array.
 * @param array2 The second array.
 * @param difference Output array where each element is (array1[i] - array2[i]).
 * @param size The number of elements.
 */
void calculateArrayDifference(const double array1[], const double array2[], double difference[], int size)
{
  for (int i = 0; i < size; i++)
  {
    difference[i] = array1[i] - array2[i];
  }
}

/**
 * @brief Computes the element‐wise product of two arrays.
 *
 * @param array1 The first array.
 * @param array2 The second array.
 * @param size The number of elements.
 * @return Pointer to a dynamically allocated array containing the product.
 *
 * @note Caller must free the returned array.
 */
double *calculateElementwiseProduct(const double array1[], const double array2[], int size)
{
  double *product = malloc(sizeof(double) * size);
  if (!product)
  {
    fprintf(stderr, "Error allocating memory in calculateElementwiseProduct.\n");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < size; i++)
  {
    product[i] = array1[i] * array2[i];
  }
  return product;
}

/**
 * @brief Calculates the sample standard deviation of an array.
 *
 * @param array The input array.
 * @param length The number of elements.
 * @return The standard deviation.
 */
double calculateStandardDeviation(const double array[], int length)
{
  double meanValue = calculateMean(array, length);
  double sumSqDiff = 0.0;
  for (int i = 0; i < length; i++)
  {
    double diff = array[i] - meanValue;
    sumSqDiff += diff * diff;
  }
  double variance = sumSqDiff / (length - 1);
  return sqrt(variance);
}

/**
 * @brief Generates a "lead" version of an array by shifting forward.
 *
 * @param array The original array.
 * @param leadArray Output array where leadArray[i] = array[i + lag].
 * @param length The number of elements in the original array.
 * @param lag The number of positions to shift.
 */
void calculateLead(const double array[], double leadArray[], int length, int lag)
{
  for (int i = 0; i < (length - lag); i++)
  {
    leadArray[i] = array[i + lag];
  }
}

/**
 * @brief Generates a "lagged" version of an array.
 *
 * @param array The original array.
 * @param lagArray Output array (the first (length-lag) elements are copied).
 * @param length The number of elements.
 * @param lag The lag amount.
 */
void calculateLag(const double array[], double lagArray[], int length, int lag)
{
  for (int i = 0; i < (length - lag); i++)
  {
    lagArray[i] = array[i];
  }
}

/**
 * @brief Generates a new array shifted by a specified lag (useful for lead-lag operations).
 *
 * @param array The original array.
 * @param leadArray Output array shifted by lag.
 * @param length The number of elements.
 * @param lead The lead count.
 * @param lag The lag offset.
 */
void calculateLeadWithLag(const double array[], double leadArray[], int length, int lead, int lag)
{
  for (int i = 0; i < (length - lead - lag); i++)
  {
    leadArray[i] = array[i + lag];
  }
}

/**
 * @brief Normalizes a 2D array by subtracting the column means.
 *
 * @param numRows The number of rows.
 * @param numCols The number of columns.
 * @param matrix The input 2D array.
 * @param normalized Output normalized array.
 */
void normalize2DArray(int numRows, int numCols, double matrix[][numCols], double normalized[][numCols])
{
  double columnMeans[1][numCols];
  for (int j = 0; j < numCols; j++)
  {
    double col[numRows];
    for (int i = 0; i < numRows; i++)
    {
      col[i] = matrix[i][j];
    }
    columnMeans[0][j] = calculateMean(col, numRows);
  }
  for (int i = 0; i < numRows; i++)
  {
    for (int j = 0; j < numCols; j++)
    {
      normalized[i][j] = matrix[i][j] - columnMeans[0][j];
    }
  }
}

/**
 * @brief Transposes a matrix.
 *
 * @param numRows Number of rows in the input matrix.
 * @param numCols Number of columns in the input matrix.
 * @param matrix The input matrix.
 * @param transposed Output matrix that is the transpose.
 */
void transposeMatrix(int numRows, int numCols, double matrix[][numCols], double transposed[][numRows])
{
  for (int i = 0; i < numCols; i++)
  {
    for (int j = 0; j < numRows; j++)
    {
      transposed[i][j] = matrix[j][i];
    }
  }
}

/**
 * @brief Multiplies two matrices.
 *
 * @param rowsA Number of rows in matrix A.
 * @param colsA Number of columns in matrix A (and rows in matrix B).
 * @param colsB Number of columns in matrix B.
 * @param A Matrix A.
 * @param B Matrix B.
 * @param result Output matrix (dimensions: rowsA x colsB).
 */
void matrixMultiply(int rowsA, int colsA, int colsB, double A[][colsA], double B[][colsB], double result[][colsB])
{
  for (int i = 0; i < rowsA; i++)
  {
    for (int j = 0; j < colsB; j++)
    {
      double sum = 0.0;
      for (int k = 0; k < colsA; k++)
      {
        sum += A[i][k] * B[k][j];
      }
      result[i][j] = sum;
    }
  }
}

/*--- QR Inversion Code Begin ---*/

/** @def IDX(i,j,ld)
 *  @brief A macro to index a 2D array stored in column‐major order.
 *
 *  @param i Row index.
 *  @param j Column index.
 *  @param ld Leading dimension (the number of rows).
 *
 *  This macro returns the 1D offset in the column‐major array for the element
 *  at \c (i,j). In column‐major order, elements of a column are consecutive in memory,
 *  so the offset is \c (i + j*ld).
 */
#define IDX(i,j,ld) ((i) + (j)*(ld))

/**
 * @brief Performs a rank–revealing blocked QR decomposition with column pivoting
 *        and refined norm updates (including reorthogonalization passes).
 *
 * @details
 * Given an \c m×n matrix \c A (stored in column‐major order), this function computes
 * a QR factorization \f$A P = Q R\f$ (with column pivoting) in-place. The final matrix \c A
 * contains:
 *   - The upper triangle of \c A corresponds to the \f$R\f$ factor.
 *   - The lower portion (and the first element in each Householder vector implicitly \c 1)
 *     encodes the Householder vectors used to form \f$Q\f$ implicitly.
 *
 * Column pivoting is used to enhance numerical stability and reveal rank deficiency by
 * swapping columns based on the largest updated column norms. The block updates improve
 * cache efficiency and can provide better performance on larger matrices.
 *
 * **Algorithm Sketch:**
 * - For each column \c k (pivot), find the column among \c [k..n-1] with the largest
 *   updated norm and swap it into position \c k.
 * - Compute a Householder reflector for column \c k that zeroes out all elements below
 *   the diagonal. Store the reflector in \c A.
 * - Apply the reflector to the trailing submatrix in a *blocked* manner (up to \c block_size
 *   columns at a time), and perform a “reorthogonalization” pass to reduce numerical errors.
 * - Update each trailing column’s norm estimate, and if it has dropped significantly,
 *   recompute the norm to avoid error accumulation.
 *
 * @param[in]  m            Number of rows of \c A.
 * @param[in]  n            Number of columns of \c A.
 * @param[in,out] A         Pointer to the \c m×n matrix in column‐major order. On exit, the upper
 *                          triangle contains \f$R\f$, and the lower part encodes Householder vectors.
 * @param[in]  lda          Leading dimension (must be \c >= m).
 * @param[out] jpvt         An integer array of length \c n. On exit, \c jpvt[k] is the column index
 *                          of the \c k–th pivot, indicating how columns were permuted.
 * @param[in]  block_size   The block size for updating the trailing submatrix. For small problems,
 *                          \c 1 is often sufficient; for larger problems, a bigger block size may
 *                          improve performance.
 *
 * @pre \c A must be a valid pointer of size at least \c lda*n, storing the matrix in column‐major format.
 * @pre \c jpvt must be a valid pointer to an integer array of length \c n.
 *
 * @note If \c block_size is too large (bigger than \c n–k), it is automatically reduced in each iteration.
 *
 * @warning If \c A has many zero rows or if it is numerically rank‐deficient, the computed
 *          reflectors might produce very small pivots in the trailing columns. This is precisely
 *          why column pivoting is recommended, so that small pivots appear in later columns.
 *
 * **Reference**: G. W. Stewart, *Matrix Algorithms, Volume 1: Basic Decompositions*, SIAM, 1998.
 */
void qr_decomp_colpivot_blocked(int m, int n, double *A, int lda, int *jpvt, int block_size) {
    int i, j, k, nb;

    double *norms = malloc(n * sizeof(double));
    double *norms_updated = malloc(n * sizeof(double));
    if (!norms || !norms_updated) {
        fprintf(stderr, "Memory allocation error in qr_decomp_colpivot_blocked.\n");
        exit(EXIT_FAILURE);
    }

    for (j = 0; j < n; j++) {
        jpvt[j] = j;
        double sum = 0.0;
        for (i = 0; i < m; i++) {
            double aij = A[IDX(i,j,lda)];
            sum += aij * aij;
        }
        norms[j] = sqrt(sum);
        norms_updated[j] = norms[j];
    }

    for (k = 0; k < n && k < m; k++) {
        // Column pivoting: find column with max updated norm.
        int max_index = k;
        for (j = k; j < n; j++) {
            if (norms_updated[j] > norms_updated[max_index])
                max_index = j;
        }
        if (max_index != k) {
            for (i = 0; i < m; i++) {
                double tmp = A[IDX(i, k, lda)];
                A[IDX(i, k, lda)] = A[IDX(i, max_index, lda)];
                A[IDX(i, max_index, lda)] = tmp;
            }
            int tmp_int = jpvt[k];
            jpvt[k] = jpvt[max_index];
            jpvt[max_index] = tmp_int;

            double tmp_norm = norms_updated[k];
            norms_updated[k] = norms_updated[max_index];
            norms_updated[max_index] = tmp_norm;
            tmp_norm = norms[k];
            norms[k] = norms[max_index];
            norms[max_index] = tmp_norm;
        }
        // Compute Householder vector for column k.
        double norm_x = 0.0;
        for (i = k; i < m; i++) {
            norm_x += A[IDX(i, k, lda)] * A[IDX(i, k, lda)];
        }
        norm_x = sqrt(norm_x);
        if (norm_x == 0.0) {
            A[IDX(k, k, lda)] = 0.0;
            continue;
        }
        double sign = (A[IDX(k, k, lda)] >= 0) ? -1.0 : 1.0;
        double *v = malloc((m - k) * sizeof(double));
        if (!v) {
            fprintf(stderr, "Memory allocation error in qr_decomp_colpivot_blocked (v).\n");
            exit(EXIT_FAILURE);
        }
        v[0] = A[IDX(k, k, lda)] - sign * norm_x;
        for (i = k + 1; i < m; i++) {
            v[i - k] = A[IDX(i, k, lda)];
        }
        double norm_v = 0.0;
        for (i = 0; i < m - k; i++) {
            norm_v += v[i] * v[i];
        }
        norm_v = sqrt(norm_v);
        if (norm_v != 0) {
            for (i = 0; i < m - k; i++) {
                v[i] /= norm_v;
            }
        }
        A[IDX(k, k, lda)] = sign * norm_x;
        for (i = k + 1; i < m; i++) {
            A[IDX(i, k, lda)] = v[i - k];
        }
        // Blocked update of trailing columns.
        nb = ((k + block_size) < n) ? block_size : (n - k);
        for (j = k + 1; j < k + nb; j++) {
            double dot = 0.0;
            for (i = k; i < m; i++) {
                double vi = (i == k) ? 1.0 : A[IDX(i, k, lda)];
                dot += vi * A[IDX(i, j, lda)];
            }
            for (i = k; i < m; i++) {
                double vi = (i == k) ? 1.0 : A[IDX(i, k, lda)];
                A[IDX(i, j, lda)] -= 2 * vi * dot;
            }
            // Reorthogonalization pass.
            dot = 0.0;
            for (i = k; i < m; i++) {
                double vi = (i == k) ? 1.0 : A[IDX(i, k, lda)];
                dot += vi * A[IDX(i, j, lda)];
            }
            for (i = k; i < m; i++) {
                double vi = (i == k) ? 1.0 : A[IDX(i, k, lda)];
                A[IDX(i, j, lda)] -= 2 * vi * dot;
            }
            double new_norm_sq = 0.0;
            for (i = k+1; i < m; i++) {
                new_norm_sq += A[IDX(i, j, lda)] * A[IDX(i, j, lda)];
            }
            double new_norm = sqrt(new_norm_sq);
            if (new_norm < 0.1 * norms[j]) {
                new_norm_sq = 0.0;
                for (i = k+1; i < m; i++) {
                    new_norm_sq += A[IDX(i, j, lda)] * A[IDX(i, j, lda)];
                }
                new_norm = sqrt(new_norm_sq);
            }
            norms_updated[j] = new_norm;
            norms[j] = new_norm;
        }
        for (j = k + nb; j < n; j++) {
            double dot = 0.0;
            for (i = k; i < m; i++) {
                double vi = (i == k) ? 1.0 : A[IDX(i, k, lda)];
                dot += vi * A[IDX(i, j, lda)];
            }
            for (i = k; i < m; i++) {
                double vi = (i == k) ? 1.0 : A[IDX(i, k, lda)];
                A[IDX(i, j, lda)] -= 2 * vi * dot;
            }
            dot = 0.0;
            for (i = k; i < m; i++) {
                double vi = (i == k) ? 1.0 : A[IDX(i, k, lda)];
                dot += vi * A[IDX(i, j, lda)];
            }
            for (i = k; i < m; i++) {
                double vi = (i == k) ? 1.0 : A[IDX(i, k, lda)];
                A[IDX(i, j, lda)] -= 2 * vi * dot;
            }
            double new_norm_sq = 0.0;
            for (i = k+1; i < m; i++) {
                new_norm_sq += A[IDX(i, j, lda)] * A[IDX(i, j, lda)];
            }
            double new_norm = sqrt(new_norm_sq);
            if (new_norm < 0.1 * norms[j]) {
                new_norm_sq = 0.0;
                for (i = k+1; i < m; i++) {
                    new_norm_sq += A[IDX(i, j, lda)] * A[IDX(i, j, lda)];
                }
                new_norm = sqrt(new_norm_sq);
            }
            norms_updated[j] = new_norm;
            norms[j] = new_norm;
        }
        free(v);
    }
    free(norms);
    free(norms_updated);
}

/**
 * @brief Applies the implicit Qᵀ (from a QR factorization) to a vector \c b in-place.
 *
 * @details
 * The matrix \c Q is stored implicitly in \c A (of size \c n×n). Each column \c k
 * contains the Householder vector from row \c k downwards (the first element
 * is implicitly \c 1.0). To apply \f$ Q^T \f$ to a vector, we iterate over each
 * Householder reflector in turn and perform the rank‐1 update.
 *
 * @param[in]  n   The dimension of the square matrix \c A.
 * @param[in]  A   Pointer to the \c n×n matrix (column‐major) that encodes the Householder vectors.
 * @param[in]  lda Leading dimension (must be \c >= n).
 * @param[in,out] b The input vector of length \c n. On output, \c b is overwritten by \f$Q^T b\f$.
 *
 * @warning This routine relies on the same storage convention as \c qr_decomp_colpivot_blocked,
 *          i.e., Householder vectors are stored in the lower portion of \c A from row \c k downwards
 *          in column \c k, with the first element of the reflector implicitly being \c 1.
 */
void applyQTranspose(int n, double *A, int lda, double *b) {
    int k, i;
    for (k = 0; k < n; k++) {
        double dot = 0.0;
        for (i = k; i < n; i++) {
            double vi = (i == k) ? 1.0 : A[IDX(i, k, lda)];
            dot += vi * b[i];
        }
        for (i = k; i < n; i++) {
            double vi = (i == k) ? 1.0 : A[IDX(i, k, lda)];
            b[i] -= 2 * vi * dot;
        }
    }
}

/**
 * @brief Solves an upper-triangular system \f$R x = b\f$ by back substitution.
 *
 * @param[in]  n     The size of the system (dimension).
 * @param[in]  A     Pointer to the \c n×n matrix (column‐major) whose upper triangle
 *                   represents \f$R\f$.
 * @param[in]  lda   Leading dimension of \c A (must be \c >= n).
 * @param[in]  b     The right‐hand side vector (length \c n).
 * @param[out] x     Output solution vector (length \c n).
 *
 * @pre \c R must be a nonsingular upper‐triangular matrix, or the solution
 *      will contain infinities/NaN if zero pivots are encountered.
 *
 * @note This is a standard back substitution procedure. We assume
 *       \f$R[i,i] \neq 0\f$ for all \c i.
 */
void backSubstitution(int n, double *A, int lda, double *b, double *x) {
    int i, j;
    for (i = n - 1; i >= 0; i--) {
        double sum = b[i];
        for (j = i + 1; j < n; j++) {
            sum -= A[IDX(i, j, lda)] * x[j];
        }
        x[i] = sum / A[IDX(i, i, lda)];
    }
}

/**
 * @brief Inverts a square matrix \c A (n×n) in column‐major order using
 *        the QR decomposition with column pivoting.
 *
 * @details
 * We first make a copy of \c A_in into a local array \c A. Then we call
 * \c qr_decomp_colpivot_blocked to factor it (and do column pivoting).
 * We have \f$A P = Q R\f$. To find \f$A^{-1}\f$, we solve \f$R \, y = Q^T e_i\f$
 * for each standard basis vector \f$ e_i \f$, and then apply the inverse pivot
 * to get each column of \f$A^{-1}\f$ in the correct position.
 *
 * The final result is stored in a newly allocated array of length \c n*n
 * in column‐major order.
 *
 * **Algorithm Sketch**:
 * 1. Copy \c A_in into local array \c A.
 * 2. Perform a rank‐revealing QR factorization (\c qr_decomp_colpivot_blocked), giving
 *    \f$ A P = Q R \f$. The pivot array \c jpvt keeps track of how columns were permuted.
 * 3. For each \c i in \c [0..n-1], solve \f$R \, y = Q^T e_i\f$ by:
 *    - Setting \c b = e_i (the i‐th unit vector).
 *    - Applying \c applyQTranspose to get \f$Q^T b\f$.
 *    - Using \c backSubstitution on \f$R y = Q^T e_i\f$ to find \f$y\f$.
 *    - Reorder \f$y\f$ according to \c inv_p (the inverse pivot) to get the correct column of \f$A^{-1}\f$.
 * 4. Store the result in \c A_inv, allocated by \c calloc.
 *
 * This approach combines the stability benefits of Householder reflectors,
 * column pivoting, and blocked updates into a single method for small
 * matrices (where \c block_size can be set to \c 1). Larger matrices might
 * benefit further from specialized BLAS/LAPACK routines, but this code
 * is illustrative of the general principle.
 *
 * @param[in]  A_in Pointer to the input \c n×n matrix in column‐major order.
 * @param[in]  n    The dimension of the matrix.
 * @return Pointer to a newly allocated array (length \c n*n) containing the inverse,
 *         also in column‐major order. The caller is responsible for \c free()ing it.
 *
 * @warning If \c A_in is singular or nearly singular, the results may be numerical garbage
 *          (or cause division by zero). The pivoting logic attempts to mitigate this by
 *          identifying small pivots and moving them to the right columns.
 *
 * @note For large \c n, using an optimized LAPACK routine (like \c dgeqp3 or \c dgeqrf)
 *       could be faster. However, this code demonstrates how to implement such an algorithm
 *       in a self‐contained manner.
 */
double *invertMatrixQR(const double *A_in, int n) {
    int lda = n;
    double *A = malloc(n * n * sizeof(double));
    if (!A) {
        fprintf(stderr, "Memory allocation error in invertMatrixQR.\n");
        exit(EXIT_FAILURE);
    }
    // Copy A_in into A.
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            A[IDX(i,j,lda)] = A_in[j * n + i];  // assume A_in is column-major
        }
    }
    int *jpvt = malloc(n * sizeof(int));
    if (!jpvt) {
        fprintf(stderr, "Memory allocation error in invertMatrixQR (jpvt).\n");
        exit(EXIT_FAILURE);
    }
    // For small matrices, block_size = 1.
    qr_decomp_colpivot_blocked(n, n, A, lda, jpvt, 1);

    // Build inverse permutation array.
    int *inv_p = malloc(n * sizeof(int));
    if (!inv_p) {
        fprintf(stderr, "Memory allocation error in invertMatrixQR (inv_p).\n");
        exit(EXIT_FAILURE);
    }
    for (int k = 0; k < n; k++) {
        inv_p[jpvt[k]] = k;
    }
    double *A_inv = calloc(n * n, sizeof(double));
    if (!A_inv) {
        fprintf(stderr, "Memory allocation error in invertMatrixQR (A_inv).\n");
        exit(EXIT_FAILURE);
    }
    double *b = malloc(n * sizeof(double));
    double *y = malloc(n * sizeof(double));
    double *x = malloc(n * sizeof(double));
    if (!b || !y || !x) {
        fprintf(stderr, "Memory allocation error in invertMatrixQR (temp vectors).\n");
        exit(EXIT_FAILURE);
    }
    // Solve R y = Qᵀ e_i for each unit vector e_i.
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            b[k] = (k == i) ? 1.0 : 0.0;
        }
        applyQTranspose(n, A, lda, b);
        backSubstitution(n, A, lda, b, y);
        for (int j = 0; j < n; j++) {
            x[j] = y[inv_p[j]];
        }
        for (int j = 0; j < n; j++) {
            A_inv[IDX(j, i, lda)] = x[j];
        }
    }
    free(A); free(jpvt); free(inv_p); free(b); free(y); free(x);
    return A_inv;
}

/*
 * Wrapper for inverting a 3×3 matrix.
 * Input matrix A3 must be in column–major order (length 9).
 * Returns a new dynamically allocated 3×3 inverse (column–major order).
 */
double *invert3x3matrixQR(const double A3[9]) {
    return invertMatrixQR(A3, 3);
}

/*
 * Wrapper for inverting a 4×4 matrix.
 * Input matrix A4 must be in column–major order (length 16).
 * Returns a new dynamically allocated 4×4 inverse (column–major order).
 */
double *invert4x4matrixQR(const double A4[16]) {
    return invertMatrixQR(A4, 4);
}


/*========================================================================
  Linear Regression Functions
========================================================================*/

/**
 * @brief Performs univariate linear regression.
 *
 * @param predictor The independent variable.
 * @param response The dependent variable.
 * @param length Number of observations.
 * @return Pointer to a dynamically allocated array [slope, intercept].
 *
 * This function centers the data, computes standard deviations and the Pearson
 * correlation, and then calculates:
 *   - slope = correlation * (std(response) / std(predictor))
 *   - intercept = mean(response) - slope * mean(predictor)
 */
double *performUnivariateLinearRegression(double predictor[], double response[], int length)
{
  double predictorDiff[length], responseDiff[length];
  double meanPredictor = calculateMean(predictor, length);
  double meanResponse = calculateMean(response, length);

  for (int i = 0; i < length; i++)
  {
    predictorDiff[i] = predictor[i] - meanPredictor;
    responseDiff[i] = response[i] - meanResponse;
  }

  double stdPredictor = calculateStandardDeviation(predictor, length);
  double stdResponse = calculateStandardDeviation(response, length);

  double *prodDiff = calculateElementwiseProduct(predictorDiff, responseDiff, length);
  double covariance = calculateArraySum(prodDiff, length) / (length - 1);
  free(prodDiff);

  double correlation = covariance / (stdPredictor * stdResponse);

  double slope = correlation * (stdResponse / stdPredictor);
  double intercept = meanResponse - slope * meanPredictor;

  double *estimates = malloc(sizeof(double) * 2);
  if (!estimates)
    exit(EXIT_FAILURE);
  estimates[0] = slope;
  estimates[1] = intercept;

  DEBUG_PRINT("Univariate Regression: meanX=%lf, meanY=%lf, corr=%lf, stdX=%lf, stdY=%lf\n",
              meanPredictor, meanResponse, correlation, stdPredictor, stdResponse);
  return estimates;
}

/**
 * @brief Generates predictions for a univariate regression model.
 *
 * @param predictor The independent variable.
 * @param predictions Output array for predicted values.
 * @param slope The slope coefficient.
 * @param intercept The intercept.
 * @param length Number of predictions to generate.
 */
void predictUnivariate(double predictor[], double predictions[], double slope, double intercept, int length)
{
  for (int i = 0; i < length; i++)
  {
    predictions[i] = predictor[i] * slope + intercept;
  }
}

/**
 * @brief Performs bivariate linear regression.
 *
 * @param predictor1 The first independent variable.
 * @param predictor2 The second independent variable.
 * @param response The dependent variable.
 * @param length Number of observations.
 * @return Pointer to a dynamically allocated array [beta1, beta2, intercept].
 */
double *performBivariateLinearRegression(double predictor1[], double predictor2[], double response[], int length)
{
  double pred1Diff[length], pred2Diff[length], respDiff[length];
  double meanPred1 = calculateMean(predictor1, length);
  double meanPred2 = calculateMean(predictor2, length);
  double meanResp = calculateMean(response, length);

  for (int i = 0; i < length; i++)
  {
    pred1Diff[i] = predictor1[i] - meanPred1;
    pred2Diff[i] = predictor2[i] - meanPred2;
    respDiff[i] = response[i] - meanResp;
  }

  double sumPred1 = calculateArraySum(predictor1, length);
  double sumPred2 = calculateArraySum(predictor2, length);
  double sumResp = calculateArraySum(response, length);

  double *prodPred1Resp = calculateElementwiseProduct(predictor1, response, length);
  double sumProdPred1Resp = calculateArraySum(prodPred1Resp, length);
  free(prodPred1Resp);

  double *prodPred2Resp = calculateElementwiseProduct(predictor2, response, length);
  double sumProdPred2Resp = calculateArraySum(prodPred2Resp, length);
  free(prodPred2Resp);

  double *prodPred1Pred2 = calculateElementwiseProduct(predictor1, predictor2, length);
  double sumProdPred1Pred2 = calculateArraySum(prodPred1Pred2, length);
  free(prodPred1Pred2);

  double variancePred1 = calculateArraySum(squareArray(pred1Diff, length), length);
  double variancePred2 = calculateArraySum(squareArray(pred2Diff, length), length);
  double covariancePred1Resp = sumProdPred1Resp - ((sumPred1 * sumResp) / length);
  double covariancePred2Resp = sumProdPred2Resp - ((sumPred2 * sumResp) / length);
  double covariancePred1Pred2 = sumProdPred1Pred2 - ((sumPred1 * sumPred2) / length);

  double denominator = (variancePred1 * variancePred2) - (covariancePred1Pred2 * covariancePred1Pred2);
  double beta1 = ((variancePred2 * covariancePred1Resp) - (covariancePred1Pred2 * covariancePred2Resp)) / denominator;
  double beta2 = ((variancePred1 * covariancePred2Resp) - (covariancePred1Pred2 * covariancePred1Resp)) / denominator;
  double intercept = meanResp - beta1 * meanPred1 - beta2 * meanPred2;

  double *estimates = malloc(sizeof(double) * 3);
  if (!estimates)
    exit(EXIT_FAILURE);
  estimates[0] = beta1;
  estimates[1] = beta2;
  estimates[2] = intercept;
  return estimates;
}

/**
 * @brief Generates predictions for a bivariate regression model.
 *
 * @param predictor1 The first independent variable.
 * @param predictor2 The second independent variable.
 * @param predictions Output array for predicted values.
 * @param beta1 Coefficient for predictor1.
 * @param beta2 Coefficient for predictor2.
 * @param intercept The intercept.
 * @param length Number of predictions.
 */
void predictBivariate(double predictor1[], double predictor2[], double predictions[], double beta1, double beta2, double intercept, int length)
{
  for (int i = 0; i < length; i++)
  {
    predictions[i] = predictor1[i] * beta1 + predictor2[i] * beta2 + intercept;
  }
}

/*========================================================================
  Updated Multivariate Linear Regression using QR Inversion
========================================================================*/

/**
 * @brief Performs multivariate linear regression.
 *
 * This version normalizes the design matrix and then computes the normal equations:
 * XᵀX * beta = XᵀY. It then inverts the (small) XᵀX matrix using our QR-based
 * inversion routines (for 3×3 or 4×4 systems) and computes beta.
 * The intercept is computed separately.
 *
 * @param numObservations Number of observations (rows).
 * @param numPredictors Number of predictors (columns).
 * @param X The design matrix.
 * @param Y The response vector (as a matrix with one column).
 * @return Pointer to a dynamically allocated array containing [beta coefficients..., intercept].
 */
double *performMultivariateLinearRegression(int numObservations, int numPredictors, 
                                              double X[][numPredictors], double Y[][1]) {
    double X_normalized[numObservations][numPredictors], Y_normalized[numObservations][1];
    double X_means[1][numPredictors], Y_mean[1][1];
    double Xt[numPredictors][numObservations], XtX[numPredictors][numPredictors];
    double XtX_inv[numPredictors][numPredictors];
    double XtX_inv_Xt[numPredictors][numObservations], beta[numPredictors][1];
    double *estimates = malloc(sizeof(double) * (numPredictors + 1));
    if (!estimates)
        exit(EXIT_FAILURE);

    // Normalize X and Y.
    normalize2DArray(numObservations, numPredictors, X, X_normalized);
    normalize2DArray(numObservations, 1, Y, Y_normalized);

    // Transpose X.
    transposeMatrix(numObservations, numPredictors, X_normalized, Xt);

    // Compute XtX = Xt * X.
    matrixMultiply(numPredictors, numObservations, numPredictors, Xt, X_normalized, XtX);

    int n = numPredictors;
    if (n == 1) {
        // For 1x1 matrix, the inverse is simply 1/(XtX[0][0]).
        XtX_inv[0][0] = 1.0 / XtX[0][0];
    } else if (n == 2) {
        // For 2x2 matrix, compute the inverse using the closed-form solution.
        double a = XtX[0][0];
        double b = XtX[0][1];
        double c = XtX[1][0];
        double d = XtX[1][1];
        double det = a * d - b * c;
        if (fabs(det) < 1e-12) {
            fprintf(stderr, "Error: 2x2 matrix is singular.\n");
            exit(EXIT_FAILURE);
        }
        XtX_inv[0][0] = d / det;
        XtX_inv[0][1] = -b / det;
        XtX_inv[1][0] = -c / det;
        XtX_inv[1][1] = a / det;
    } else if (n == 3) {
        double tempMatrix[n*n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                tempMatrix[IDX(i,j,n)] = XtX[i][j]; // convert to column-major
            }
        }
        double *invTemp = invert3x3matrixQR(tempMatrix);
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                XtX_inv[i][j] = invTemp[IDX(i,j,n)];
            }
        }
        free(invTemp);
    } else if (n == 4) {
        double tempMatrix[n*n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                tempMatrix[IDX(i,j,n)] = XtX[i][j]; // convert to column-major
            }
        }
        double *invTemp = invert4x4matrixQR(tempMatrix);
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                XtX_inv[i][j] = invTemp[IDX(i,j,n)];
            }
        }
        free(invTemp);
    } else if (n == 5) {
        // For 5x5 matrix, use our general QR inversion routine.
        double tempMatrix[n*n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                tempMatrix[IDX(i,j,n)] = XtX[i][j]; // convert to column-major
            }
        }
        double *invTemp = invertMatrixQR(tempMatrix, n);
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                XtX_inv[i][j] = invTemp[IDX(i,j,n)];
            }
        }
        free(invTemp);
    } else {
        fprintf(stderr, "Error: QR inversion for n=%d not implemented.\n", n);
        exit(EXIT_FAILURE);
    }


    // Compute XtX_inv_Xt = XtX_inv * Xt.
    matrixMultiply(numPredictors, numPredictors, numObservations, XtX_inv, Xt, XtX_inv_Xt);

    // Compute beta = XtX_inv_Xt * Y_normalized.
    matrixMultiply(numPredictors, numObservations, 1, XtX_inv_Xt, Y_normalized, beta);
    for (int i = 0; i < numPredictors; i++) {
        estimates[i] = beta[i][0];
    }

    // Compute intercept from the original means.
    for (int j = 0; j < numPredictors; j++) {
        double col[numObservations];
        for (int i = 0; i < numObservations; i++) {
            col[i] = X[i][j];
        }
        X_means[0][j] = calculateMean(col, numObservations);
    }
    {
        double yCol[numObservations];
        for (int i = 0; i < numObservations; i++) {
            yCol[i] = Y[i][0];
        }
        Y_mean[0][0] = calculateMean(yCol, numObservations);
    }
    double intercept = Y_mean[0][0];
    for (int i = 0; i < numPredictors; i++) {
        intercept -= estimates[i] * X_means[0][i];
    }
    estimates[numPredictors] = intercept;
    return estimates;
}



/**
 * @brief Computes the Pearson correlation coefficient between two arrays.
 *
 * @param array1 The first array.
 * @param array2 The second array.
 * @param length The number of elements.
 * @return The correlation coefficient.
 */
double computeCorrelation(double array1[], double array2[], int length)
{
  double diff1[length], diff2[length];
  double mean1 = calculateMean(array1, length);
  double mean2 = calculateMean(array2, length);
  for (int i = 0; i < length; i++)
  {
    diff1[i] = array1[i] - mean1;
    diff2[i] = array2[i] - mean2;
  }
  double std1 = calculateStandardDeviation(array1, length);
  double std2 = calculateStandardDeviation(array2, length);
  double *prodDiff = calculateElementwiseProduct(diff1, diff2, length);
  double correlation = calculateArraySum(prodDiff, length) / ((length - 1) * std1 * std2);
  free(prodDiff);
  return correlation;
}

/*========================================================================
  Stationarity Testing and Drift Adjustment
========================================================================*/

/**
 * @brief Differences a time series to achieve stationarity.
 *
 * Given an input series and a differencing order d, this function returns a
 * new dynamically allocated array containing the d-th order differenced series.
 * The length of the returned series is (original length - d).
 *
 * @param series The input time series array.
 * @param length The number of observations in the input series.
 * @param d The order of differencing to apply.
 * @return Pointer to the differenced series. Caller is responsible for freeing the memory.
 *
 * @note Differencing is a core step in ARIMA models (the "I" component). This function
 *       does not automatically determine the optimal d; it assumes d is provided (e.g., via
 *       iterative testing with an ADF test).
 */
double *differenceSeries(const double series[], int length, int d)
{
  if (d < 0 || d >= length)
  {
    fprintf(stderr, "Error: Invalid differencing order d=%d for series of length %d.\n", d, length);
    exit(EXIT_FAILURE);
  }

  // Allocate a working copy to hold the current version of the series.
  int currentLength = length;
  double *current = malloc(sizeof(double) * currentLength);
  if (!current)
  {
    fprintf(stderr, "Memory allocation error in differenceSeries.\n");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < currentLength; i++)
  {
    current[i] = series[i];
  }

  // Apply differencing d times.
  for (int diff = 0; diff < d; diff++)
  {
    int newLength = currentLength - 1;
    double *temp = malloc(sizeof(double) * newLength);
    if (!temp)
    {
      fprintf(stderr, "Memory allocation error in differenceSeries (temp).\n");
      exit(EXIT_FAILURE);
    }
    for (int i = 0; i < newLength; i++)
    {
      temp[i] = current[i + 1] - current[i];
    }
    free(current);
    current = temp;
    currentLength = newLength;
  }

  return current; // Length is (length - d)
}

/**
 * @brief Integrates a differenced series to recover the original scale.
 *
 * Given the forecasted values on the differenced scale and the last observed value
 * (or a vector of drift/recovery values), this function reconstructs the forecast
 * on the original scale via cumulative summation.
 *
 * @param diffForecast The forecast on the differenced scale.
 * @param recoveryValue The last observed value of the original series (or the drift vector).
 * @param forecastLength Number of forecast steps.
 * @return A new dynamically allocated array containing the integrated (recovered) forecast.
 *
 * @note For one order of differencing, the integration is essentially:
 *       forecast_original[0] = recoveryValue + diffForecast[0]
 *       forecast_original[i] = forecast_original[i-1] + diffForecast[i]
 */
double *integrateSeries(const double diffForecast[], double recoveryValue, int forecastLength)
{
  double *integrated = malloc(sizeof(double) * forecastLength);
  if (!integrated)
  {
    fprintf(stderr, "Memory allocation error in integrateSeries.\n");
    exit(EXIT_FAILURE);
  }
  integrated[0] = recoveryValue + diffForecast[0];
  for (int i = 1; i < forecastLength; i++)
  {
    integrated[i] = integrated[i - 1] + diffForecast[i];
  }
  return integrated;
}

/* --- New constants for the extended ADF test --- */
#define MODEL_CONSTANT_ONLY 0
#define MODEL_CONSTANT_TREND 1

// Critical values for ADF test (approximate) for the intercept-only model:
static const double ADF_CRIT_CONSTANT[3] = {-3.43, -2.86, -2.57}; // 1%, 5%, 10% levels
// Critical values for ADF test with constant and trend:
static const double ADF_CRIT_TREND[3] = {-4.15, -3.41, -3.12}; // 1%, 5%, 10% levels

/* --- Helper function: Approximate p-value from ADF test statistic --- */
double adfPValue(double tstat, int modelType)
{
  // For simplicity, we do piecewise linear interpolation.
  // If tstat is below the 1% critical value, return 0.005; if above 10%, return 0.15.
  const double *cv = (modelType == MODEL_CONSTANT_TREND) ? ADF_CRIT_TREND : ADF_CRIT_CONSTANT;
  const double pvals[3] = {0.01, 0.05, 0.10};
  if (tstat <= cv[0])
    return 0.005;
  if (tstat >= cv[2])
    return 0.15;
  if (tstat <= cv[1])
  {
    double frac = (tstat - cv[0]) / (cv[1] - cv[0]);
    return pvals[0] + frac * (pvals[1] - pvals[0]);
  }
  else
  { // between cv[1] and cv[2]
    double frac = (tstat - cv[1]) / (cv[2] - cv[1]);
    return pvals[1] + frac * (pvals[2] - pvals[1]);
  }
}

/* --- Helper functions for the extended ADF test --- */

// Compute autocorrelation for a lag (simple implementation)
double autocorrelation(const double series[], int n, int lag)
{
  double mean = 0.0;
  for (int i = 0; i < n; i++)
    mean += series[i];
  mean /= n;
  double num = 0.0, den = 0.0;
  for (int i = 0; i < n - lag; i++)
  {
    num += (series[i] - mean) * (series[i + lag] - mean);
  }
  for (int i = 0; i < n; i++)
  {
    double diff = series[i] - mean;
    den += diff * diff;
  }
  return num / den;
}

/* --- Extended ADF test function --- */
/**
 * @brief Extended Augmented Dickey–Fuller test.
 *
 * This function performs an augmented Dickey–Fuller regression with a flexible lag order
 * chosen by AIC minimization and with an option to include a trend term.
 *
 * The regression model is:
 *    Δy_t = α + β * y_{t-1} + [δ * t] + Σ_{i=1}^{p} γ_i Δy_{t-i} + ε_t
 *
 * where p is chosen (0 ≤ p ≤ maxLag) by selecting the model with the smallest AIC.
 *
 * The test statistic is taken as the estimated coefficient on y_{t-1} (β). An approximate
 * p-value is computed using pre-stored critical values.
 *
 * @param series The input time series (length observations).
 * @param length The length of the series.
 * @param maxLag Maximum lag to consider for the augmentation.
 * @param modelType Set to MODEL_CONSTANT_ONLY (0) for constant only, or MODEL_CONSTANT_TREND (1) for constant and trend.
 * @param tStat Output pointer for the estimated test statistic (β).
 * @param pValue Output pointer for the approximate p-value.
 * @return 1 if the null (unit root present) is rejected (i.e. series is stationary), 0 otherwise.
 */
int ADFTestExtendedAutoLag(double series[], int length, int modelType, double *tStat, double *pValue)
{
  // Common rule-of-thumb for ADF test:
  int pMax = (int)floor(pow(length - 1, ADF_LAG_EXPONENT));
  if (pMax < 0)
    pMax = 0; // safety check

  int bestP = 0;
  double bestAIC = 1e30;
  double bestBeta = 0.0;

  for (int p = 0; p <= pMax; p++)
  {
    int nEff = length - p - 1;
    int k = 2 + p; // constant and lagged level plus p lagged differences
    if (modelType == MODEL_CONSTANT_TREND)
    {
      k += 1; // add trend term if requested
    }
    if (nEff < k + 1)
    {
      // If you don't have enough data points to fit these many parameters, break/continue.
      // or you can skip this p if nEff < k+1.
      // We'll just break here for simplicity:
      break;
    }

    double X[nEff][k];
    double Y[nEff];

    // Build the ADF regression design for t = p+1 ... length-1
    for (int i = 0; i < nEff; i++)
    {
      int t = i + p + 1;
      Y[i] = series[t] - series[t - 1]; // Δy_t = y_t - y_{t-1}
      int col = 0;
      X[i][col++] = 1.0;           // constant
      X[i][col++] = series[t - 1]; // lagged level (y_{t-1})
      if (modelType == MODEL_CONSTANT_TREND)
      {
        X[i][col++] = (double)t; // optional trend: e.g. t or (t - p - 1)
      }
      // p lagged differences
      for (int j = 1; j <= p; j++)
      {
        X[i][col++] = series[t - j] - series[t - j - 1];
      }
    }

    // OLS
    double *betaEst = performMultivariateLinearRegression(nEff, k, X, (double(*)[1])Y);

    // Compute RSS
    double RSS = 0.0;
    for (int i = 0; i < nEff; i++)
    {
      double pred = 0.0;
      for (int j = 0; j < k; j++)
      {
        pred += betaEst[j] * X[i][j];
      }
      double err = Y[i] - pred;
      RSS += err * err;
    }

    // AIC = nEff * ln(RSS/nEff) + 2*k
    double AIC = nEff * log(RSS / nEff) + 2.0 * k;

    if (AIC < bestAIC)
    {
      bestAIC = AIC;
      bestP = p;
      bestBeta = betaEst[1]; // The coefficient on y_{t-1}
    }
    free(betaEst);
  }

  // Now we have bestBeta from the best (lowest AIC) model
  *tStat = bestBeta;
  *pValue = adfPValue(bestBeta, modelType);

  // Stationary if pValue < 0.05, for example
  return (*pValue < 0.05) ? 1 : 0;
}

/* --- Chi-square CDF implementation for the Ljung-Box test --- */
/**
 * @brief Computes the regularized lower incomplete gamma function P(a, x).
 *
 * This implementation uses a series expansion for x < a+1 and a continued fraction for x >= a+1.
 *
 * @param a Parameter a.
 * @param x Parameter x.
 * @return The regularized lower incomplete gamma function P(a, x).
 */
double gammaP(double a, double x)
{
  const int ITMAX = 100;
  const double EPS = 3.0e-7;
  double gln = lgamma(a);
  if (x < a + 1.0)
  {
    // Series expansion
    double sum = 1.0 / a;
    double del = sum;
    for (int n = 1; n <= ITMAX; n++)
    {
      a += 1.0;
      del *= x / a;
      sum += del;
      if (fabs(del) < fabs(sum) * EPS)
        break;
    }
    return sum * exp(-x + (a - 1) * log(x) - gln);
  }
  else
  {
    // Continued fraction
    double b = x + 1.0 - a;
    double c = 1.0 / 1.0e-30;
    double d = 1.0 / b;
    double h = d;
    for (int n = 1; n <= ITMAX; n++)
    {
      double an = -n * (n - a);
      b += 2.0;
      d = an * d + b;
      if (fabs(d) < 1.0e-30)
        d = 1.0e-30;
      c = b + an / c;
      if (fabs(c) < 1.0e-30)
        c = 1.0e-30;
      d = 1.0 / d;
      double delta = d * c;
      h *= delta;
      if (fabs(delta - 1.0) < EPS)
        break;
    }
    return 1.0 - h * exp(-x + (a - 1) * log(x) - gln);
  }
}

/**
 * @brief Computes the chi-square CDF for x with k degrees of freedom.
 *
 * @param x The chi-square statistic.
 * @param k Degrees of freedom.
 * @return The CDF value.
 */
double chiSquareCDF(double x, int k)
{
  return gammaP(k / 2.0, x / 2.0);
}

/* --- Ljung–Box test functions --- */
/**
 * @brief Computes the Ljung–Box Q-statistic for a series of residuals.
 *
 * @param residuals The residuals from the model.
 * @param n The number of residuals.
 * @param h The number of lags to include in the test.
 * @return The Ljung–Box Q-statistic.
 */
double ljungBoxQ(const double residuals[], int n, int h)
{
  double Q = 0.0;
  for (int k = 1; k <= h; k++)
  {
    double r = autocorrelation(residuals, n, k);
    Q += r * r / (n - k);
  }
  Q *= n * (n + 2);
  return Q;
}

/**
 * @brief Computes the Ljung–Box test p-value.
 *
 * @param residuals The residuals from the model.
 * @param n The number of residuals.
 * @param h The number of lags used in the test.
 * @param m The number of parameters estimated in the model (used to adjust degrees of freedom).
 * @return The approximate p-value.
 */
double ljungBoxPValue(const double residuals[], int n, int h, int m)
{
  double Q = ljungBoxQ(residuals, n, h);
  int df = h - m; // degrees of freedom
  double p = 1.0 - chiSquareCDF(Q, df);
  return p;
}

/**
 * @brief Adjusts a differenced series to recover the original level (drift correction).
 *
 * @param originalSeries The original series.
 * @param adjustedSeries The series to adjust.
 * @param length The number of observations.
 * @param diffOrder The order of differencing previously applied.
 */
void adjustDrift(double originalSeries[], double adjustedSeries[], int length, int diffOrder)
{
  for (int i = 0; i < diffOrder; i++)
  {
    for (int j = 0; j < (length - diffOrder); j++)
    {
      adjustedSeries[j] = adjustedSeries[j + diffOrder] - adjustedSeries[j];
    }
  }
}

/**
 * @brief Helper function to check residual diagnostics via the Ljung–Box test.
 *
 * If the p-value is below 0.05, a warning is printed.
 *
 * @param residuals The residuals.
 * @param n Number of residuals.
 * @param h Number of lags for the test.
 * @param m Number of estimated parameters.
 */
void checkResidualDiagnostics(const double residuals[], int n, int h, int m)
{
  double pVal = ljungBoxPValue(residuals, n, h, m);
  if (pVal < 0.05)
  {
    printf("Warning: Residuals exhibit significant autocorrelation (Ljung–Box p = %.4lf).\n", pVal);
  }
  else
  {
    printf("Residuals pass the Ljung–Box test (p = %.4lf).\n", pVal);
  }
}

/*========================================================================
  Diagnostic Matrix: Extended Autocorrelation Function (EAF)
========================================================================*/

/**
 * @brief Computes the extended autocorrelation (EAF) matrix.
 *
 * @param series The input time series.
 * @param eafMatrix Output 3x3 matrix holding the extended autocorrelations.
 * @param length The number of observations.
 *
 * The function computes:
 *  - The correlations between the series and its various leads/lags.
 *  - Fits an AR(1) model and computes the autocorrelations of the errors.
 *  - Repeats the process for an AR(2) model.
 */
void computeEAFMatrix(double series[], double eafMatrix[][3], int length)
{
  // Temporary arrays for shifted versions.
  double seriesLead0[length - 3], seriesLead1[length - 3], seriesLead2[length - 3], seriesLead3[length - 3];
  double forecastSeries[length - 3], errorSeries[length - 3];
  double ar1Slope, ar1Intercept, ar2Slope1, ar2Slope2, ar2Intercept;
  double errorLead[length - 6], errorLead1[length - 6], errorLead2[length - 6], errorLead3[length - 6];
  double *ar1Estimates, *ar2Estimates;

  // Generate lead/lag series.
  calculateLeadWithLag(series, seriesLead0, length, 0, 3);
  calculateLeadWithLag(series, seriesLead1, length, 1, 2);
  calculateLeadWithLag(series, seriesLead2, length, 2, 1);
  calculateLeadWithLag(series, seriesLead3, length, 3, 0);
  int newLength = length - 3;

  // First row: correlations between seriesLead0 and shifted versions.
  eafMatrix[0][0] = computeCorrelation(seriesLead0, seriesLead1, newLength);
  eafMatrix[0][1] = computeCorrelation(seriesLead0, seriesLead2, newLength);
  eafMatrix[0][2] = computeCorrelation(seriesLead0, seriesLead3, newLength);

  // Fit AR(1) model on seriesLead1 vs seriesLead0.
  ar1Estimates = performUnivariateLinearRegression(seriesLead1, seriesLead0, newLength);
  ar1Slope = ar1Estimates[0];
  ar1Intercept = ar1Estimates[1];
  predictUnivariate(seriesLead1, forecastSeries, ar1Slope, ar1Intercept, newLength);
  calculateArrayDifference(seriesLead0, forecastSeries, errorSeries, newLength);

  // Compute error correlations.
  calculateLeadWithLag(errorSeries, errorLead, newLength, 0, 3);
  calculateLeadWithLag(errorSeries, errorLead1, newLength, 1, 2);
  calculateLeadWithLag(errorSeries, errorLead2, newLength, 2, 1);
  calculateLeadWithLag(errorSeries, errorLead3, newLength, 3, 0);
  eafMatrix[1][0] = computeCorrelation(errorLead, errorLead1, newLength - 3);
  eafMatrix[1][1] = computeCorrelation(errorLead, errorLead2, newLength - 3);
  eafMatrix[1][2] = computeCorrelation(errorLead, errorLead3, newLength - 3);

  // Fit AR(2) model.
  ar2Estimates = performBivariateLinearRegression(seriesLead1, seriesLead2, seriesLead0, newLength);
  ar2Slope1 = ar2Estimates[0];
  ar2Slope2 = ar2Estimates[1];
  ar2Intercept = ar2Estimates[2];
  predictBivariate(seriesLead1, seriesLead2, forecastSeries, ar2Slope1, ar2Slope2, ar2Intercept, newLength);
  calculateArrayDifference(seriesLead0, forecastSeries, errorSeries, newLength);
  calculateLeadWithLag(errorSeries, errorLead, newLength, 0, 3);
  calculateLeadWithLag(errorSeries, errorLead1, newLength, 1, 2);
  calculateLeadWithLag(errorSeries, errorLead2, newLength, 2, 1);
  calculateLeadWithLag(errorSeries, errorLead3, newLength, 3, 0);
  eafMatrix[2][0] = computeCorrelation(errorLead, errorLead1, newLength - 3);
  eafMatrix[2][1] = computeCorrelation(errorLead, errorLead2, newLength - 3);
  eafMatrix[2][2] = computeCorrelation(errorLead, errorLead3, newLength - 3);

  free(ar1Estimates);
  free(ar2Estimates);
  DEBUG_PRINT("EAF Matrix computed.\n");
}

/*========================================================================
  Error Metrics & Moving Average
========================================================================*/

/**
 * @brief Calculates the Mean Absolute Percentage Error (MAPE).
 *
 * @param actual The array of actual values.
 * @param predicted The array of predicted values.
 * @param length The number of elements.
 * @return MAPE as a percentage.
 */
double calculateMAPE(double actual[], double predicted[], int length)
{
  double errorPercentage[length];
  for (int i = 0; i < length; i++)
  {
    errorPercentage[i] = fabs((actual[i] - predicted[i]) / actual[i]);
  }
  return (calculateArraySum(errorPercentage, length) / length) * 100;
}

/**
 * @brief Calculates the Mean Absolute Error (MAE).
 *
 * @param actual The array of actual values.
 * @param predicted The array of predicted values.
 * @param length The number of elements.
 * @return MAE as a percentage.
 */
double calculateMAE(double actual[], double predicted[], int length)
{
  double absoluteError[length];
  for (int i = 0; i < length; i++)
  {
    absoluteError[i] = fabs(actual[i] - predicted[i]);
  }
  return (calculateArraySum(absoluteError, length) / length) * 100;
}

/*========================================================================
  Forecasting Model: AR(1)
========================================================================*/

/**
 * @brief Forecasts future values using an AR(1) model.
 *
 * @param series The input time series.
 * @param length The length of the series.
 * @return A dynamically allocated array of forecasted values; element FORECAST_HORIZON contains the MAPE.
 *
 * The function estimates an AR(1) model using the last (length-1) observations,
 * computes the in-sample prediction error (MAPE), and then recursively forecasts
 * FORECAST_HORIZON amount of future values.
 */
double *forecastAR1(double series[], int length)
{
  int newLength = length - 1;
  double *regressionEstimates = performUnivariateLinearRegression(series, series + 1, newLength);
  double phi = regressionEstimates[0];
  double intercept = regressionEstimates[1];
  // Compute residuals and estimate sigma^2
  double predictions[newLength];
  predictUnivariate(series, predictions, phi, intercept, newLength);
  double residuals[newLength];
  calculateArrayDifference(series + 1, predictions, residuals, newLength);
  double rss = calculateArraySum(squareArray(residuals, newLength), newLength);
  double sigma2 = rss / (newLength - 2);

  double *forecast = malloc(sizeof(double) * FORECAST_ARRAY_SIZE);
  if (!forecast)
    exit(EXIT_FAILURE);

  double lastValue = series[newLength - 1];
  forecast[0] = lastValue * phi + intercept;

  // Compute cumulative forecast error variance using the formula:
  // Var(h) = sigma2 * sum_{i=0}^{h-1} (phi^(2*i))
  double cumulativeVariance = sigma2; // for horizon h=1
  for (int i = 1; i < FORECAST_HORIZON; i++)
  {
    forecast[i] = forecast[i - 1] * phi + intercept;
    cumulativeVariance += sigma2 * pow(phi, 2 * i);
  }
  // Store the cumulative variance (as a proxy for uncertainty) in forecast[FORECAST_HORIZON]
  forecast[FORECAST_HORIZON] = cumulativeVariance;
  free(regressionEstimates);
  return forecast;
}

/**
 * @brief Recovers forecasted values to the original scale by adding back drift.
 *
 * @param forecastDiff The forecasted values on the differenced scale.
 * @param recoveryValues The recovery (drift) values saved during differencing.
 * @param finalForecast Output array where recovered forecasts are stored.
 * @param numForecasts Number of forecasted values.
 * @param diffOrder The order of differencing that was applied.
 *
 * This function applies cumulative summation to convert forecasts on
 * the differenced scale back to the original scale.
 */
void recoverForecast(double forecastDiff[], double recoveryValues[], double finalForecast[], int numForecasts, int diffOrder)
{
  copyArray(forecastDiff, finalForecast, numForecasts);
  for (int i = 0; i < diffOrder; i++)
  {
    double drift = recoveryValues[diffOrder - i - 1];
    for (int j = 0; j < numForecasts; j++)
    {
      finalForecast[j] += drift;
      drift = finalForecast[j]; // propagate drift cumulatively
    }
  }
}

/**
 * @brief Prepares a series for ARMA modeling by testing for stationarity using the extended ADF test.
 *
 * This function iteratively differences the input series until the series is deemed stationary
 * (according to the extended ADF test with automatic lag selection) or until a maximum differencing order is reached.
 *
 * @param series The original time series.
 * @param length The length of the original series.
 * @param maxDiffOrder Maximum differencing order to try.
 * @param modelType Set to MODEL_CONSTANT_ONLY or MODEL_CONSTANT_TREND for the ADF test.
 * @param outDiffOrder (Output) The order of differencing that was applied.
 * @param outNewLength (Output) The length of the differenced series.
 * @return Pointer to the differenced series (which should be stationary). Caller is responsible for freeing the memory.
 */
double *prepareSeriesForARMA(const double series[], int length, int maxDiffOrder, int modelType, int *outDiffOrder, int *outNewLength)
{
  int d = 0;
  int currentLength = length;
  double *currentSeries = malloc(sizeof(double) * currentLength);
  if (!currentSeries)
  {
    fprintf(stderr, "Memory allocation error in prepareSeriesForARMA.\n");
    exit(EXIT_FAILURE);
  }
  memcpy(currentSeries, series, sizeof(double) * currentLength);

  double tStat, pValue;
  // Use the auto-lag version of the ADF test.
  int isStationary = ADFTestExtendedAutoLag(currentSeries, currentLength, modelType, &tStat, &pValue);

  // While the series is not stationary and we haven't reached the maximum differencing order,
  // difference the series one time and retest.
  while (!isStationary && d < maxDiffOrder)
  {
    d++;
    double *temp = differenceSeries(currentSeries, currentLength, 1);
    free(currentSeries);
    currentSeries = temp;
    currentLength--; // each differencing reduces length by 1
    isStationary = ADFTestExtendedAutoLag(currentSeries, currentLength, modelType, &tStat, &pValue);
  }

  *outDiffOrder = d;
  *outNewLength = currentLength;
  return currentSeries;
}


/**
 * @brief Computes the autocorrelation function (ACF) up to a specified maximum lag.
 *
 * @param series The input time series.
 * @param length The number of observations in the series.
 * @param maxLag The maximum lag for which to compute the ACF.
 * @param acf Output array of length (maxLag + 1) where acf[lag] is the autocorrelation at that lag.
 */
void computeACF(const double series[], int length, int maxLag, double acf[]) {
    double mean = calculateMean(series, length);
    double denom = 0.0;
    for (int i = 0; i < length; i++) {
        double diff = series[i] - mean;
        denom += diff * diff;
    }
    for (int lag = 0; lag <= maxLag; lag++) {
        double num = 0.0;
        for (int i = 0; i < length - lag; i++) {
            num += (series[i] - mean) * (series[i + lag] - mean);
        }
        acf[lag] = num / denom;
    }
}

/**
 * @brief Computes the partial autocorrelation function (PACF) using the Levinson–Durbin recursion.
 *
 * @param series The input time series.
 * @param length The number of observations.
 * @param maxLag The maximum lag for which to compute the PACF.
 * @param pacf Output array of length (maxLag + 1) where pacf[lag] is the PACF at that lag.
 *
 * @note The PACF at lag 0 is conventionally set to 1.
 */
void computePACF(const double series[], int length, int maxLag, double pacf[]) {
    double *acf = malloc(sizeof(double) * (maxLag + 1));
    if (!acf) {
        fprintf(stderr, "Memory allocation error in computePACF.\n");
        exit(EXIT_FAILURE);
    }
    computeACF(series, length, maxLag, acf);

    // Allocate arrays for the recursion (indices 1..maxLag are used)
    double *phi = malloc(sizeof(double) * (maxLag + 1));
    double *prev_phi = malloc(sizeof(double) * (maxLag + 1));
    if (!phi || !prev_phi) {
        fprintf(stderr, "Memory allocation error in computePACF.\n");
        exit(EXIT_FAILURE);
    }

    pacf[0] = 1.0;  // By definition
    // For m = 1:
    phi[1] = acf[1] / acf[0];
    pacf[1] = phi[1];
    double E = acf[0] * (1 - phi[1] * phi[1]);
    prev_phi[1] = phi[1];

    // For m = 2, 3, ..., maxLag:
    for (int m = 2; m <= maxLag; m++) {
        double sum = 0.0;
        for (int k = 1; k < m; k++) {
            sum += prev_phi[k] * acf[m - k];
        }
        phi[m] = (acf[m] - sum) / E;
        pacf[m] = phi[m];
        // Update the intermediate AR coefficients for lags 1,..., m-1:
        for (int k = 1; k < m; k++) {
            phi[k] = prev_phi[k] - phi[m] * prev_phi[m - k];
        }
        E = E * (1 - phi[m] * phi[m]);
        for (int k = 1; k <= m; k++) {
            prev_phi[k] = phi[k];
        }
    }
    free(acf);
    free(phi);
    free(prev_phi);
}

/**
 * @brief Automatically selects AR (p) and MA (q) orders based on the sample PACF and ACF.
 *
 * This function computes the sample ACF and PACF up to a specified maximum lag and
 * then uses a simple significance threshold (±1.96/√n) to decide which lags are significant.
 * The AR order is set to the highest lag where the PACF is significant and the MA order is
 * set to the highest lag where the ACF is significant.
 *
 * @param series The input time series.
 * @param length The number of observations.
 * @param maxLag The maximum lag to consider (e.g., 20).
 * @param selectedAR Output pointer for the selected AR order.
 * @param selectedMA Output pointer for the selected MA order.
 */
void selectARMAOrders(const double series[], int length, int maxLag, int *selectedAR, int *selectedMA) {
    double *acf = malloc(sizeof(double) * (maxLag + 1));
    double *pacf = malloc(sizeof(double) * (maxLag + 1));
    if (!acf || !pacf) {
        fprintf(stderr, "Memory allocation error in selectARMAOrders.\n");
        exit(EXIT_FAILURE);
    }

    computeACF(series, length, maxLag, acf);
    computePACF(series, length, maxLag, pacf);

    double threshold = 1.96 / sqrt((double)length);
    int p = 0, q = 0;
    // Check lags 1 to maxLag (lag 0 is trivial)
    for (int lag = 1; lag <= maxLag; lag++) {
        if (fabs(pacf[lag]) > threshold) {
            p = lag;  // Update AR order to the highest significant lag in PACF
        }
        if (fabs(acf[lag]) > threshold) {
            q = lag;  // Update MA order to the highest significant lag in ACF
        }
    }
    *selectedAR = p;
    *selectedMA = q;
    free(acf);
    free(pacf);
}

/**
 * @brief Uses ACF/PACF to get initial ARMA orders and then validates them using the EAFMatrix.
 *
 * This function first calls selectARMAOrders to get initial candidate orders.
 * Then, it fits a simple AR model (assuming d = 0 for simplicity) to compute in‐sample residuals.
 * It computes the EAFMatrix on the residuals and examines the off‐diagonal elements.
 * If these diagnostics suggest that there is remaining structure (i.e. average absolute off-diagonals
 * exceed a threshold), the AR order is increased (you could similarly adjust the MA order).
 *
 * @param series     The input time series.
 * @param length     The number of observations.
 * @param maxLag     The maximum lag to consider for ACF/PACF.
 * @param finalAR    Output pointer for the final selected AR order.
 * @param finalMA    Output pointer for the final selected MA order.
 */
void selectOrdersWithFeedback(const double series[], int length, int maxLag, int *finalAR, int *finalMA) {
    // Step 1: Get initial candidate orders using ACF/PACF.
    int candidateAR = 0, candidateMA = 0;
    selectARMAOrders(series, length, maxLag, &candidateAR, &candidateMA);

    // We'll use a simple feedback loop to adjust the AR order if needed.
    // (For simplicity, we assume d = 0 in this demonstration.)
    int iterations = 0;
    int maxIterations = 5;
    // Set a threshold for the average absolute off-diagonal element in the EAFMatrix.
    // This threshold can be adjusted (e.g., 0.1, 0.15, etc.)
    double eafThreshold = 0.1;
    int adjust = 1;
    
    while (adjust && iterations < maxIterations) {
        // Step 2: Fit a candidate AR model with candidateAR order and compute residuals.
        // For t = candidateAR ... length-1, the regression is:
        //   series[t] = intercept + sum_{j=1}^{candidateAR} phi_j * series[t-j] + error[t]
        int m = length - candidateAR;
        double *fitted = malloc(sizeof(double) * m);
        if (!fitted) {
            fprintf(stderr, "Memory allocation error in selectOrdersWithFeedback (fitted).\n");
            exit(EXIT_FAILURE);
        }
        if (candidateAR > 0 && m > 0) {
            // Build design matrix X (m x candidateAR) and response Y (m x 1).
            double X[m][candidateAR];
            double Y[m][1];
            for (int t = candidateAR; t < length; t++) {
                Y[t - candidateAR][0] = series[t];
                for (int j = 0; j < candidateAR; j++) {
                    X[t - candidateAR][j] = series[t - j - 1];
                }
            }
            // Estimate coefficients (returns candidateAR coefficients followed by intercept).
            double *estimates = performMultivariateLinearRegression(m, candidateAR, X, Y);
            // Compute fitted values.
            for (int t = candidateAR; t < length; t++) {
                double pred = estimates[candidateAR]; // intercept
                for (int j = 0; j < candidateAR; j++) {
                    pred += estimates[j] * series[t - j - 1];
                }
                fitted[t - candidateAR] = pred;
            }
            free(estimates);
        } else {
            // If candidateAR is 0, then the fitted value is just the series itself.
            for (int i = 0; i < m; i++) {
                fitted[i] = series[i];
            }
        }
        
        // Step 3: Compute residuals.
        double *residuals = malloc(sizeof(double) * m);
        if (!residuals) {
            fprintf(stderr, "Memory allocation error in selectOrdersWithFeedback (residuals).\n");
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < m; i++) {
            residuals[i] = series[candidateAR + i] - fitted[i];
        }
        
        // Step 4: Compute the EAFMatrix on the residuals.
        // Note: EAFMatrix requires at least 6 observations (since it uses length-3 internally).
        double eaf[3][3] = {0};
        if (m >= 6) {
            computeEAFMatrix(residuals, eaf, m);
        }
        
        // Evaluate the off-diagonal elements from row 1 and row 2.
        double sum = 0.0;
        int count = 0;
        for (int i = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (j != i) {
                    sum += fabs(eaf[i][j]);
                    count++;
                }
            }
        }
        double avgOffDiag = (count > 0) ? (sum / count) : 0.0;
        
        // Debug print (if desired):
        // printf("Iteration %d: candidateAR=%d, avgOffDiag=%.4lf\n", iterations, candidateAR, avgOffDiag);
        
        // Step 5: If the average off-diagonal correlation is above the threshold,
        // increase candidateAR (or candidateMA) as additional structure might be present.
        if (avgOffDiag > eafThreshold) {
            candidateAR++;  // Increase AR order
        } else {
            adjust = 0;  // No adjustment needed
        }
        
        free(fitted);
        free(residuals);
        iterations++;
    }
    
    // Return the final orders.
    *finalAR = candidateAR;
    *finalMA = candidateMA; // In this example, we did not adjust MA order.
}


/**
 * @brief Forecasts future values using a generalized ARIMA(p,d,q) model.
 *
 * @param series       The original time series data.
 * @param seriesLength The number of observations in the series.
 * @param p            AR order.
 * @param d            Differencing order.
 * @param q            MA order.
 * @return Pointer to a dynamically allocated forecast array (of length FORECAST_HORIZON+1).
 *
 * This function first differences the series d times. Then if p>0 it constructs
 * an AR model by regressing the differenced series on its p lagged values. If q>0,
 * it uses a generalized adaptive gradient descent to estimate q MA parameters (plus an intercept)
 * based on the AR residuals. Forecasts are produced recursively assuming future shocks are zero.
 * Finally, if d>0 the forecast is integrated back to the original scale.
 */
double *forecastARIMA(double series[], int seriesLength, int p, int d, int q) {
    // ----- Step 1. Difference the series d times -----
    int currentLength = seriesLength;
    double *currentSeries = malloc(sizeof(double) * currentLength);
    if (!currentSeries) {
        fprintf(stderr, "Memory allocation error in forecastARIMA.\n");
        exit(EXIT_FAILURE);
    }
    memcpy(currentSeries, series, sizeof(double)*currentLength);
    for (int i = 0; i < d; i++) {
        double *temp = differenceSeries(currentSeries, currentLength, 1);
        free(currentSeries);
        currentSeries = temp;
        currentLength--; // each differencing reduces length by 1
    }
    
    // ----- Step 2. Estimate AR coefficients (if p > 0) using OLS -----
    int m = currentLength - p;  // number of observations usable for AR regression
    if (p > 0 && m < (p + 1)) {
        fprintf(stderr, "Error: Not enough observations for AR regression (m = %d, p = %d).\n", m, p);
        exit(EXIT_FAILURE);
    }

    double *arEstimates = NULL; // will be of length (p+1): first p coefficients then intercept
    if (p > 0 && m > 0) {
        // Build design matrix X (m x p) and response vector Y (m x 1)
        double X_mat[m][p];
        double Y_mat[m][1];
        for (int t = p; t < currentLength; t++) {
            Y_mat[t - p][0] = currentSeries[t];
            for (int j = 0; j < p; j++) {
                X_mat[t - p][j] = currentSeries[t - j - 1];
            }
        }
        arEstimates = performMultivariateLinearRegression(m, p, X_mat, Y_mat);
        // arEstimates[0..p-1] are AR coefficients; arEstimates[p] is the intercept.
    }
    
    // ----- Step 3. Compute in-sample AR residuals (if MA part is needed) -----
    double *arResiduals = NULL;
    if (q > 0 && p > 0 && m > 0) {
        arResiduals = malloc(sizeof(double) * m);
        if (!arResiduals) {
            fprintf(stderr, "Memory allocation error in forecastARIMA (arResiduals).\n");
            exit(EXIT_FAILURE);
        }
        for (int t = p; t < currentLength; t++) {
            double pred = arEstimates[p]; // intercept
            for (int j = 0; j < p; j++) {
                pred += arEstimates[j] * currentSeries[t - j - 1];
            }
            arResiduals[t - p] = currentSeries[t] - pred;
        }
    }
    
    // ----- Step 4. Estimate MA parameters (if q > 0) via adaptive gradient descent -----
    // The MA model is: residual[t] ≈ (MA intercept) + sum_{j=1}^{q} theta_j * residual[t - j]
    // We use the in-sample residuals for t = q ... (m-1) to estimate a (q+1)-dimensional parameter vector.
    double *maEstimates = NULL; // length q+1: [theta_1, theta_2, ..., theta_q, MA intercept]
    if (q > 0 && m > q) {
        int n_ma = m - q;  // number of observations for MA regression
        // Build design matrix for MA estimation: MA_X[i][j] = arResiduals[i + q - j - 1] for i=0,...,n_ma-1, j=0,...,q-1.
        double **MA_X = malloc(n_ma * sizeof(double *));
        double *MA_Y = malloc(n_ma * sizeof(double));
        if (!MA_X || !MA_Y) {
            fprintf(stderr, "Memory allocation error in forecastARIMA (MA estimation).\n");
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < n_ma; i++) {
            MA_X[i] = malloc(q * sizeof(double));
            if (!MA_X[i]) {
                fprintf(stderr, "Memory allocation error in forecastARIMA (MA_X[%d]).\n", i);
                exit(EXIT_FAILURE);
            }
        }
        for (int i = 0; i < n_ma; i++) {
            MA_Y[i] = arResiduals[i + q];  // target residual
            for (int j = 0; j < q; j++) {
                MA_X[i][j] = arResiduals[i + q - j - 1]; // predictor: past residuals
            }
        }
        // Now estimate parameters using adaptive gradient descent.
        maEstimates = malloc(sizeof(double) * (q + 1));
        if (!maEstimates) {
            fprintf(stderr, "Memory allocation error in forecastARIMA (maEstimates).\n");
            exit(EXIT_FAILURE);
        }
        // Initialize parameters to 0.
        for (int j = 0; j < q + 1; j++) {
            maEstimates[j] = 0.0;
        }
        double learningRate = INITIAL_MA_LEARNING_RATE;
        int iter = 0;
        double currentObjective = 0.0, newObjective = 0.0;
        // Compute initial objective: sum_{i=0}^{n_ma-1} [MA_Y[i] - (maInter + sum_{j=0}^{q-1} theta_j * MA_X[i][j])]^2.
        for (int i = 0; i < n_ma; i++) {
            double pred = maEstimates[q]; // intercept is last element
            for (int j = 0; j < q; j++) {
                pred += maEstimates[j] * MA_X[i][j];
            }
            double err = MA_Y[i] - pred;
            currentObjective += err * err;
        }
        while (iter < MAX_ITERATIONS) {
            double *grad = malloc(sizeof(double) * (q + 1));
            if (!grad) {
                fprintf(stderr, "Memory allocation error in forecastARIMA (grad).\n");
                exit(EXIT_FAILURE);
            }
            for (int j = 0; j < q + 1; j++) {
                grad[j] = 0.0;
            }
            // Compute gradient.
            for (int i = 0; i < n_ma; i++) {
                double pred = maEstimates[q];
                for (int j = 0; j < q; j++) {
                    pred += maEstimates[j] * MA_X[i][j];
                }
                double err = MA_Y[i] - pred;
                for (int j = 0; j < q; j++) {
                    grad[j] += -2.0 * err * MA_X[i][j];
                }
                grad[q] += -2.0 * err;
            }
            // Compute norm of gradient.
            double gradNorm = 0.0;
            for (int j = 0; j < q + 1; j++) {
                gradNorm += grad[j] * grad[j];
            }
            gradNorm = sqrt(gradNorm);
            if (gradNorm < CONVERGENCE_TOLERANCE) {
                free(grad);
                break;
            }
            double *proposed = malloc(sizeof(double) * (q + 1));
            if (!proposed) {
                fprintf(stderr, "Memory allocation error in forecastARIMA (proposed).\n");
                exit(EXIT_FAILURE);
            }
            for (int j = 0; j < q + 1; j++) {
                proposed[j] = maEstimates[j] - learningRate * grad[j];
            }
            newObjective = 0.0;
            for (int i = 0; i < n_ma; i++) {
                double pred = proposed[q];
                for (int j = 0; j < q; j++) {
                    pred += proposed[j] * MA_X[i][j];
                }
                double err = MA_Y[i] - pred;
                newObjective += err * err;
            }
            if (newObjective < currentObjective) {
                for (int j = 0; j < q + 1; j++) {
                    maEstimates[j] = proposed[j];
                }
                currentObjective = newObjective;
                learningRate *= 1.1;
            } else {
                learningRate *= 0.5;
                if (learningRate < MIN_MA_LEARNING_RATE) {
                    free(proposed);
                    free(grad);
                    break;
                }
            }
            free(proposed);
            free(grad);
            iter++;
        }
        // Free MA design memory.
        for (int i = 0; i < n_ma; i++) {
            free(MA_X[i]);
        }
        free(MA_X);
        free(MA_Y);
    }
    
    // ----- Step 5. Produce forecasts on the differenced scale using the AR part -----
    // For ARMA forecasting the standard assumption is that future shocks (and hence the MA terms) have zero mean.
    // Thus the forecast becomes essentially the AR part plus (if estimated) the MA intercept.
    double *forecast = malloc(sizeof(double) * FORECAST_ARRAY_SIZE);
    if (!forecast) {
        fprintf(stderr, "Memory allocation error in forecastARIMA (forecast).\n");
        exit(EXIT_FAILURE);
    }
    // One-step ahead forecast: use last p observations.
    double oneStep = 0.0;
    if (p > 0) {
        oneStep = arEstimates ? arEstimates[p] : 0.0;
        for (int j = 0; j < p; j++) {
            oneStep += arEstimates[j] * currentSeries[currentLength - j - 1];
        }
    } else {
        oneStep = currentSeries[currentLength - 1];
    }
    if (q > 0) {
        oneStep += maEstimates[q]; // add estimated MA intercept if available
    }
    forecast[0] = oneStep;
    
    // Recursive forecast for h = 2 to FORECAST_HORIZON:
    // We use a simple recursion with the AR part; future MA terms vanish.
    for (int h = 1; h < FORECAST_HORIZON; h++) {
        double f = 0.0;
        if (p > 0) {
            f = arEstimates ? arEstimates[p] : 0.0; // AR intercept
            // For lags: if forecast value exists use it; otherwise use the last available observation.
            for (int j = 0; j < p; j++) {
                double value;
                if (h - j - 1 >= 0) {
                    value = forecast[h - j - 1];
                } else {
                    value = currentSeries[currentLength + h - j - 1];
                }
                f += arEstimates[j] * value;
            }
        } else {
            f = currentSeries[currentLength - 1];
        }
        if (q > 0) {
            f += maEstimates[q];
        }
        forecast[h] = f;
    }
    // Optionally store a placeholder (e.g. forecast variance) at forecast[FORECAST_HORIZON].
    forecast[FORECAST_HORIZON] = 0.0;
    
    // ----- Step 6. If differencing was applied, integrate the forecast back to original scale -----
    if (d > 0) {
        // For simplicity we assume that the last observed value of the original series is used as recovery.
        double recoveryValue = series[seriesLength - 1];
        double *integrated = integrateSeries(forecast, recoveryValue, FORECAST_HORIZON);
        for (int i = 0; i < FORECAST_HORIZON; i++) {
            forecast[i] = integrated[i];
        }
        free(integrated);
    }
    
    // ----- Cleanup -----
    free(currentSeries);
    if (arEstimates) free(arEstimates);
    if (maEstimates) free(maEstimates);
    if (arResiduals) free(arResiduals);
    
    return forecast;
}

/*========================================================================
  Main Function (for testing purposes)
========================================================================*/

int main(void)
{
  double sampleData[] = {10.544653, 10.688583, 10.666841, 10.662732, 10.535033, 10.612065, 10.577628, 10.524487, 10.511290, 10.520899, 10.605484, 10.506456, 10.693456, 10.667562, 10.640863, 10.553473, 10.684760, 10.752397, 10.671068, 10.667091, 10.641893, 10.625706, 10.701795, 10.607544, 10.689169, 10.695256, 10.717050, 10.677475, 10.691141, 10.730298, 10.732664, 10.710082, 10.713123, 10.759815, 10.696599, 10.663845, 10.716597, 10.780855, 10.795759, 10.802620, 10.720496, 10.753401, 10.709436, 10.746909, 10.737377, 10.754609, 10.765248, 10.692602, 10.837926, 10.755324, 10.756213, 10.843190, 10.862529, 10.751269, 10.902390, 10.817731, 10.859796, 10.887362, 10.835401, 10.824412, 10.860767, 10.819504, 10.907496, 10.831528, 10.821727, 10.830010, 10.915317, 10.858694, 10.921139, 10.927524, 10.894352, 10.889785, 10.956356, 10.938758, 11.093567, 10.844841, 11.094493, 11.035941, 10.982765, 11.071057, 10.996308, 11.099276, 11.142057, 11.137176, 11.157537, 11.007247, 11.144075, 11.183029, 11.172096, 11.164571, 11.192833, 11.227109, 11.141589, 11.311490, 11.239783, 11.295933, 11.199566, 11.232262, 11.333208, 11.337874, 11.322334, 11.288216, 11.280459, 11.247973, 11.288277, 11.415095, 11.297583, 11.360763, 11.288338, 11.434631, 11.456051, 11.578981, 11.419166, 11.478404, 11.660141, 11.544303, 11.652028, 11.638368, 11.651792, 11.621518, 11.763853, 11.760687, 11.771138, 11.678104, 11.783163, 11.932094, 11.948678, 11.962627, 11.937934, 12.077570, 11.981595, 12.096366, 12.032683, 12.094221, 11.979764, 12.217793, 12.235930, 12.129859, 12.411867, 12.396301, 12.413920, 12.445867, 12.480462, 12.470674, 12.537774, 12.562252, 12.810248, 12.733546, 12.861890, 12.918012, 13.033087, 13.245610, 13.184196, 13.414342, 13.611838, 13.626345, 13.715446, 13.851129, 14.113374, 14.588537, 14.653982, 15.250756, 15.618371, 16.459558, 18.144264, 23.523062, 40.229511, 38.351265, 38.085281, 37.500885, 37.153946, 36.893066, 36.705956, 36.559536, 35.938847, 36.391586, 36.194046, 36.391586, 36.119102, 35.560543, 35.599018, 34.958851, 35.393860, 34.904797, 35.401318, 34.863518, 34.046680, 34.508522, 34.043182, 34.704235, 33.556644, 33.888481, 33.533638, 33.452129, 32.930935, 32.669731, 32.772537, 32.805634, 32.246761, 32.075809, 31.864927, 31.878294, 32.241131, 31.965626, 31.553604, 30.843288, 30.784569, 31.436094, 31.170496, 30.552132, 30.500242, 30.167421, 29.911989, 29.586046, 29.478958, 29.718994, 29.611095, 29.557945, 28.463432, 29.341291, 28.821512, 28.447210, 27.861872, 27.855633, 27.910660, 28.425800, 27.715517, 27.617193, 27.093372, 26.968832, 26.977205, 27.170172, 26.251677, 26.633236, 26.224941, 25.874708, 25.593761, 26.392395, 24.904768, 25.331600, 24.530737, 25.074808, 25.310865, 24.337013, 24.442986, 24.500193, 24.130409, 24.062714, 24.064592, 23.533037, 23.977909, 22.924667, 22.806379, 23.130791, 22.527645, 22.570505, 22.932512, 22.486126, 22.594856, 22.383926, 22.115181, 22.105082, 21.151754, 21.074114, 21.240192, 20.977468, 20.771507, 21.184586, 20.495111, 20.650751, 20.656075, 20.433039, 20.005697, 20.216360, 19.982117, 19.703951, 19.572884, 19.332155, 19.544645, 18.666328, 19.219872, 18.934229, 19.186989, 18.694986, 18.096903, 18.298306, 17.704309, 18.023785, 18.224157, 18.182484, 17.642824, 17.739542, 17.474176, 17.270575, 17.604120, 17.631210, 16.639175, 17.107626, 17.024216, 16.852285, 16.780111, 16.838861, 16.539309, 16.092861, 16.131529, 16.221350, 16.087164, 15.821659, 15.695448, 15.693087, 16.047991, 15.682863, 15.724131, 15.263708, 15.638486, 15.443835, 15.602257, 15.122874, 14.918172, 14.968882, 14.843689, 14.861169, 15.052527, 15.056897, 14.690192, 14.686479, 14.567565, 14.365212, 14.253309, 14.289158, 14.227124, 14.069589, 14.074703, 13.869432, 13.861959, 13.782178, 13.882711, 13.908362, 13.727641, 13.600214, 13.594969, 13.535290, 13.602018, 13.502626, 13.579159, 13.207825, 13.426789, 13.178141, 13.286413, 12.958746, 13.189507, 13.079733, 13.138372, 12.986096, 12.854589, 12.858962, 12.903029, 12.852099, 12.644394, 12.558786, 12.636994};

  int dataLength = sizeof(sampleData) / sizeof(sampleData[0]);
  
  
  // --- Step 1: Use selectOrdersWithFeedback to determine candidate AR and MA orders ---
  int finalAR, finalMA;
  int maxLag = 4;  // You can adjust this maximum lag
  selectOrdersWithFeedback(sampleData, dataLength, maxLag, &finalAR, &finalMA);
  printf("Selected orders (via feedback): AR = %d, MA = %d\n", finalAR, finalMA);
  
  // --- (Optional) Step 1b: Perform an ADF test on the series ---
  // Uncomment the lines below if you want to run the ADF test.
  /*
  double tStat, pValue;
  int isStationary = ADFTestExtendedAutoLag(sampleData, dataLength, MODEL_CONSTANT_ONLY, &tStat, &pValue);
  printf("ADF test: tStat = %lf, pValue = %lf, Stationary = %d\n", tStat, pValue, isStationary);
  */

  // Forecast using AR(1)
  double *ar1Forecast = forecastAR1(sampleData, 183);
  printf("AR(1) Forecast: ");
  for (int i = 0; i < 17; i++)
  {
    printf("%lf ", ar1Forecast[i]);
  }
  printf("\n");
  free(ar1Forecast);


    // Forecast using the generalized ARIMA function with AR(1)-MA(1)
    // (Here we set p = 1, d = 0, and q = 1.)
    double *armaForecast = forecastARIMA(sampleData, 181, 2, 0, 4);
    printf("Generalized ARIMA (AR(1)-MA(1)) Forecast:\n");
    for (int i = 0; i < FORECAST_HORIZON; i++) {
        printf("%lf ", armaForecast[i]);
    }
    printf("\n");

    free(armaForecast);

  // Additional forecasting models (e.g., AR(2)-MA(1), AR(2)-MA(2)) can be tested here.

  return 0;
}
