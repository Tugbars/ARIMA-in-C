#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <complex.h>

/*==================== Defined Constants ====================*/

// Used to check when iterative methods like Newton-Raphson converge; 0.01 ensures reasonable precision without too many iterations.
#define CONVERGENCE_TOLERANCE 0.01

// Forces at least 30 iterations in optimization to avoid stopping too early; ensures stability before convergence check.
#define MIN_ITERATIONS 30

// Sets a tolerance of 1.001 for unit circle checks in AR/MA roots; slightly above 1 to allow small numerical errors.
#define UNIT_TOLERANCE 1.001

// Caps iterative algorithms at 300 steps to prevent infinite loops; balances computation time with convergence needs.
#define MAX_ITERATIONS 300

// Defines a threshold (1e-12) for detecting near-zero pivots in matrix operations; prevents division-by-zero errors.
#define SINGULARITY_THRESHOLD 1e-12

// Sets forecast length to 16 steps; chosen for typical short-to-medium term forecasting horizons in time series.
#define FORECAST_HORIZON 16

// Allocates array size as horizon + 2 (16 + 2 = 18) for forecast plus extra slots (e.g., variance); ensures sufficient space.
#define FORECAST_ARRAY_SIZE (FORECAST_HORIZON + 2)

// Critical value (-3.5) for ADF test to reject unit root; approximates common statistical threshold for stationarity.
#define ADF_CRITICAL_VALUE -3.5

// Exponent (1/3) for ADF lag selection (floor((n-1)^(1/3))); standard heuristic to balance lag length with sample size.
#define ADF_LAG_EXPONENT (1.0 / 3.0)

// Limits Newton-Raphson to 500 iterations for MA estimation; increased from default to improve convergence on complex models.
#define MAX_NEWTON_ITER 500

// Convergence tolerance (1e-6) for Newton-Raphson step size; small value ensures precise MA parameter estimates.
#define NEWTON_TOL 1e-6

// Threshold (1.001) for AR/MA root magnitude checks; slightly above 1 to account for numerical precision in stability tests.
#define ROOT_TOLERANCE 1.001

#ifdef DEBUG
#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define DEBUG_PRINT(...)
#endif

/*==================== Data Structures =====================*/

typedef struct {
    double ar[3];
    double ma[3];
    double intercept;
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

/*==================== QR Inversion ====================*/
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
    
    // Compute initial column norms.
    for (j = 0; j < n; j++) {
        jpvt[j] = j;
        double sum = 0.0;
        for (i = 0; i < m; i++) {
            double aij = A[IDX(i, j, lda)];
            sum += aij * aij;
        }
        norms[j] = sqrt(sum);
        norms_updated[j] = norms[j];
        if (norms[j] < SINGULARITY_THRESHOLD) {
            fprintf(stderr, "Warning: Column %d is nearly singular (norm = %e).\n", j, norms[j]);
            // Optionally, you can set the norm to the threshold to avoid division by nearly zero.
            norms[j] = SINGULARITY_THRESHOLD;
            norms_updated[j] = SINGULARITY_THRESHOLD;
        }
    }
    
    // Main factorization loop.
    for (k = 0; k < n && k < m; k++) {
        // Column pivoting: find the column with the maximum updated norm.
        int max_index = k;
        for (j = k; j < n; j++) {
            if (norms_updated[j] > norms_updated[max_index])
                max_index = j;
        }
        if (max_index != k) {
            // Swap columns k and max_index.
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
        
        // Compute the 2–norm of the k–th column from row k onward.
        double norm_x = 0.0;
        for (i = k; i < m; i++) {
            norm_x += A[IDX(i, k, lda)] * A[IDX(i, k, lda)];
        }
        norm_x = sqrt(norm_x);
        
        // Robust handling: if the pivot norm is nearly zero, warn and set pivot to zero.
        if (norm_x < SINGULARITY_THRESHOLD) {
            fprintf(stderr, "Warning: Nearly singular pivot encountered at column %d (norm = %e). Setting pivot to zero.\n", k, norm_x);
            A[IDX(k, k, lda)] = 0.0;
            continue;  // Skip reflector computation for this column.
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
        if (norm_v < SINGULARITY_THRESHOLD) {
            fprintf(stderr, "Warning: Householder vector nearly zero at column %d.\n", k);
            free(v);
            continue;
        }
        // Normalize the Householder vector.
        for (i = 0; i < m - k; i++) {
            v[i] /= norm_v;
        }
        // Store the reflector.
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
            // Reorthogonalization pass to improve stability.
            dot = 0.0;
            for (i = k; i < m; i++) {
                double vi = (i == k) ? 1.0 : A[IDX(i, k, lda)];
                dot += vi * A[IDX(i, j, lda)];
            }
            for (i = k; i < m; i++) {
                double vi = (i == k) ? 1.0 : A[IDX(i, k, lda)];
                A[IDX(i, j, lda)] -= 2 * vi * dot;
            }
            // Update the norm.
            double new_norm_sq = 0.0;
            for (i = k+1; i < m; i++) {
                new_norm_sq += A[IDX(i, j, lda)] * A[IDX(i, j, lda)];
            }
            double new_norm = sqrt(new_norm_sq);
            // If the new norm is very small, warn and set it to zero.
            if (new_norm < 0.1 * norms[j]) {
                if (new_norm < SINGULARITY_THRESHOLD) {
                    fprintf(stderr, "Warning: Updated norm for column %d is nearly singular (new_norm = %e).\n", j, new_norm);
                    new_norm = 0.0;
                } else {
                    new_norm_sq = 0.0;
                    for (i = k+1; i < m; i++) {
                        new_norm_sq += A[IDX(i, j, lda)] * A[IDX(i, j, lda)];
                    }
                    new_norm = sqrt(new_norm_sq);
                }
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
                if (new_norm < SINGULARITY_THRESHOLD) {
                    fprintf(stderr, "Warning: Updated norm for column %d is nearly singular (new_norm = %e).\n", j, new_norm);
                    new_norm = 0.0;
                } else {
                    new_norm_sq = 0.0;
                    for (i = k+1; i < m; i++) {
                        new_norm_sq += A[IDX(i, j, lda)] * A[IDX(i, j, lda)];
                    }
                    new_norm = sqrt(new_norm_sq);
                }
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


/*==================== Linear Regression ====================*/

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
double *performUnivariateLinearRegression(double predictor[], double response[], int length) {
    double predictorDiff[length], responseDiff[length];
    double meanPredictor = calculateMean(predictor, length);
    double meanResponse = calculateMean(response, length);
    for (int i = 0; i < length; i++) {
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
    if (!estimates) exit(EXIT_FAILURE);
    estimates[0] = slope;
    estimates[1] = intercept;
    return estimates;
}

void predictUnivariate(double predictor[], double predictions[], double slope, double intercept, int length) {
    for (int i = 0; i < length; i++) predictions[i] = predictor[i] * slope + intercept;
}

/**
 * @brief Performs multivariate linear regression to estimate coefficients for multiple predictors.
 *
 * @details 
 * **Purpose**: This function estimates the coefficients (betas) of a linear model where multiple predictors 
 * influence a single response variable. It’s essential for ARIMA’s AR parameter estimation and other 
 * multivariate fits, providing a way to model relationships between lagged values and the current value.
 *
 * **Method Used**: The method employs the normal equations approach, solving \( (X^T X) \beta = X^T Y \) 
 * using QR decomposition for matrix inversion. Regularization is added to \( X^T X \) to handle 
 * ill-conditioned matrices. The process involves:
 * - Normalizing the design matrix \( X \) and response \( Y \) by subtracting column means.
 * - Computing \( X^T X \) and \( X^T Y \).
 * - Inverting \( X^T X \) with QR decomposition and regularization.
 * - Solving for \( \beta \) and adjusting for the intercept.
 *
 * **Why This Method**: 
 * - **Normal Equations**: Efficient for small to medium-sized problems, leveraging matrix algebra to find 
 *   the least squares solution directly.
 * - **QR Decomposition**: Chosen for numerical stability over direct inversion (e.g., Gaussian elimination), 
 *   as it avoids issues with near-singular matrices by decomposing \( X^T X \) into orthogonal (\( Q \)) 
 *   and upper triangular (\( R \)) components.
 * - **Regularization**: Adds a small constant \( \lambda \) to the diagonal of \( X^T X \) to ensure 
 *   invertibility, addressing multicollinearity or singularity common in time series data with high lags.
 *
 * **Downsides and Limitations**:
 * - **Computational Cost**: QR decomposition scales as \( O(n^3) \) for an \( n \times n \) matrix, 
 *   becoming inefficient for large \( n \) (many predictors).
 * - **Regularization Bias**: Adding \( \lambda \) introduces a slight bias in \( \beta \), potentially 
 *   skewing estimates away from the true least squares solution.
 * - **Numerical Precision**: For very ill-conditioned matrices, even QR with regularization may fail to 
 *   produce accurate inverses, leading to unstable coefficients.
 * - **Memory Usage**: Variable-length arrays (VLAs) for large datasets can exceed stack size, risking 
 *   undefined behavior or crashes.
 *
 * @param numObservations Number of observations (rows in \( X \) and \( Y \)).
 * @param numPredictors Number of predictor variables (columns in \( X \)).
 * @param X Design matrix where each row is an observation and each column a predictor.
 * @param Y Response vector as a single-column matrix.
 * @return Pointer to an array of coefficients [beta_1, ..., beta_p, intercept], dynamically allocated; 
 *         caller must free it.
 */
double *performMultivariateLinearRegression(int numObservations, int numPredictors, 
                                            double X[][numPredictors], double Y[][1]) {
    // Declare VLAs for normalized matrices and intermediate results
    double X_normalized[numObservations][numPredictors], Y_normalized[numObservations][1];
    double X_means[1][numPredictors], Y_mean[1][1];
    double Xt[numPredictors][numObservations], XtX[numPredictors][numPredictors];
    double XtX_inv[numPredictors][numPredictors];
    double XtX_inv_Xt[numPredictors][numObservations], beta[numPredictors][1];

    // Allocate memory for the output coefficients (p betas + 1 intercept)
    double *estimates = malloc(sizeof(double) * (numPredictors + 1));
    if (!estimates) {
        fprintf(stderr, "Memory allocation error in performMultivariateLinearRegression.\n");
        exit(EXIT_FAILURE);
    }

    // Step 1: Normalize X and Y by subtracting column means to center the data
    // Mathematically: X_normalized[i][j] = X[i][j] - mean(X[:,j])
    // This ensures the intercept can be computed separately, simplifying the regression
    normalize2DArray(numObservations, numPredictors, X, X_normalized);
    normalize2DArray(numObservations, 1, Y, Y_normalized);

    // Step 2: Compute the transpose of X_normalized (Xt = X^T)
    // This prepares for forming the normal equations: XtX = X^T * X
    transposeMatrix(numObservations, numPredictors, X_normalized, Xt);

    // Step 3: Compute XtX = X^T * X
    // Mathematically: XtX[i][j] = sum(Xt[i][k] * X_normalized[k][j]) over k
    // This matrix represents the covariance structure of predictors
    matrixMultiply(numPredictors, numObservations, numPredictors, Xt, X_normalized, XtX);

    // Step 4: Add regularization to XtX to ensure invertibility
    // Mathematically: XtX[i][i] += lambda, where lambda is a small positive constant
    // This prevents singularity when predictors are highly correlated
    double lambda = 1e-6;
    for (int i = 0; i < numPredictors; i++) {
        XtX[i][i] += lambda;
    }

    // Step 5: Invert XtX based on the number of predictors (n)
    int n = numPredictors;
    if (n == 1) {
        // For a single predictor, inversion is trivial: 1 / XtX[0][0]
        XtX_inv[0][0] = 1.0 / XtX[0][0];
    } else if (n == 2) {
        // For 2 predictors, use the analytical 2x2 inversion formula
        // Given XtX = [[a, b], [c, d]], inverse is (1/det) * [[d, -b], [-c, a]]
        double a = XtX[0][0], b = XtX[0][1], c = XtX[1][0], d = XtX[1][1];
        double det = a * d - b * c;
        if (fabs(det) < 1e-12) {
            // Regularization should prevent this, but adjust det if near-zero
            det += lambda;
        }
        XtX_inv[0][0] = d / det;
        XtX_inv[0][1] = -b / det;
        XtX_inv[1][0] = -c / det;
        XtX_inv[1][1] = a / det;
    } else {
        // For n >= 3, use QR decomposition for stability
        // Allocate a column-major matrix for invertMatrixQR
        double *tempMatrix = malloc(n * n * sizeof(double));
        if (!tempMatrix) {
            fprintf(stderr, "Memory allocation error in performMultivariateLinearRegression.\n");
            free(estimates);
            exit(EXIT_FAILURE);
        }
        // Convert XtX to column-major format: tempMatrix[i + j*n] = XtX[i][j]
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                tempMatrix[IDX(i, j, n)] = XtX[i][j];
            }
        }
        // Invert using QR decomposition
        double *invTemp = invertMatrixQR(tempMatrix, n);
        if (!invTemp) {
            fprintf(stderr, "Error: QR inversion failed for n=%d.\n", n);
            free(tempMatrix);
            free(estimates);
            exit(EXIT_FAILURE);
        }
        // Copy back to row-major XtX_inv
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                XtX_inv[i][j] = invTemp[IDX(i, j, n)];
            }
        }
        free(invTemp);
        free(tempMatrix);
    }

    // Step 6: Compute XtX_inv * Xt = (X^T X)^(-1) * X^T
    // This is part of solving beta = (X^T X)^(-1) * X^T * Y
    matrixMultiply(numPredictors, numPredictors, numObservations, XtX_inv, Xt, XtX_inv_Xt);

    // Step 7: Compute beta = (X^T X)^(-1) * X^T * Y
    // Mathematically: beta = XtX_inv_Xt * Y_normalized
    matrixMultiply(numPredictors, numObservations, 1, XtX_inv_Xt, Y_normalized, beta);
    for (int i = 0; i < numPredictors; i++) {
        estimates[i] = beta[i][0]; // Store coefficients
    }

    // Step 8: Compute the intercept using original means
    // Mathematically: intercept = mean(Y) - sum(beta[i] * mean(X[:,i]))
    for (int j = 0; j < numPredictors; j++) {
        double col[numObservations];
        for (int i = 0; i < numObservations; i++) col[i] = X[i][j];
        X_means[0][j] = calculateMean(col, numObservations);
    }
    double yCol[numObservations];
    for (int i = 0; i < numObservations; i++) yCol[i] = Y[i][0];
    Y_mean[0][0] = calculateMean(yCol, numObservations);
    double intercept = Y_mean[0][0];
    for (int i = 0; i < numPredictors; i++) {
        intercept -= estimates[i] * X_means[0][i];
    }
    estimates[numPredictors] = intercept;

    return estimates;
}

/*==================== Stationarity and ARIMA ====================*/
/**
 * @brief Differences a time series to remove trends or non-stationarity.
 *
 * @details 
 * **Purpose**: This function applies differencing to a time series, a key step in ARIMA’s "I" 
 * (Integrated) component, to transform a non-stationary series into a stationary one by computing 
 * successive differences.
 *
 * **Method Used**: Iterative first-order differencing is applied \( d \) times. For each iteration:
 * - Compute \( y_t' = y_{t+1} - y_t \) for \( t = 0 \) to \( n-2 \), reducing the series length by 1.
 * - Repeat \( d \) times to achieve the specified order of differencing.
 *
 * **Why This Method**: 
 * - **Simplicity**: First-order differencing is a straightforward way to remove linear trends, and 
 *   higher orders can address polynomial trends of degree \( d \).
 * - **ARIMA Compatibility**: Directly supports the ARIMA model by preparing the series for AR and MA 
 *   estimation on a stationary basis.
 *
 * **Downsides and Limitations**:
 * - **Over-Differencing**: Applying too many differences (\( d \) too high) can introduce artificial 
 *   noise or overdampen the series, losing important information.
 * - **Data Loss**: Each differencing reduces the series length by 1, which can be problematic for 
 *   short series.
 * - **Assumption**: Assumes differencing alone can achieve stationarity, which may not hold for 
 *   series with seasonality or structural breaks.
 *
 * @param series The input time series array.
 * @param length The number of observations in the input series.
 * @param d The order of differencing to apply.
 * @return Pointer to the differenced series (length = original length - d); caller must free it.
 */
double *differenceSeries(const double series[], int length, int d) {
    // Validate input to prevent invalid differencing
    if (d < 0 || d >= length) {
        fprintf(stderr, "Error: Invalid differencing order d=%d for length %d.\n", d, length);
        exit(EXIT_FAILURE);
    }

    // Initialize working copy of the series
    int currentLength = length;
    double *current = malloc(sizeof(double) * currentLength);
    if (!current) {
        fprintf(stderr, "Memory allocation error in differenceSeries.\n");
        exit(EXIT_FAILURE);
    }
    // Copy original series to avoid modifying input
    copyArray(series, current, currentLength);

    // Apply differencing d times
    for (int diff = 0; diff < d; diff++) {
        int newLength = currentLength - 1; // Each difference reduces length by 1
        double *temp = malloc(sizeof(double) * newLength);
        if (!temp) {
            fprintf(stderr, "Memory allocation error in differenceSeries.\n");
            free(current);
            exit(EXIT_FAILURE);
        }
        // Compute first-order difference: temp[i] = current[i+1] - current[i]
        for (int i = 0; i < newLength; i++) {
            temp[i] = current[i + 1] - current[i];
        }
        free(current); // Free previous series
        current = temp; // Update to new differenced series
        currentLength = newLength;
    }

    return current; // Return the d-th differenced series
}

/**
 * @brief Integrates a differenced series to recover the original scale for forecasting.
 *
 * @details 
 * **Purpose**: This function reverses the differencing process applied in ARIMA to convert forecasts 
 * from the differenced scale back to the original scale, ensuring interpretable predictions.
 *
 * **Method Used**: Cumulative summation is used to integrate the differenced forecast:
 * - Start with the last observed value of the original series (\( y_{n} \)).
 * - Compute \( forecast[0] = y_{n} + diffForecast[0] \).
 * - For subsequent steps, \( forecast[i] = forecast[i-1] + diffForecast[i] \).
 *
 * **Why This Method**: 
 * - **Direct Reversal**: Cumulative summation is the mathematical inverse of differencing, accurately 
 *   reconstructing the original series level given the differenced forecasts and a starting point.
 * - **Simplicity**: Requires minimal computation and aligns with ARIMA’s integration step.
 *
 * **Downsides and Limitations**:
 * - **Error Propagation**: Any error in the differenced forecast (e.g., from AR/MA estimation) 
 *   accumulates through the summation, potentially amplifying inaccuracies.
 * - **Single Starting Point**: Assumes the last observed value is sufficient to anchor the forecast, 
 *   ignoring potential drift or multiple differencing levels requiring more recovery values.
 * - **Assumption**: Works only for first-order integration; higher-order differencing requires 
 *   additional recovery values (not implemented here).
 *
 * @param diffForecast The forecast values on the differenced scale.
 * @param recoveryValue The last value of the original series to anchor the integration.
 * @param forecastLength Number of forecast steps.
 * @return Pointer to the integrated forecast array; caller must free it.
 */
double *integrateSeries(const double diffForecast[], double recoveryValue, int forecastLength) {
    // Allocate memory for the integrated forecast
    double *integrated = malloc(sizeof(double) * forecastLength);
    if (!integrated) {
        fprintf(stderr, "Memory allocation error in integrateSeries.\n");
        exit(EXIT_FAILURE);
    }

    // Step 1: Initialize first forecast value
    // Mathematically: integrated[0] = y_n + diffForecast[0], where y_n is the last original value
    integrated[0] = recoveryValue + diffForecast[0];

    // Step 2: Cumulatively sum the differenced forecasts
    // For i >= 1: integrated[i] = integrated[i-1] + diffForecast[i]
    // This reconstructs the level by adding each differenced step
    for (int i = 1; i < forecastLength; i++) {
        integrated[i] = integrated[i - 1] + diffForecast[i];
    }

    return integrated;
}


#define MODEL_CONSTANT_ONLY 0
static const double ADF_CRIT_CONSTANT[3] = {-3.43, -2.86, -2.57};

/**
 * @brief Computes an approximate p-value for the ADF test statistic.
 *
 * @details 
 * **Purpose**: This function provides a p-value for the Augmented Dickey-Fuller (ADF) test statistic 
 * to assess stationarity, aiding in determining the appropriate differencing order in ARIMA.
 *
 * **Method Used**: Piecewise linear interpolation between predefined critical values:
 * - Uses critical values at 1%, 5%, and 10% significance levels (-3.43, -2.86, -2.57 for constant-only model).
 * - Maps the test statistic to a p-value: < -3.43 → 0.005, > -2.57 → 0.15, with linear interpolation between.
 *
 * **Why This Method**: 
 * - **Simplicity**: Avoids complex distribution tables or numerical integration, making it lightweight 
 *   and suitable for embedded use in ARIMA.
 * - **Approximation**: Provides a reasonable estimate for decision-making (e.g., p < 0.05) without 
 *   requiring full ADF distribution data.
 *
 * **Downsides and Limitations**:
 * - **Crudeness**: Linear interpolation is a rough approximation; true ADF p-values follow a non-linear 
 *   distribution, potentially leading to inaccurate significance thresholds.
 * - **Limited Range**: Only interpolates between 1% and 10%, underestimating p-values outside this range.
 * - **Model Specificity**: Critical values are for constant-only model; trend or no-constant models need 
 *   different values, limiting flexibility.
 *
 * @param tstat The ADF test statistic (coefficient on lagged level).
 * @param modelType Model type (MODEL_CONSTANT_ONLY = 0, currently only supported).
 * @return Approximate p-value for the test statistic.
 */
double adfPValue(double tstat, int modelType) {
    // Use critical values for constant-only model
    const double *cv = ADF_CRIT_CONSTANT; // [-3.43, -2.86, -2.57]
    const double pvals[3] = {0.01, 0.05, 0.10}; // Corresponding p-values: 1%, 5%, 10%

    // Step 1: Map tstat to p-value ranges
    if (tstat <= cv[0]) return 0.005; // Below 1% critical value
    if (tstat >= cv[2]) return 0.15; // Above 10% critical value

    // Step 2: Linear interpolation between critical values
    if (tstat <= cv[1]) {
        // Between 1% (-3.43) and 5% (-2.86)
        // p = pvals[0] + (tstat - cv[0]) / (cv[1] - cv[0]) * (pvals[1] - pvals[0])
        return pvals[0] + (tstat - cv[0]) / (cv[1] - cv[0]) * (pvals[1] - pvals[0]);
    } else {
        // Between 5% (-2.86) and 10% (-2.57)
        // p = pvals[1] + (tstat - cv[1]) / (cv[2] - cv[1]) * (pvals[2] - pvals[1])
        return pvals[1] + (tstat - cv[1]) / (cv[2] - cv[1]) * (pvals[2] - pvals[1]);
    }
}

/**
 * @brief Computes the sample autocorrelation at a specified lag.
 *
 * @details 
 * **Purpose**: Calculates the autocorrelation function (ACF) at a given lag to measure the linear 
 * relationship between a time series and its lagged values, used in ARIMA order selection and MA estimation.
 *
 * **Method Used**: Standard sample autocorrelation formula:
 * - \( r_k = \frac{\sum_{t=1}^{n-k} (y_t - \bar{y})(y_{t+k} - \bar{y})}{\sum_{t=1}^{n} (y_t - \bar{y})^2} \)
 * - Computes the mean \( \bar{y} \), then numerator and denominator separately.
 *
 * **Why This Method**: 
 * - **Statistical Standard**: Widely accepted for measuring serial correlation in time series, directly 
 *   applicable to ARIMA’s ACF-based diagnostics.
 * - **Efficiency**: Simple computation with \( O(n) \) complexity per lag.
 *
 * **Downsides and Limitations**:
 * - **Stationarity Assumption**: Assumes the series is stationary; non-stationary series yield 
 *   misleading ACF values.
 * - **Sample Size Sensitivity**: For small \( n \), estimates are noisy and less reliable.
 * - **Bias**: Sample ACF can be biased for short series or high lags due to reduced effective sample size.
 *
 * @param series The input time series.
 * @param n Length of the series.
 * @param lag The lag at which to compute autocorrelation.
 * @return Autocorrelation coefficient at the specified lag.
 */
double autocorrelation(const double series[], int n, int lag) {
    // Step 1: Compute the series mean
    double mean = calculateMean(series, n);

    // Step 2: Compute numerator: sum of (y_t - mean) * (y_{t+lag} - mean)
    double num = 0.0;
    for (int i = 0; i < n - lag; i++) {
        num += (series[i] - mean) * (series[i + lag] - mean);
    }

    // Step 3: Compute denominator: sum of squared deviations from mean
    double den = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = series[i] - mean;
        den += diff * diff;
    }

    // Step 4: Return autocorrelation coefficient
    return num / den;
}

/**
 * @brief Performs an Augmented Dickey-Fuller test with automatic lag selection to assess stationarity.
 *
 * @details 
 * **Purpose**: Tests for a unit root in the time series to determine if differencing is needed in ARIMA, 
 * selecting the optimal lag length based on AIC to balance fit and complexity.
 *
 * **Method Used**: 
 * - Fits the ADF regression: \( \Delta y_t = \alpha + \beta y_{t-1} + \sum_{j=1}^{p} \gamma_j \Delta y_{t-j} + \epsilon_t \)
 * - Iterates over lag \( p \) from 0 to a maximum (based on \( \lfloor (n-1)^{1/3} \rfloor \)), 
 *   computing AIC for each model.
 * - Selects the model with minimum AIC, using the coefficient \( \beta \) as the test statistic.
 * - Computes an approximate p-value via interpolation.
 *
 * **Why This Method**: 
 * - **Unit Root Detection**: Essential for ARIMA to ensure stationarity; \( \beta < 0 \) and significant 
 *   indicates no unit root (stationary).
 * - **AIC-Based Lag Selection**: Balances overfitting and underfitting, automating a critical parameter choice.
 * - **OLS Estimation**: Robust and standard for regression-based tests like ADF.
 *
 * **Downsides and Limitations**:
 * - **AIC Bias**: May select suboptimal lags in small samples, favoring overly complex models.
 * - **Power**: ADF has low power against near-unit-root processes, potentially missing stationarity.
 * - **Simplified P-Value**: Interpolation is crude, reducing precision compared to full distribution tables.
 * - **Model Limitation**: Only supports constant-only model; trend or no-constant variants need adjustments.
 *
 * @param series The input time series.
 * @param length Length of the series.
 * @param modelType Model type (MODEL_CONSTANT_ONLY = 0).
 * @param tStat Output pointer for the test statistic (\( \beta \)).
 * @param pValue Output pointer for the approximate p-value.
 * @return 1 if null hypothesis (unit root) is rejected (stationary), 0 otherwise.
 */
int ADFTestExtendedAutoLag(double series[], int length, int modelType, double *tStat, double *pValue) {
    // Step 1: Determine maximum lag based on series length
    // Heuristic: pMax = floor((n-1)^(1/3))
    int pMax = (int)floor(pow(length - 1, ADF_LAG_EXPONENT));
    if (pMax < 0) pMax = 0;

    int bestP = 0;
    double bestAIC = 1e30;
    double bestBeta = 0.0;

    // Step 2: Iterate over possible lags to find best model
    for (int p = 0; p <= pMax; p++) {
        int nEff = length - p - 1; // Effective sample size after lagging
        int k = 2 + p; // Parameters: constant, y_{t-1}, p lagged differences
        if (nEff < k + 1) break; // Ensure enough data points

        // Step 3: Build ADF regression design matrix and response
        double X[nEff][k];
        double Y[nEff];
        for (int i = 0; i < nEff; i++) {
            int t = i + p + 1;
            Y[i] = series[t] - series[t - 1]; // \( \Delta y_t = y_t - y_{t-1} \)
            int col = 0;
            X[i][col++] = 1.0; // Constant term
            X[i][col++] = series[t - 1]; // Lagged level \( y_{t-1} \)
            for (int j = 1; j <= p; j++) {
                X[i][col++] = series[t - j] - series[t - j - 1]; // Lagged differences
            }
        }

        // Step 4: Estimate coefficients using OLS
        double *betaEst = performMultivariateLinearRegression(nEff, k, X, (double(*)[1])Y);

        // Step 5: Compute residual sum of squares (RSS) for AIC
        double RSS = 0.0;
        for (int i = 0; i < nEff; i++) {
            double pred = 0.0;
            for (int j = 0; j < k; j++) pred += betaEst[j] * X[i][j];
            double err = Y[i] - pred;
            RSS += err * err;
        }

        // Step 6: Compute AIC: nEff * log(RSS/nEff) + 2 * k
        double AIC = nEff * log(RSS / nEff) + 2.0 * k;

        // Step 7: Update best model if AIC improves
        if (AIC < bestAIC) {
            bestAIC = AIC;
            bestP = p;
            bestBeta = betaEst[1]; // \( \beta \) is the coefficient on y_{t-1}
        }
        free(betaEst);
    }

    // Step 8: Set outputs and determine stationarity
    *tStat = bestBeta;
    *pValue = adfPValue(bestBeta, modelType);
    return (*pValue < 0.05) ? 1 : 0; // Reject null if p < 0.05
}

/**
 * @brief Ensures a time series is stationary by applying differencing based on ADF test results.
 *
 * @details 
 * **Purpose**: Prepares a time series for ARIMA modeling by differencing it until stationary, 
 * respecting the user-specified differencing order while validating with the ADF test.
 *
 * **Method Used**: 
 * - Applies the ADF test iteratively, differencing the series once each time the test fails 
 * (p-value >= 0.05) up to a maximum order.
 * - If specified \( d \) exceeds the ADF-determined order, continues differencing to match \( d \).
 *
 * **Why This Method**: 
 * - **ADF-Driven**: Ensures stationarity aligns with statistical testing, critical for valid ARIMA modeling.
 * - **Flexibility**: Balances user input (\( d \)) with empirical evidence, avoiding over- or under-differencing.
 *
 * **Downsides and Limitations**:
 * - **ADF Weakness**: Inherits ADF’s low power against near-unit-root processes, potentially stopping 
 *   differencing too early.
 * - **Data Reduction**: Each difference reduces series length, risking insufficient data for high \( p \) or \( q \).
 * - **Single Criterion**: Relies solely on ADF; other tests (e.g., KPSS) could provide complementary insight.
 *
 * @param series The input time series.
 * @param length Pointer to series length (updated after differencing).
 * @param specifiedD The user-specified differencing order.
 * @return Pointer to the stationary series; caller must free it.
 */
double *ensureStationary(double series[], int *length, int specifiedD) {
    double tStat, pValue;
    // Allocate initial copy of the series
    double *currentSeries = malloc(*length * sizeof(double));
    if (!currentSeries) {
        fprintf(stderr, "Memory allocation error in ensureStationary.\n");
        exit(EXIT_FAILURE);
    }
    copyArray(series, currentSeries, *length);

    // Step 1: Determine maximum differencing based on specifiedD or default (2)
    int d = 0;
    int maxD = specifiedD > 0 ? specifiedD : 2;

    // Step 2: Difference until stationary or maxD reached
    while (d < maxD && !ADFTestExtendedAutoLag(currentSeries, *length, MODEL_CONSTANT_ONLY, &tStat, &pValue)) {
        double *temp = differenceSeries(currentSeries, *length, 1);
        free(currentSeries);
        currentSeries = temp;
        (*length)--;
        d++;
        printf("Differenced %d times, tStat = %.4f, pValue = %.4f\n", d, tStat, pValue);
    }

    // Step 3: Apply additional differencing if specifiedD > d
    if (d < specifiedD) {
        for (; d < specifiedD; d++) {
            double *temp = differenceSeries(currentSeries, *length, 1);
            free(currentSeries);
            currentSeries = temp;
            (*length)--;
        }
    }

    return currentSeries;
}

/**
 * @brief Computes the autocorrelation function (ACF) up to a specified maximum lag.
 *
 * @details 
 * **Purpose**: Generates the ACF for use in ARIMA order selection (e.g., MA order via significant lags) 
 * and initial MA parameter estimation, measuring how the series correlates with itself at various lags.
 *
 * **Method Used**: 
 * - Computes the sample ACF using: \( r_k = \frac{\sum_{t=1}^{n-k} (y_t - \bar{y})(y_{t+k} - \bar{y})}{\sum_{t=1}^{n} (y_t - \bar{y})^2} \)
 * - Iterates over lags from 0 to maxLag, storing results in an output array.
 *
 * **Why This Method**: 
 * - **Standard Tool**: ACF is a cornerstone of time series analysis, directly informing ARIMA’s \( q \) 
 *   parameter by identifying significant lags.
 * - **Robustness**: Normalizes by total variance, making it scale-invariant.
 *
 * **Downsides and Limitations**:
 * - **Stationarity**: Requires a stationary series; non-stationary data produces unreliable ACF patterns.
 * - **Lag Limitation**: High lags reduce effective sample size (\( n - k \)), increasing variance in estimates.
 * - **Bias**: Slightly biased downward for small samples due to finite \( n \).
 *
 * @param series The input time series.
 * @param length Length of the series.
 * @param maxLag Maximum lag to compute ACF for.
 * @param acf Output array of autocorrelation coefficients (length maxLag + 1).
 */
void computeACF(const double series[], int length, int maxLag, double acf[]) {
    // Step 1: Compute mean of the series
    double mean = calculateMean(series, length);

    // Step 2: Compute denominator: total variance of the series
    double denom = 0.0;
    for (int i = 0; i < length; i++) {
        double diff = series[i] - mean;
        denom += diff * diff;
    }

    // Step 3: Compute ACF for each lag from 0 to maxLag
    for (int lag = 0; lag <= maxLag; lag++) {
        double num = 0.0;
        // Sum products of deviations for lag k
        for (int i = 0; i < length - lag; i++) {
            num += (series[i] - mean) * (series[i + lag] - mean);
        }
        acf[lag] = num / denom; // Normalize by variance
    }
}

/**
 * @brief Estimates AR coefficients using the Yule-Walker equations.
 *
 * @details 
 * **Purpose**: Provides initial AR parameter estimates for ARIMA by solving the Yule-Walker equations, 
 * which relate autocorrelations to AR coefficients, ensuring a stationary model fit.
 *
 * **Method Used**: 
 * - Constructs a Toeplitz matrix \( R \) from ACF values: \( R[i][j] = r_{|i-j|} \).
 * - Forms the right-hand side vector \( r = [r_1, r_2, ..., r_p] \) from ACF lags.
 * - Solves \( R \phi = r \) using OLS via `performMultivariateLinearRegression`.
 *
 * **Why This Method**: 
 * - **Theoretical Basis**: Yule-Walker equations are derived from the AR process’s stationarity 
 *   properties, ensuring \( \phi \) coefficients match the autocorrelation structure.
 * - **Efficiency**: Leverages existing OLS solver, avoiding custom matrix inversion for small \( p \).
 *
 * **Downsides and Limitations**:
 * - **Stationarity Requirement**: Assumes the series is stationary; non-stationary data yields invalid \( \phi \).
 * - **Small Sample Bias**: ACF estimates are less accurate for short series, affecting \( \phi \).
 * - **Numerical Stability**: Toeplitz matrix may be ill-conditioned for large \( p \) or highly correlated lags.
 *
 * @param series The input time series (assumed stationary).
 * @param length Length of the series.
 * @param p AR order (number of coefficients to estimate).
 * @param phi Output array for AR coefficients (length p).
 */
void yuleWalker(double series[], int length, int p, double phi[]) {
    // Step 1: Compute ACF up to lag p
    double acf[p + 1];
    computeACF(series, length, p, acf);

    // Step 2: Form Toeplitz matrix R and vector r
    // R is symmetric with elements R[i][j] = r_{|i-j|}
    double R[p][p], r[p];
    for (int i = 0; i < p; i++) {
        r[i] = acf[i + 1]; // r[i] = autocorrelation at lag i+1
        for (int j = 0; j < p; j++) {
            R[i][j] = acf[abs(i - j)]; // Fill Toeplitz matrix
        }
    }

    // Step 3: Convert r to 2D array for regression
    double r_2d[p][1];
    for (int i = 0; i < p; i++) {
        r_2d[i][0] = r[i];
    }

    // Step 4: Solve R * phi = r using OLS
    // Mathematically: phi = (R^T R)^(-1) R^T r (normal equations)
    double *phi_est = performMultivariateLinearRegression(p, p, R, r_2d);
    if (!phi_est) {
        fprintf(stderr, "Error in Yule-Walker estimation.\n");
        exit(EXIT_FAILURE);
    }

    // Step 5: Copy estimated coefficients to output
    for (int i = 0; i < p; i++) {
        phi[i] = phi_est[i];
    }
    free(phi_est);
}

/**
 * @brief Computes eigenvalues of an AR or MA polynomial’s companion matrix to check stationarity/invertibility.
 *
 * @details 
 * **Purpose**: Determines the roots of the characteristic polynomial for AR or MA components in ARIMA, 
 * ensuring stationarity (AR: roots outside unit circle) or invertibility (MA: roots outside unit circle).
 *
 * **Method Used**: 
 * - Constructs a companion matrix for the polynomial \( 1 - \phi_1 z - \phi_2 z^2 - ... - \phi_p z^p \) 
 *   (AR) or \( 1 + \theta_1 z + ... + \theta_q z^q \) (MA).
 * - Uses power iteration to approximate eigenvalues, iterating until convergence.
 *
 * **Why This Method**: 
 * - **Companion Matrix**: Transforms polynomial root-finding into an eigenvalue problem, solvable with 
 *   standard linear algebra techniques.
 * - **Power Iteration**: Simple and lightweight for small matrices (low \( p \) or \( q \)), avoiding 
 *   the need for external libraries like LAPACK.
 *
 * **Downsides and Limitations**:
 * - **Approximation**: Power iteration finds the dominant eigenvalue and may miss others or fail for 
 *   complex roots without modification (e.g., deflation).
 * - **Convergence**: Fixed 50 iterations may not suffice for all cases, risking inaccurate roots.
 * - **Stability**: Ill-suited for large \( p \) or \( q \) due to slow convergence or numerical instability.
 *
 * @param coeffs Polynomial coefficients (e.g., \( [\phi_1, \phi_2, ..., \phi_p] \) for AR).
 * @param order Polynomial order (p for AR, q for MA).
 * @param roots Output array for complex roots (length = order).
 * @return 1 if successful, 0 if memory allocation fails.
 */
int computeCompanionEigenvalues(double coeffs[], int order, double complex* roots) {
    if (order <= 0) return 1; // No roots to compute for order 0

    // Step 1: Allocate companion matrix (order x order)
    double *A = calloc(order * order, sizeof(double));
    if (!A) return 0;

    // Step 2: Fill companion matrix
    // Top row: -coeffs (negative for AR polynomial convention)
    // Subdiagonal: 1s to shift terms
    for (int j = 0; j < order; j++) {
        A[j] = -coeffs[j]; // First row: -phi_1, -phi_2, ..., -phi_p
    }
    for (int i = 1; i < order; i++) {
        A[i * order + (i - 1)] = 1.0; // Identity subdiagonal
    }

    // Step 3: Power iteration to find eigenvalues
    for (int k = 0; k < order; k++) {
        double complex lambda = 0.0;
        double v[order], v_new[order];
        // Initialize eigenvector guess with 1s
        for (int i = 0; i < order; i++) v[i] = 1.0;

        // Iterate 50 times to approximate eigenvalue
        for (int iter = 0; iter < 50; iter++) {
            // Matrix-vector multiply: v_new = A * v
            for (int i = 0; i < order; i++) {
                v_new[i] = 0.0;
                for (int j = 0; j < order; j++) {
                    v_new[i] += A[i * order + j] * v[j];
                }
            }
            // Normalize v_new and estimate eigenvalue via Rayleigh quotient
            double norm = 0.0;
            for (int i = 0; i < order; i++) norm += v_new[i] * v_new[i];
            norm = sqrt(norm);
            lambda = 0.0;
            for (int i = 0; i < order; i++) {
                v_new[i] /= norm; // Normalize eigenvector
                lambda += v_new[i] * A[i * order + k]; // Approximate eigenvalue
            }
            memcpy(v, v_new, order * sizeof(double)); // Update eigenvector
        }
        roots[k] = lambda; // Store approximated eigenvalue
    }
    free(A);
    return 1;
}


/**
 * @brief Checks if polynomial roots ensure AR stationarity or MA invertibility.
 *
 * @details 
 * **Purpose**: Validates AR or MA coefficients in ARIMA by checking if their polynomial roots lie 
 * outside the unit circle (magnitude > 1), ensuring stability (AR) or invertibility (MA).
 *
 * **Method Used**: 
 * - Calls `computeCompanionEigenvalues` to get roots.
 * - Checks each root’s magnitude against \( ROOT_TOLERANCE \) (1.001), flagging violations.
 *
 * **Why This Method**: 
 * - **Root Condition**: Directly tests the theoretical requirement for AR stationarity and MA 
 *   invertibility, critical for valid ARIMA forecasts.
 * - **Simplicity**: Uses existing eigenvalue computation, avoiding redundant polynomial solvers.
 *
 * **Downsides and Limitations**:
 * - **Root Accuracy**: Depends on `computeCompanionEigenvalues`’s power iteration, which may miss 
 *   roots or lack precision for complex cases.
 * - **Tolerance**: Fixed \( ROOT_TOLERANCE = 1.001 \) may be too strict or lenient depending on context.
 * - **Diagnostics**: Warns but doesn’t adjust coefficients here, leaving correction to the caller.
 *
 * @param coeffs Polynomial coefficients (AR: \( \phi \), MA: \( \theta \)).
 * @param order Polynomial order (p for AR, q for MA).
 * @param isAR 1 for AR (stationarity), 0 for MA (invertibility).
 * @return 1 if all roots satisfy the condition, 0 if any violate it (with warning).
 */
int checkRoots(double coeffs[], int order, int isAR) {
    // Allocate array for roots
    double complex *roots = malloc(order * sizeof(double complex));
    if (!roots || !computeCompanionEigenvalues(coeffs, order, roots)) {
        fprintf(stderr, "Error: Root computation failed.\n");
        free(roots);
        return 0;
    }

    // Step 1: Check each root’s magnitude
    int valid = 1;
    for (int i = 0; i < order; i++) {
        double mag = cabs(roots[i]); // Compute complex magnitude
        if (mag <= ROOT_TOLERANCE) {
            // Root inside or on unit circle violates condition
            fprintf(stderr, "Warning: %s root magnitude %.4f <= %.4f\n", 
                    isAR ? "AR" : "MA", mag, ROOT_TOLERANCE);
            valid = 0;
        }
    }
    free(roots);
    return valid;
}


/**
 * @brief Provides initial MA parameter estimates using the ACF of residuals.
 *
 * @details 
 * **Purpose**: Generates starting values for MA coefficients in ARIMA’s MLE estimation, improving 
 * convergence by leveraging the residual autocorrelation structure.
 *
 * **Method Used**: 
 * - Computes the ACF of residuals up to lag \( q \).
 * - Sets \( \theta_i = r_{i+1} * 0.5 \), scaling down ACF values to avoid overestimation.
 *
 *
 * @param residuals Residuals from AR fit (or original series if no AR).
 * @param length Length of residuals.
 * @param q MA order.
 * @param theta Output array for initial MA coefficients (length q).
 */
void initialMAFromACF(double residuals[], int length, int q, double theta[]) {
    // Step 1: Compute ACF up to lag q
    double acf[q + 1];
    computeACF(residuals, length, q, acf);

    // Step 2: Set initial theta values
    // Use ACF lags 1 to q, scaled by 0.5 for stability
    for (int i = 0; i < q; i++) {
        theta[i] = acf[i + 1] * 0.5;
    }
}

/**
 * @brief Computes the negative log-likelihood for an MA(q) model.
 *
 * @details 
 * **Purpose**: Evaluates the goodness-of-fit of MA parameters during MLE optimization in ARIMA, 
 * providing the objective function to minimize.
 *
 * **Method Used**: 
 * - Models residuals as an MA(q) process: \( y_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + ... + \theta_q \epsilon_{t-q} \).
 * - Recursively computes errors \( \epsilon_t = y_t - (\mu + \sum_{j=1}^{q} \theta_j \epsilon_{t-j}) \).
 * - Assumes \( \epsilon_t \sim N(0, \sigma^2) \), computing negative log-likelihood: 
 *   \( -\log L = \frac{n}{2} (\log(2\pi \sigma^2) + 1) \), where \( \sigma^2 = \frac{1}{n} \sum \epsilon_t^2 \).
 *
 * **Why This Method**: 
 * - **Likelihood Maximization**: Standard for MLE, directly optimizing fit to observed residuals.
 * - **Gaussian Assumption**: Simplifies computation, aligning with common time series assumptions.
 *
 * **Downsides and Limitations**:
 * - **Normality**: Assumes Gaussian errors, which may not hold for real data with heavy tails or outliers.
 * - **Initial Errors**: Assumes \( \epsilon_{t} = 0 \) for \( t < q \), introducing bias in early terms.
 * - **Numerical Stability**: Large residuals or poor \( \theta \) guesses can inflate \( \sigma^2 \), skewing likelihood.
 *
 * @param theta MA parameters [theta_1, ..., theta_q, intercept].
 * @param residuals Input residuals to fit.
 * @param n Length of residuals.
 * @param q MA order.
 * @return Negative log-likelihood value.
 */
double computeMANegLogLikelihood(double theta[], double residuals[], int n, int q) {
    // Allocate array for computed errors
    double *errors = calloc(n, sizeof(double));
    if (!errors) return INFINITY;

    // Step 1: Compute errors recursively
    double sum_sq = 0.0;
    for (int t = q; t < n; t++) {
        double pred = theta[q]; // Start with intercept (mu)
        // Add MA terms: sum theta[j] * epsilon_{t-j-1}
        for (int j = 0; j < q && t - j - 1 >= 0; j++) {
            pred += theta[j] * errors[t - j - 1];
        }
        errors[t] = residuals[t] - pred; // \( \epsilon_t = y_t - prediction \)
        sum_sq += errors[t] * errors[t]; // Accumulate squared errors
    }

    // Step 2: Estimate variance
    // \( \sigma^2 = \frac{1}{n} \sum \epsilon_t^2 \)
    double sigma2 = sum_sq / n;

    // Step 3: Compute negative log-likelihood
    // Assuming Gaussian errors: \( -\log L = \frac{n}{2} (\log(2\pi \sigma^2) + 1) \)
    double nll = n * 0.5 * (log(2 * M_PI * sigma2) + 1);
    free(errors);
    return nll;
}

/**
 * @brief Estimates MA parameters using Newton-Raphson maximum likelihood estimation.
 *
 * @details 
 * **Purpose**: Fits MA(q) parameters in ARIMA by maximizing the likelihood of residuals, refining 
 * initial guesses to accurate coefficients for forecasting.
 *
 * **Method Used**: 
 * - Starts with initial \( \theta \) from `initialMAFromACF`.
 * - Iteratively updates \( \theta \) using Newton-Raphson: \( \theta_{new} = \theta_{old} - H^{-1} g \),
 *   where \( g \) is the gradient and \( H \) is the Hessian of the negative log-likelihood.
 * - Gradient: \( g_i = -\sum \epsilon_t \epsilon_{t-i} \) (for \( i = 1, ..., q \)), \( g_{q+1} = -\sum \epsilon_t \) (intercept).
 * - Hessian: \( H_{ij} = \sum \epsilon_{t-i} \epsilon_{t-j} \) (approximation).
 * - Uses line search to ensure likelihood decreases.
 *
 * **Why This Method**: 
 * - **MLE**: Provides statistically optimal estimates under Gaussian assumptions, critical for MA accuracy.
 * - **Newton-Raphson**: Fast convergence (quadratic) near the optimum, leveraging gradient and curvature.
 *
 * **Downsides and Limitations**:
 * - **Convergence**: May fail if initial guesses are poor or Hessian is ill-conditioned, requiring many iterations.
 * - **Hessian Approximation**: Simplified Hessian may not capture full curvature, slowing convergence or leading to local minima.
 * - **Computational Cost**: \( O(q^2 n) \) per iteration, expensive for large \( q \) or \( n \).
 *
 * @param residuals Residuals from AR fit (or original series if no AR).
 * @param length Length of residuals.
 * @param q MA order.
 * @return Pointer to array [theta_1, ..., theta_q, intercept]; caller must free it.
 */
double *estimateMAWithMLE(double residuals[], int length, int q) {
    // Allocate theta array (q coefficients + intercept)
    double *theta = malloc((q + 1) * sizeof(double));
    if (!theta) return NULL;

    // Step 1: Set initial guesses
    initialMAFromACF(residuals, length, q, theta);
    theta[q] = calculateMean(residuals, length); // Intercept as mean

    // Step 2: Newton-Raphson optimization
    for (int iter = 0; iter < MAX_NEWTON_ITER; iter++) {
        double grad[q + 1];
        double hess[q + 1][q + 1];
        memset(grad, 0, (q + 1) * sizeof(double));
        memset(hess, 0, (q + 1) * (q + 1) * sizeof(double));
        double *errors = calloc(length, sizeof(double));
        if (!errors) { free(theta); return NULL; }

        // Compute errors and gradient/Hessian
        for (int t = q; t < length; t++) {
            double pred = theta[q];
            for (int j = 0; j < q && t - j - 1 >= 0; j++) pred += theta[j] * errors[t - j - 1];
            errors[t] = residuals[t] - pred;
            // Gradient: partial derivatives of -log L w.r.t. theta
            for (int i = 0; i < q; i++) {
                if (t - i - 1 >= 0) {
                    grad[i] += -errors[t] * errors[t - i - 1]; // \( \frac{\partial}{\partial \theta_i} = -\sum \epsilon_t \epsilon_{t-i} \)
                    for (int j = 0; j <= i; j++) {
                        if (t - j - 1 >= 0) {
                            hess[i][j] += errors[t - i - 1] * errors[t - j - 1]; // Approximate Hessian
                            hess[j][i] = hess[i][j];
                        }
                    }
                }
            }
            grad[q] += -errors[t]; // Intercept gradient
            hess[q][q] += 1.0; // Hessian for intercept (simplified)
        }

        // Step 3: Solve for update direction: H * delta = -grad
        double grad_mat[q + 1][1];
        for (int i = 0; i < q + 1; i++) grad_mat[i][0] = grad[i];
        double *delta = performMultivariateLinearRegression(q + 1, q + 1, hess, grad_mat);
        if (!delta) { free(errors); free(theta); return NULL; }

        // Step 4: Line search to ensure likelihood improves
        double step_size = 1.0;
        double old_nll = computeMANegLogLikelihood(theta, residuals, length, q);
        double new_theta[q + 1];
        for (int i = 0; i < q + 1; i++) new_theta[i] = theta[i] - step_size * delta[i];
        double new_nll = computeMANegLogLikelihood(new_theta, residuals, length, q);
        while (new_nll > old_nll && step_size > NEWTON_TOL) {
            step_size *= 0.5;
            for (int i = 0; i < q + 1; i++) new_theta[i] = theta[i] - step_size * delta[i];
            new_nll = computeMANegLogLikelihood(new_theta, residuals, length, q);
        }

        // Step 5: Update theta
        memcpy(theta, new_theta, (q + 1) * sizeof(double));
        free(delta);
        free(errors);
        if (step_size < NEWTON_TOL) break; // Converged
    }

    // Step 6: Check invertibility and adjust
    if (!checkRoots(theta, q, 0)) {
        fprintf(stderr, "Adjusting MA coefficients for invertibility.\n");
        for (int i = 0; i < q; i++) theta[i] *= 0.95;
    }
    return theta;
}


/**
 * @brief Forecasts future values using an ARIMA(p,d,q) model.
 *
 * @details 
 * **Purpose**: Generates multi-step forecasts for a time series using an ARIMA model, integrating AR, 
 * differencing, and MA components to predict future values.
 *
 * **Method Used**: 
 * - Ensures stationarity via differencing (\( d \) times), guided by ADF test.
 * - Estimates AR coefficients (\( \phi \)) using Yule-Walker, with OLS fallback based on in-sample MAE.
 * - Estimates MA coefficients (\( \theta \)) via MLE with Newton-Raphson.
 * - Forecasts recursively: 
 *   - \( y_{t+h} = \mu + \phi_1 y_{t+h-1} + ... + \phi_p y_{t+h-p} + \theta_1 \epsilon_{t+h-1} + ... + \theta_q \epsilon_{t+h-q} \)
 *   - Uses past errors for \( h \leq q \), then assumes \( \epsilon = 0 \) for \( h > q \).
 * - Integrates differenced forecasts back to the original scale.
 *
 * - **ARIMA Framework**: Combines AR, I, and MA for flexible modeling of stationary and non-stationary series.
 * - **Recursive Forecasting**: Standard for ARIMA, balancing simplicity with predictive power.
 * - **Hybrid Estimation**: Combines Yule-Walker’s stability with OLS’s fit, optimizing AR accuracy.
 *
 * **Downsides and Limitations**:
 * - **Error Assumption**: Assumes future \( \epsilon = 0 \) beyond \( q \), underestimating uncertainty.
 * - **Complexity**: High \( p \) or \( q \) increases computation and risk of overfitting.
 * - **Stationarity Dependence**: Relies on accurate differencing; misjudged \( d \) skews results.
 * - **Short-Term Focus**: Long-horizon forecasts may diverge due to lack of stochastic error modeling.
 *
 * @param series The input time series.
 * @param seriesLength Length of the series.
 * @param p AR order.
 * @param d Differencing order.
 * @param q MA order.
 * @return Pointer to forecast array (length FORECAST_ARRAY_SIZE); caller must free it.
 */
double *forecastARIMA(double series[], int seriesLength, int p, int d, int q) {
    if (seriesLength < p + d + q + 1) {
        fprintf(stderr, "Error: Insufficient data for ARIMA(%d,%d,%d). Need %d points, got %d.\n",
                p, d, q, p + d + q + 1, seriesLength);
        exit(EXIT_FAILURE);
    }

    int currentLength = seriesLength;
    double *currentSeries = ensureStationary(series, &currentLength, d);
    if (p > currentLength / 2) p = currentLength / 2;

    double *arEstimates = NULL;
    if (p > 0 && currentLength > p) {
        arEstimates = malloc((p + 1) * sizeof(double));
        if (!arEstimates) { free(currentSeries); exit(EXIT_FAILURE); }
        yuleWalker(currentSeries, currentLength, p, arEstimates);
        arEstimates[p] = calculateMean(currentSeries, currentLength);
        if (!checkRoots(arEstimates, p, 1)) {
            fprintf(stderr, "Warning: AR coefficients adjusted.\n");
            for (int i = 0; i < p; i++) arEstimates[i] *= 0.95;
        }
        double X[currentLength - p][p], Y[currentLength - p][1];
        for (int t = p; t < currentLength; t++) {
            Y[t - p][0] = currentSeries[t];
            for (int j = 0; j < p; j++) X[t - p][j] = currentSeries[t - j - 1];
        }
        double *olsEst = performMultivariateLinearRegression(currentLength - p, p, X, Y);
        double mae_ols = 0.0, mae_yw = 0.0;
        for (int t = p; t < currentLength; t++) {
            double pred_ols = olsEst[p], pred_yw = arEstimates[p];
            for (int j = 0; j < p; j++) {
                pred_ols += olsEst[j] * currentSeries[t - j - 1];
                pred_yw += arEstimates[j] * currentSeries[t - j - 1];
            }
            mae_ols += fabs(currentSeries[t] - pred_ols);
            mae_yw += fabs(currentSeries[t] - pred_yw);
        }
        mae_ols /= (currentLength - p);
        mae_yw /= (currentLength - p);
        printf("In-sample MAE: OLS = %.4f, Yule-Walker = %.4f\n", mae_ols, mae_yw);
        if (mae_ols < mae_yw) {
            printf("Using OLS estimates for AR.\n");
            for (int i = 0; i <= p; i++) arEstimates[i] = olsEst[i];
        }
        free(olsEst);
    }

    double *arResiduals = NULL;
    if (q > 0 && p > 0 && currentLength > p) {
        arResiduals = malloc(sizeof(double) * (currentLength - p));
        if (!arResiduals) { free(currentSeries); if (arEstimates) free(arEstimates); exit(EXIT_FAILURE); }
        for (int t = p; t < currentLength; t++) {
            double pred = arEstimates[p];
            for (int j = 0; j < p; j++) pred += arEstimates[j] * currentSeries[t - j - 1];
            arResiduals[t - p] = currentSeries[t] - pred;
        }
    } else if (q > 0) {
        arResiduals = malloc(sizeof(double) * currentLength);
        if (!arResiduals) { free(currentSeries); if (arEstimates) free(arEstimates); exit(EXIT_FAILURE); }
        copyArray(currentSeries, arResiduals, currentLength);
    }

    double *maEstimates = NULL;
    if (q > 0 && arResiduals && currentLength - p > q) {
        maEstimates = estimateMAWithMLE(arResiduals, currentLength - p, q);
    }

    double *forecast = malloc(sizeof(double) * FORECAST_ARRAY_SIZE);
    if (!forecast) { free(currentSeries); if (arEstimates) free(arEstimates); if (maEstimates) free(maEstimates); if (arResiduals) free(arResiduals); exit(EXIT_FAILURE); }
    double pastErrors[q];
    memset(pastErrors, 0, q * sizeof(double));
    if (q > 0 && arResiduals) {
        for (int i = 0; i < q && i < currentLength - p; i++) pastErrors[i] = arResiduals[currentLength - p - 1 - i];
    }

    double oneStep = arEstimates ? arEstimates[p] : 0.0;
    if (p > 0) for (int j = 0; j < p; j++) oneStep += arEstimates[j] * currentSeries[currentLength - j - 1];
    else oneStep = currentSeries[currentLength - 1];
    if (q > 0 && maEstimates) for (int j = 0; j < q; j++) oneStep += maEstimates[j] * pastErrors[j];
    forecast[0] = oneStep;

    for (int h = 1; h < FORECAST_HORIZON; h++) {
        double f = arEstimates ? arEstimates[p] : 0.0;
        for (int j = 0; j < p; j++) {
            double value = (h - j - 1 >= 0) ? forecast[h - j - 1] : currentSeries[currentLength + h - j - 1];
            f += arEstimates[j] * value;
        }
        if (q > 0 && maEstimates) {
            for (int j = 0; j < q; j++) {
                double error = (h - j - 1 >= 0) ? 0.0 : pastErrors[j + h];
                f += maEstimates[j] * error;
            }
        }
        forecast[h] = f;
    }
    forecast[FORECAST_HORIZON] = 0.0;

    if (d > 0) {
        double recoveryValue = series[seriesLength - 1];
        double *integrated = integrateSeries(forecast, recoveryValue, FORECAST_HORIZON);
        memcpy(forecast, integrated, FORECAST_HORIZON * sizeof(double));
        free(integrated);
    }

    free(currentSeries);
    if (arEstimates) free(arEstimates);
    if (maEstimates) free(maEstimates);
    if (arResiduals) free(arResiduals);
    return forecast;
}

/*==================== Main Function ====================*/
int main(void) {
    double sampleData[] = {10.544653, 10.688583, 10.666841, 10.662732, 10.535033, 10.612065, 10.577628, 10.524487, 10.511290, 10.520899, 10.605484, 10.506456, 10.693456, 10.667562, 10.640863, 10.553473, 10.684760, 10.752397, 10.671068, 10.667091, 10.641893, 10.625706, 10.701795, 10.607544, 10.689169, 10.695256, 10.717050, 10.677475, 10.691141, 10.730298, 10.732664, 10.710082, 10.713123, 10.759815, 10.696599, 10.663845, 10.716597, 10.780855, 10.795759, 10.802620, 10.720496, 10.753401, 10.709436, 10.746909, 10.737377, 10.754609, 10.765248, 10.692602, 10.837926, 10.755324, 10.756213, 10.843190, 10.862529, 10.751269, 10.902390, 10.817731, 10.859796, 10.887362, 10.835401, 10.824412, 10.860767, 10.819504, 10.907496, 10.831528, 10.821727, 10.830010, 10.915317, 10.858694, 10.921139, 10.927524, 10.894352, 10.889785, 10.956356, 10.938758, 11.093567, 10.844841, 11.094493, 11.035941, 10.982765, 11.071057, 10.996308, 11.099276, 11.142057, 11.137176, 11.157537, 11.007247, 11.144075, 11.183029, 11.172096, 11.164571, 11.192833, 11.227109, 11.141589, 11.311490, 11.239783, 11.295933, 11.199566, 11.232262, 11.333208, 11.337874, 11.322334, 11.288216, 11.280459, 11.247973, 11.288277, 11.415095, 11.297583, 11.360763, 11.288338, 11.434631, 11.456051, 11.578981, 11.419166, 11.478404, 11.660141, 11.544303, 11.652028, 11.638368, 11.651792, 11.621518, 11.763853, 11.760687, 11.771138, 11.678104, 11.783163, 11.932094, 11.948678, 11.962627, 11.937934, 12.077570, 11.981595, 12.096366, 12.032683, 12.094221, 11.979764, 12.217793, 12.235930, 12.129859, 12.411867, 12.396301, 12.413920, 12.445867, 12.480462, 12.470674, 12.537774, 12.562252, 12.810248, 12.733546, 12.861890, 12.918012, 13.033087, 13.245610, 13.184196, 13.414342, 13.611838, 13.626345, 13.715446, 13.851129, 14.113374, 14.588537, 14.653982, 15.250756, 15.618371, 16.459558, 18.144264, 23.523062, 40.229511, 38.351265, 38.085281, 37.500885, 37.153946, 36.893066, 36.705956, 36.559536, 35.938847, 36.391586, 36.194046, 36.391586, 36.119102, 35.560543, 35.599018, 34.958851, 35.393860, 34.904797, 35.401318, 34.863518, 34.046680, 34.508522, 34.043182, 34.704235, 33.556644, 33.888481, 33.533638, 33.452129, 32.930935, 32.669731, 32.772537, 32.805634, 32.246761, 32.075809, 31.864927, 31.878294, 32.241131, 31.965626, 31.553604, 30.843288, 30.784569, 31.436094, 31.170496, 30.552132, 30.500242, 30.167421, 29.911989, 29.586046, 29.478958, 29.718994, 29.611095, 29.557945, 28.463432, 29.341291, 28.821512, 28.447210, 27.861872, 27.855633, 27.910660, 28.425800, 27.715517, 27.617193, 27.093372, 26.968832, 26.977205, 27.170172, 26.251677, 26.633236, 26.224941, 25.874708, 25.593761, 26.392395, 24.904768, 25.331600, 24.530737, 25.074808, 25.310865, 24.337013, 24.442986, 24.500193, 24.130409, 24.062714, 24.064592, 23.533037, 23.977909, 22.924667, 22.806379, 23.130791, 22.527645, 22.570505, 22.932512, 22.486126, 22.594856, 22.383926, 22.115181, 22.105082, 21.151754, 21.074114, 21.240192, 20.977468, 20.771507, 21.184586, 20.495111, 20.650751, 20.656075, 20.433039, 20.005697, 20.216360, 19.982117, 19.703951, 19.572884, 19.332155, 19.544645, 18.666328, 19.219872, 18.934229, 19.186989, 18.694986, 18.096903, 18.298306, 17.704309, 18.023785, 18.224157, 18.182484, 17.642824, 17.739542, 17.474176, 17.270575, 17.604120, 17.631210, 16.639175, 17.107626, 17.024216, 16.852285, 16.780111, 16.838861, 16.539309, 16.092861, 16.131529, 16.221350, 16.087164, 15.821659, 15.695448, 15.693087, 16.047991, 15.682863, 15.724131, 15.263708, 15.638486, 15.443835, 15.602257, 15.122874, 14.918172, 14.968882, 14.843689, 14.861169, 15.052527, 15.056897, 14.690192, 14.686479, 14.567565, 14.365212, 14.253309, 14.289158, 14.227124, 14.069589, 14.074703, 13.869432, 13.861959, 13.782178, 13.882711, 13.908362, 13.727641, 13.600214, 13.594969, 13.535290, 13.602018, 13.502626, 13.579159, 13.207825, 13.426789, 13.178141, 13.286413, 12.958746, 13.189507, 13.079733, 13.138372, 12.986096, 12.854589, 12.858962, 12.903029, 12.852099, 12.644394, 12.558786, 12.636994};
    int dataLength = sizeof(sampleData) / sizeof(sampleData[0]);

    double *forecast = forecastARIMA(sampleData, 185, 2, 1, 4); // Adjusted to ARIMA(2,1,1)
    printf("ARIMA(2,1,1) Forecast:\n");
    for (int i = 0; i < FORECAST_HORIZON; i++) printf("%.4f ", forecast[i]);
    printf("\n");
    free(forecast);

    return 0;
}
