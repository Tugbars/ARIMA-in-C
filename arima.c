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
    
    // Allocate arrays for initial and updated column norms
    double *norms = malloc(n * sizeof(double));
    double *norms_updated = malloc(n * sizeof(double));
    if (!norms || !norms_updated) {
        fprintf(stderr, "Memory allocation error in qr_decomp_colpivot_blocked.\n");
        exit(EXIT_FAILURE);
    }
    DEBUG_PRINT("Allocated norms arrays: m=%d, n=%d, block_size=%d\n", m, n, block_size);

    // Compute initial column norms
    // Why: Establishes baseline for pivoting decisions
    for (j = 0; j < n; j++) {
        jpvt[j] = j; // Initialize pivot array as identity
        double sum = 0.0;
        for (i = 0; i < m; i++) {
            double aij = A[IDX(i, j, lda)];
            sum += aij * aij; // Sum of squares for column j
        }
        norms[j] = sqrt(sum); // 2-norm of column j
        norms_updated[j] = norms[j];
        if (norms[j] < SINGULARITY_THRESHOLD) {
            fprintf(stderr, "Warning: Column %d is nearly singular (norm = %e).\n", j, norms[j]);
            norms[j] = SINGULARITY_THRESHOLD; // Prevent division issues
            norms_updated[j] = SINGULARITY_THRESHOLD;
        }
        DEBUG_PRINT("Initial norm for col %d: %.4e\n", j, norms[j]);
    }
    
    // Main factorization loop over columns
    // Why: Iteratively builds Q and R while pivoting for stability
    for (k = 0; k < n && k < m; k++) {
        DEBUG_PRINT("Processing column k=%d\n", k);

        // Step 1: Column pivoting
        // Why: Moves largest norm column to pivot position for stability
        int max_index = k;
        for (j = k; j < n; j++) {
            if (norms_updated[j] > norms_updated[max_index]) {
                max_index = j; // Find column with max updated norm
            }
        }
        if (max_index != k) {
            DEBUG_PRINT("Pivoting: swapping col %d with col %d\n", k, max_index);
            // Swap columns k and max_index in A
            for (i = 0; i < m; i++) {
                double tmp = A[IDX(i, k, lda)];
                A[IDX(i, k, lda)] = A[IDX(i, max_index, lda)];
                A[IDX(i, max_index, lda)] = tmp;
            }
            // Update pivot array
            int tmp_int = jpvt[k];
            jpvt[k] = jpvt[max_index];
            jpvt[max_index] = tmp_int;
            // Swap norms
            double tmp_norm = norms_updated[k];
            norms_updated[k] = norms_updated[max_index];
            norms_updated[max_index] = tmp_norm;
            tmp_norm = norms[k];
            norms[k] = norms[max_index];
            norms[max_index] = tmp_norm;
        }
        DEBUG_PRINT("After pivot: jpvt[%d]=%d, norm=%.4e\n", k, jpvt[k], norms[k]);
        
        // Step 2: Compute the 2-norm of the k-th column from row k onward
        // Why: Determines magnitude for Householder reflector
        double norm_x = 0.0;
        for (i = k; i < m; i++) {
            norm_x += A[IDX(i, k, lda)] * A[IDX(i, k, lda)];
        }
        norm_x = sqrt(norm_x);
        DEBUG_PRINT("Column %d norm_x from row %d: %.4e\n", k, k, norm_x);
        
        // Step 3: Handle near-singular pivot
        // Why: Prevents breakdown if column is effectively zero
        if (norm_x < SINGULARITY_THRESHOLD) {
            fprintf(stderr, "Warning: Nearly singular pivot encountered at column %d (norm = %e). Setting pivot to zero.\n", k, norm_x);
            A[IDX(k, k, lda)] = 0.0;
            continue; // Skip reflector for this column
        }
        
        // Step 4: Compute Householder reflector
        // Why: Zeros out subdiagonal elements in column k
        double sign = (A[IDX(k, k, lda)] >= 0) ? -1.0 : 1.0; // Choose sign to avoid cancellation
        double *v = malloc((m - k) * sizeof(double));
        if (!v) {
            fprintf(stderr, "Memory allocation error in qr_decomp_colpivot_blocked (v).\n");
            exit(EXIT_FAILURE);
        }
        v[0] = A[IDX(k, k, lda)] - sign * norm_x; // First element of reflector
        for (i = k + 1; i < m; i++) {
            v[i - k] = A[IDX(i, k, lda)]; // Copy remaining elements
        }
        double norm_v = 0.0;
        for (i = 0; i < m - k; i++) {
            norm_v += v[i] * v[i]; // Compute norm of v
        }
        norm_v = sqrt(norm_v);
        DEBUG_PRINT("Householder vector norm_v: %.4e\n", norm_v);
        if (norm_v < SINGULARITY_THRESHOLD) {
            fprintf(stderr, "Warning: Householder vector nearly zero at column %d.\n", k);
            free(v);
            continue; // Skip if reflector is trivial
        }
        
        // Step 5: Normalize and store the reflector
        // Why: Prepares v for orthogonal update and stores it in A
        for (i = 0; i < m - k; i++) {
            v[i] /= norm_v; // Normalize v
        }
        A[IDX(k, k, lda)] = sign * norm_x; // Store pivot value
        for (i = k + 1; i < m; i++) {
            A[IDX(i, k, lda)] = v[i - k]; // Store reflector below diagonal
        }
        DEBUG_PRINT("Stored reflector for col %d: A[%d,%d]=%.4e\n", k, k, k, A[IDX(k, k, lda)]);
        
        // Step 6: Blocked update of trailing columns
        // Why: Applies reflector to remaining columns efficiently
        nb = ((k + block_size) < n) ? block_size : (n - k); // Adjust block size if needed
        DEBUG_PRINT("Updating block: k=%d, nb=%d\n", k, nb);
        for (j = k + 1; j < k + nb; j++) {
            double dot = 0.0;
            for (i = k; i < m; i++) {
                double vi = (i == k) ? 1.0 : A[IDX(i, k, lda)]; // Implicit 1 for v[0]
                dot += vi * A[IDX(i, j, lda)]; // v^T * A_j
            }
            for (i = k; i < m; i++) {
                double vi = (i == k) ? 1.0 : A[IDX(i, k, lda)];
                A[IDX(i, j, lda)] -= 2 * vi * dot; // A_j = A_j - 2 * v * (v^T A_j)
            }
            // Reorthogonalization pass
            dot = 0.0;
            for (i = k; i < m; i++) {
                double vi = (i == k) ? 1.0 : A[IDX(i, k, lda)];
                dot += vi * A[IDX(i, j, lda)];
            }
            for (i = k; i < m; i++) {
                double vi = (i == k) ? 1.0 : A[IDX(i, k, lda)];
                A[IDX(i, j, lda)] -= 2 * vi * dot; // Second pass for stability
            }
            // Update norm
            double new_norm_sq = 0.0;
            for (i = k + 1; i < m; i++) {
                new_norm_sq += A[IDX(i, j, lda)] * A[IDX(i, j, lda)];
            }
            double new_norm = sqrt(new_norm_sq);
            if (new_norm < 0.1 * norms[j]) {
                if (new_norm < SINGULARITY_THRESHOLD) {
                    fprintf(stderr, "Warning: Updated norm for column %d is nearly singular (new_norm = %e).\n", j, new_norm);
                    new_norm = 0.0;
                } else {
                    new_norm_sq = 0.0;
                    for (i = k + 1; i < m; i++) {
                        new_norm_sq += A[IDX(i, j, lda)] * A[IDX(i, j, lda)];
                    }
                    new_norm = sqrt(new_norm_sq);
                }
            }
            norms_updated[j] = new_norm;
            norms[j] = new_norm;
            DEBUG_PRINT("Updated norm for col %d: %.4e\n", j, new_norm);
        }
        // Non-blocked update for remaining columns
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
            for (i = k + 1; i < m; i++) {
                new_norm_sq += A[IDX(i, j, lda)] * A[IDX(i, j, lda)];
            }
            double new_norm = sqrt(new_norm_sq);
            if (new_norm < 0.1 * norms[j]) {
                if (new_norm < SINGULARITY_THRESHOLD) {
                    fprintf(stderr, "Warning: Updated norm for column %d is nearly singular (new_norm = %e).\n", j, new_norm);
                    new_norm = 0.0;
                } else {
                    new_norm_sq = 0.0;
                    for (i = k + 1; i < m; i++) {
                        new_norm_sq += A[IDX(i, j, lda)] * A[IDX(i, j, lda)];
                    }
                    new_norm = sqrt(new_norm_sq);
                }
            }
            norms_updated[j] = new_norm;
            norms[j] = new_norm;
            DEBUG_PRINT("Updated norm for col %d (non-block): %.4e\n", j, new_norm);
        }
        free(v);
    }
    // Step 7: Clean up
    // Why: Releases temporary memory
    free(norms);
    free(norms_updated);
    DEBUG_PRINT("Completed QR decomposition\n");
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
    for (int k = 0; k < n; k++) {
        double dot = 0.0;
        for (int i = k; i < n; i++) {
            double vi = (i == k) ? 1.0 : A[IDX(i, k, lda)];
            dot += vi * b[i];
        }
        for (int i = k; i < n; i++) {
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
    for (int i = n - 1; i >= 0; i--) {
        double sum = b[i];
        for (int j = i + 1; j < n; j++) sum -= A[IDX(i, j, lda)] * x[j];
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
    // Leading dimension in column-major format
    int lda = n;

    // Allocate a buffer A to store the matrix in column-major order.
    double *A = malloc(n * n * sizeof(double));
    if (!A) {
        fprintf(stderr, "Memory allocation error in invertMatrixQR.\n");
        exit(EXIT_FAILURE);
    }

    // Copy A_in (row-major) into A (column-major).
    // IDX(i, j, lda) presumably expands to j*lda + i for 2D indexing.
    // This loop transposes the data from row-major to column-major.
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            A[IDX(i, j, lda)] = A_in[j * n + i];
        }
    }

    // Allocate memory for the pivot array (jpvt), which indicates column
    // permutations during the QR factorization (column pivoting).
    int *jpvt = malloc(n * sizeof(int));
    if (!jpvt) {
        fprintf(stderr, "Memory allocation error in invertMatrixQR (jpvt).\n");
        exit(EXIT_FAILURE);
    }

    // Perform a blocked QR decomposition with column pivoting on matrix A.
    // The pivot array jpvt will hold the final column ordering.
    // The function signature is presumed: qr_decomp_colpivot_blocked(rows, cols, A, lda, jpvt, block_size).
    qr_decomp_colpivot_blocked(n, n, A, lda, jpvt, 1);

    // Allocate an array to hold the inverse permutation. 
    // inv_p[j] tells us where the j-th column ended up after pivoting.
    int *inv_p = malloc(n * sizeof(int));
    if (!inv_p) {
        fprintf(stderr, "Memory allocation error in invertMatrixQR (inv_p).\n");
        exit(EXIT_FAILURE);
    }

    // Compute the inverse of the pivot array:
    // For each column k, we set inv_p[jpvt[k]] = k.
    // This allows us to reorder the solution vector after back-substitution.
    for (int k = 0; k < n; k++) {
        inv_p[jpvt[k]] = k;
    }

    // Allocate space for the inverse of A. Zero-initialized (calloc).
    double *A_inv = calloc(n * n, sizeof(double));
    if (!A_inv) {
        fprintf(stderr, "Memory allocation error in invertMatrixQR (A_inv).\n");
        exit(EXIT_FAILURE);
    }

    // Temporary vectors used for solving each column of the identity system.
    // b: will hold the i-th column of the identity
    // y: will hold the intermediate result of back-substitution in the pivoted system
    // x: final solution re-ordered back via inv_p
    double *b = malloc(n * sizeof(double));
    double *y = malloc(n * sizeof(double));
    double *x = malloc(n * sizeof(double));
    if (!b || !y || !x) {
        fprintf(stderr, "Memory allocation error in invertMatrixQR (temp vectors).\n");
        exit(EXIT_FAILURE);
    }

    // For each column i of the identity matrix:
    // 1) Set b to the i-th column of the identity (1 at row i, 0 otherwise).
    // 2) Apply Q^T to b (applyQTranspose), which transforms the system for
    //    the next step with R.
    // 3) Perform back-substitution in the upper triangular R to get y.
    // 4) Reorder the solution y using the inverse pivot array inv_p to get x.
    // 5) Store x in the corresponding column of A_inv.
    for (int i = 0; i < n; i++) {
        // Initialize b to the i-th column of identity
        for (int k = 0; k < n; k++) {
            b[k] = (k == i) ? 1.0 : 0.0;
        }

        // Multiply b by Q^T: effectively solving Q^T * b = R * y
        applyQTranspose(n, A, lda, b);

        // Solve R * y = Q^T * b by back-substitution
        backSubstitution(n, A, lda, b, y);

        // Reorder the solution according to the inverse pivot array
        for (int j = 0; j < n; j++) {
            x[j] = y[inv_p[j]];
        }

        // Place the re-ordered solution x into the i-th column of A_inv
        for (int j = 0; j < n; j++) {
            A_inv[IDX(j, i, lda)] = x[j];
        }
    }

    // Free all temporary buffers before returning
    free(A);
    free(jpvt);
    free(inv_p);
    free(b);
    free(y);
    free(x);

    // Return pointer to the inverse matrix (column-major format)
    return A_inv;
}

/*==================== Linear Regression ====================*/

/**
 * Computes the slope and intercept of a univariate linear regression model using the least squares method.
 * 
 * The algorithm centers the predictor and response data by subtracting their means, calculates their standard deviations,
 * computes the Pearson correlation via covariance, and derives the slope as correlation * (std(response) / std(predictor))
 * and the intercept as mean(response) - slope * mean(predictor). This centers the data to simplify computations and uses
 * the correlation to scale the relationship, ensuring a best-fit line in the least squares sense.
 * 
 * Used in time series analysis (e.g., ARIMA) to estimate relationships between lagged variables or in standalone regression tasks.
 * It exists to provide a simple, efficient way to fit a linear model with one predictor, serving as a building block for more
 * complex multivariate methods or for quick trend analysis.
 * 
 * @param predictor The independent variable array.
 * @param response The dependent variable array.
 * @param length Number of observations.
 * @return Pointer to a dynamically allocated array [slope, intercept]; caller must free it.
 */
double *performUnivariateLinearRegression(double predictor[], double response[], int length) {
    double predictorDiff[length], responseDiff[length];
    // Step 1: Compute means of predictor and response to center the data
    double meanPredictor = calculateMean(predictor, length); // Mean of x: μ_x = Σx_i / n
    double meanResponse = calculateMean(response, length);   // Mean of y: μ_y = Σy_i / n

    // Step 2: Center the data by subtracting means to remove intercept bias
    for (int i = 0; i < length; i++) {
        predictorDiff[i] = predictor[i] - meanPredictor; // x_i' = x_i - μ_x
        responseDiff[i] = response[i] - meanResponse;    // y_i' = y_i - μ_y
    }

    // Step 3: Calculate standard deviations for scaling
    double stdPredictor = calculateStandardDeviation(predictor, length); // σ_x = √(Σ(x_i - μ_x)² / (n-1))
    double stdResponse = calculateStandardDeviation(response, length);   // σ_y = √(Σ(y_i - μ_y)² / (n-1))

    // Step 4: Compute covariance via element-wise product of centered data
    double *prodDiff = calculateElementwiseProduct(predictorDiff, responseDiff, length); // Product array: x_i' * y_i'
    double covariance = calculateArraySum(prodDiff, length) / (length - 1); // Cov(x,y) = Σ(x_i' * y_i') / (n-1)
    free(prodDiff); // Free temporary product array

    // Step 5: Calculate Pearson correlation coefficient
    double correlation = covariance / (stdPredictor * stdResponse); // r = Cov(x,y) / (σ_x * σ_y)

    // Step 6: Compute slope using correlation and standard deviations
    double slope = correlation * (stdResponse / stdPredictor); // β_1 = r * (σ_y / σ_x)

    // Step 7: Compute intercept using means and slope
    double intercept = meanResponse - slope * meanPredictor; // β_0 = μ_y - β_1 * μ_x

    // Step 8: Allocate and populate result array
    double *estimates = malloc(sizeof(double) * 2);
    if (!estimates) exit(EXIT_FAILURE); // Exit on memory allocation failure
    estimates[0] = slope;     // Store slope
    estimates[1] = intercept; // Store intercept

    return estimates;
}

void predictUnivariate(double predictor[], double predictions[], double slope, double intercept, int length) {
    for (int i = 0; i < length; i++) predictions[i] = predictor[i] * slope + intercept;
}

/**
 * Computes coefficients for a multivariate linear regression model using the normal equations with QR decomposition for stability.
 * 
 * The algorithm normalizes the design matrix X and response Y by subtracting column means, computes the normal equations (X^T X)β = X^T Y,
 * applies a small regularization to X^T X, inverts it using QR decomposition (or simpler methods for n=1,2), solves for β, and adjusts the
 * intercept. This approach ensures numerical stability and handles multicollinearity via regularization.
 * 
 * Used in ARIMA for AR parameter estimation and other multivariate fits where multiple predictors affect a response. It exists to extend
 * univariate regression to multiple predictors, providing a robust least squares solution for complex relationships.
 * 
 * @param numObservations Number of observations (rows in X and Y).
 * @param numPredictors Number of predictor variables (columns in X).
 * @param X Design matrix of predictors.
 * @param Y Response vector as a single-column matrix.
 * @return Pointer to array [beta_1, ..., beta_p, intercept]; caller must free it.
 */
double *performMultivariateLinearRegression(int numObservations, int numPredictors, double X[][numPredictors], double Y[][1]) {
    double X_normalized[numObservations][numPredictors], Y_normalized[numObservations][1];
    double X_means[1][numPredictors], Y_mean[1][1];
    double Xt[numPredictors][numObservations], XtX[numPredictors][numPredictors];
    double XtX_inv[numPredictors][numPredictors];
    double XtX_inv_Xt[numPredictors][numObservations], beta[numPredictors][1];

    // Step 1: Allocate memory for coefficients (betas + intercept)
    double *estimates = malloc(sizeof(double) * (numPredictors + 1));
    if (!estimates) { fprintf(stderr, "Memory allocation error in performMultivariateLinearRegression.\n"); exit(EXIT_FAILURE); }

    // Step 2: Normalize X and Y by subtracting column means to center data
    // Why: Centering simplifies intercept calculation and reduces numerical instability
    normalize2DArray(numObservations, numPredictors, X, X_normalized); // X_normalized[i][j] = X[i][j] - μ_j
    normalize2DArray(numObservations, 1, Y, Y_normalized);            // Y_normalized[i][0] = Y[i][0] - μ_Y

    // Step 3: Compute transpose of normalized X (X^T)
    // Why: Needed for normal equations: (X^T X)β = X^T Y
    transposeMatrix(numObservations, numPredictors, X_normalized, Xt);

    // Step 4: Compute X^T X
    // Formula: XtX[i][j] = Σ(X_normalized[k][i] * X_normalized[k][j]) for k = 0 to numObservations-1
    // Why: Forms the covariance matrix of predictors, core of normal equations
    matrixMultiply(numPredictors, numObservations, numPredictors, Xt, X_normalized, XtX);

    // Step 5: Add regularization to X^T X diagonal
    // Why: Prevents singularity in case of multicollinearity; λ = 1e-6 is small to minimize bias
    double lambda = 1e-6;
    for (int i = 0; i < numPredictors; i++) XtX[i][i] += lambda; // XtX[i][i] = XtX[i][i] + λ

    // Step 6: Invert X^T X based on number of predictors
    int n = numPredictors;
    if (n == 1) {
        // For n=1, direct inversion: 1 / scalar
        // Why: Simplest case, avoids overhead of QR for scalar
        XtX_inv[0][0] = 1.0 / XtX[0][0];
    } else if (n == 2) {
        // For n=2, use analytical 2x2 inversion
        // Formula: For [[a, b], [c, d]], inverse = 1/(ad - bc) * [[d, -b], [-c, a]]
        // Why: Fast and exact for 2x2, avoids QR complexity
        double a = XtX[0][0], b = XtX[0][1], c = XtX[1][0], d = XtX[1][1];
        double det = a * d - b * c;
        if (fabs(det) < 1e-12) det += lambda; // Adjust determinant if near-zero
        XtX_inv[0][0] = d / det;
        XtX_inv[0][1] = -b / det;
        XtX_inv[1][0] = -c / det;
        XtX_inv[1][1] = a / det;
    } else {
        // For n>2, use QR decomposition for inversion
        // Why: QR is numerically stable for larger matrices, handles ill-conditioning better than direct methods
        double *tempMatrix = malloc(n * n * sizeof(double));
        if (!tempMatrix) { fprintf(stderr, "Memory allocation error in performMultivariateLinearRegression.\n"); free(estimates); exit(EXIT_FAILURE); }
        for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) tempMatrix[IDX(i, j, n)] = XtX[i][j];
        double *invTemp = invertMatrixQR(tempMatrix, n);
        if (!invTemp) { fprintf(stderr, "Error: QR inversion failed for n=%d.\n", n); free(tempMatrix); free(estimates); exit(EXIT_FAILURE); }
        for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) XtX_inv[i][j] = invTemp[IDX(i, j, n)];
        free(invTemp);
        free(tempMatrix);
    }

    // Step 7: Compute (X^T X)^(-1) * X^T
    // Why: Part of solving β = (X^T X)^(-1) * X^T * Y (normal equations)
    matrixMultiply(numPredictors, numPredictors, numObservations, XtX_inv, Xt, XtX_inv_Xt);

    // Step 8: Compute β = (X^T X)^(-1) * X^T * Y
    // Formula: β = XtX_inv_Xt * Y_normalized
    // Why: Solves for coefficients in least squares sense
    matrixMultiply(numPredictors, numObservations, 1, XtX_inv_Xt, Y_normalized, beta);
    for (int i = 0; i < numPredictors; i++) estimates[i] = beta[i][0]; // Store betas

    // Step 9: Compute intercept using original means
    // Formula: intercept = μ_Y - Σ(β_i * μ_Xi)
    // Why: Adjusts for centering by reintroducing means
    for (int j = 0; j < numPredictors; j++) {
        double col[numObservations];
        for (int i = 0; i < numObservations; i++) col[i] = X[i][j];
        X_means[0][j] = calculateMean(col, numObservations); // μ_Xj = ΣX[i][j] / n
    }
    double yCol[numObservations];
    for (int i = 0; i < numObservations; i++) yCol[i] = Y[i][0];
    Y_mean[0][0] = calculateMean(yCol, numObservations); // μ_Y = ΣY[i][0] / n
    double intercept = Y_mean[0][0];
    for (int i = 0; i < numPredictors; i++) intercept -= estimates[i] * X_means[0][i];
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
void yuleWalker(const double series[], int length, int p, double phi[]) {
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
 * Computes eigenvalues of an AR or MA polynomial’s companion matrix using QR iteration for robust root detection.
 * 
 * Constructs a companion matrix for the polynomial (e.g., \( 1 - \phi_1 z - \phi_2 z^2 - ... - \phi_p z^p \) for AR),
 * then applies QR iteration to iteratively refine the matrix into a form where diagonal elements approximate all eigenvalues.
 * This replaces power iteration to accurately capture all roots, including complex ones, ensuring reliable stationarity 
 * (AR) or invertibility (MA) checks in ARIMA modeling.
 * 
 * **Method Used**: 
 * - Builds a companion matrix with coefficients on the top row and identity subdiagonal.
 * - Performs QR decomposition with column pivoting repeatedly (10 iterations), updating \( A = R \cdot Q \) each time.
 * - Extracts eigenvalues from the diagonal after convergence, assuming real roots dominate for simplicity.
 * 
 * **Why This Method**: 
 * - **Accuracy**: QR iteration converges to all eigenvalues, unlike power iteration’s focus on the dominant root, critical for higher \( p \) or \( q \).
 * - **Self-Contained**: Avoids external libraries (e.g., LAPACK’s dgeev), maintaining portability while improving over naive methods.
 * 
 * **Downsides and Limitations**:
 * - **Simplification**: Assumes real eigenvalues dominate; complex roots require full QR factorization with Q reconstruction (not implemented here).
 * - **Iterations**: Fixed at 10, which may not suffice for large or ill-conditioned matrices, though sufficient for small ARIMA orders.
 * - **Cost**: \( O(n^3) \) per iteration, higher than power iteration but justified by robustness.
 * 
 * Used in ARIMA to validate AR/MA coefficients via `checkRoots`. Improves stability/invertibility detection for accurate forecasting.
 * 
 * @param coeffs Polynomial coefficients (e.g., \( [\phi_1, \phi_2, ..., \phi_p] \) for AR).
 * @param order Polynomial order (p for AR, q for MA).
 * @param roots Output array for complex roots (length = order).
 * @return 1 if successful, 0 if memory allocation fails.
 */
int computeCompanionEigenvalues(double coeffs[], int order, double complex* roots) {
    if (order <= 0) return 1; // No roots for order 0

    // Step 1: Allocate and build companion matrix
    // Why: Transforms polynomial root-finding into an eigenvalue problem
    double *A = calloc(order * order, sizeof(double)); // Zero-initialized
    if (!A) return 0;
    for (int j = 0; j < order; j++) A[j] = -coeffs[j]; // Top row: -φ_j for AR convention
    for (int i = 1; i < order; i++) A[i * order + (i - 1)] = 1.0; // Subdiagonal 1s

    // Step 2: Allocate temporary arrays for QR iteration
    double *tempA = malloc(order * order * sizeof(double));
    int *jpvt = malloc(order * sizeof(int));
    if (!tempA || !jpvt) { free(A); free(tempA); free(jpvt); return 0; }

    // Step 3: QR iteration for eigenvalue approximation
    // Formula: A^{(k+1)} = R^{(k)} \cdot Q^{(k)}, repeated until A is nearly upper triangular
    for (int iter = 0; iter < 10; iter++) {
        memcpy(tempA, A, order * order * sizeof(double)); // Copy A for QR decomposition
        qr_decomp_colpivot_blocked(order, order, tempA, order, jpvt, 1); // QR factorization

        // Step 4: Reconstruct A = R * Q (simplified approach)
        double *R = calloc(order * order, sizeof(double));
        for (int i = 0; i < order; i++) {
            for (int j = i; j < order; j++) R[i * order + j] = tempA[i * order + j]; // Upper triangle
        }
        double *Q = calloc(order * order, sizeof(double));
        for (int i = 0; i < order; i++) Q[i * order + i] = 1.0; // Initialize Q as identity
        double *b = malloc(order * sizeof(double));
        for (int k = 0; k < order; k++) { // Build Q column-wise using Householder reflectors
            for (int i = 0; i < order; i++) b[i] = (i == k) ? 1.0 : 0.0; // Unit vector e_k
            applyQTranspose(order, tempA, order, b); // Q^T e_k
            for (int i = 0; i < order; i++) Q[i * order + k] = b[i]; // Column k of Q
        }
        matrixMultiply(order, order, order, (double (*)[])R, (double (*)[])Q, (double (*)[])A); // A = R * Q
        free(R); free(Q); free(b);
    }
    free(tempA); free(jpvt);

    // Step 5: Extract eigenvalues from diagonal
    // Why: After convergence, diagonal approximates eigenvalues (real parts only here)
    for (int i = 0; i < order; i++) {
        roots[i] = A[i * order + i]; // Assume real roots; extend for complex if needed
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
 * Computes the negative log-likelihood for an MA(q) model using Huber loss for robust estimation.
 * 
 * Models residuals as \( y_t = \mu + \epsilon_t + \sum_{j=1}^{q} \theta_j \epsilon_{t-j} \), computes errors recursively,
 * and applies Huber loss to mitigate outlier effects. Uses \( \delta = 1.345 \) for robustness in ARIMA fitting.
 * 
 * **Method Used**: 
 * - Recursively computes \( \epsilon_t \) from residuals and past errors.
 * - Applies Huber loss: \( L(\epsilon) = \frac{1}{2} \epsilon^2 \) if \( |\epsilon| \leq \delta \), else \( \delta (|\epsilon| - \frac{\delta}{2}) \).
 * - Sums loss as a proxy for negative log-likelihood.
 * 
 * **Why This Method**: 
 * - **Robustness**: Reduces sensitivity to outliers in residuals, improving MA parameter estimates.
 * - **Compatibility**: Fits within existing MLE framework, leveraging Newton-Raphson optimization.
 * 
 * **Downsides and Limitations**:
 * - **Simplification**: Lacks variance normalization; could enhance with robust variance estimation.
 * - **Initial Errors**: Assumes \( \epsilon_{t<q} = 0 \), potentially biasing early terms.
 * 
 * Used in `estimateMAWithMLE` to robustly fit MA parameters, enhancing ARIMA’s performance on noisy data.
 * 
 * @param theta MA parameters [θ_1, ..., θ_q, μ].
 * @param residuals Input residuals to fit.
 * @param n Length of residuals.
 * @param q MA order.
 * @return Negative log-likelihood value using Huber loss.
 */
double computeMANegLogLikelihood(double theta[], double residuals[], int n, int q) {
    // Step 1: Allocate error array
    double *errors = calloc(n, sizeof(double));
    if (!errors) return INFINITY;

    // Step 2: Initialize loss accumulator and Huber threshold
    double sum_loss = 0.0;
    const double delta = 1.345; // Huber δ for robustness

    // Step 3: Compute errors and apply Huber loss
    // Why: Robustly evaluates fit, dampening outlier effects
    for (int t = q; t < n; t++) {
        double pred = theta[q]; // Start with intercept μ
        for (int j = 0; j < q && t - j - 1 >= 0; j++) pred += theta[j] * errors[t - j - 1]; // Add MA terms: Σθ_j ε_{t-j}
        errors[t] = residuals[t] - pred; // ε_t = y_t - pred
        double abs_err = fabs(errors[t]);
        if (abs_err <= delta) sum_loss += 0.5 * abs_err * abs_err; // Quadratic: \( \frac{1}{2} \epsilon^2 \)
        else sum_loss += delta * (abs_err - 0.5 * delta); // Linear: \( \delta (|\epsilon| - \frac{\delta}{2}) \)
    }

    // Step 4: Return total loss as negative log-likelihood
    double nll = sum_loss;
    free(errors);
    return nll;
}


/**
 * Computes the negative log-likelihood for an AR(p) model using Huber loss for robust estimation.
 * 
 * Models the series as \( y_t = \mu + \sum_{j=1}^{p} \phi_j y_{t-j} + \epsilon_t \), computes errors recursively,
 * and applies Huber loss (quadratic for small errors, linear for large) to reduce the impact of outliers compared
 * to squared error loss. Uses \( \delta = 1.345 \) for 95% efficiency under normality, optimizing robustness in ARIMA.
 * 
 * **Method Used**: 
 * - Recursively predicts \( y_t \) using AR parameters and past values.
 * - Applies Huber loss: \( L(\epsilon) = \frac{1}{2} \epsilon^2 \) if \( |\epsilon| \leq \delta \), else \( \delta (|\epsilon| - \frac{\delta}{2}) \).
 * - Sums loss across observations as a proxy for negative log-likelihood.
 * 
 * **Why This Method**: 
 * - **Robustness**: Mitigates outlier effects (e.g., jumps in sampleData), unlike Gaussian squared error assumption.
 * - **Simplicity**: Maintains MLE framework with a modified loss, easily integrated into Newton-Raphson optimization.
 * 
 * **Downsides and Limitations**:
 * - **Approximation**: Omits variance scaling in likelihood for simplicity; could refine with robust \( \sigma^2 \) estimate.
 * - **Gradient**: Requires adjustment in optimization (not updated here), assuming current gradients adapt reasonably.
 * 
 * Used in `estimateARWithCMLE` to fit AR parameters robustly, enhancing ARIMA’s reliability with noisy data.
 * 
 * @param phi AR parameters [φ_1, ..., φ_p, μ].
 * @param series Input series (assumed stationary).
 * @param n Length of series.
 * @param p AR order.
 * @return Negative log-likelihood value using Huber loss.
 */
double computeARNegLogLikelihood(double phi[], double series[], int n, int p) {
    // Step 1: Allocate error array
    double *errors = calloc(n, sizeof(double));
    if (!errors) return INFINITY;

    // Step 2: Initialize loss accumulator and Huber threshold
    double sum_loss = 0.0;
    const double delta = 1.345; // Huber δ for 95% efficiency under normality

    // Step 3: Compute errors and apply Huber loss
    // Why: Robustly measures fit, reducing outlier influence
    for (int t = p; t < n; t++) {
        double pred = phi[p]; // Start with intercept μ
        for (int j = 0; j < p; j++) pred += phi[j] * series[t - j - 1]; // Add AR terms: Σφ_j y_{t-j}
        errors[t] = series[t] - pred; // ε_t = y_t - pred
        double abs_err = fabs(errors[t]);
        if (abs_err <= delta) sum_loss += 0.5 * abs_err * abs_err; // Quadratic: \( \frac{1}{2} \epsilon^2 \)
        else sum_loss += delta * (abs_err - 0.5 * delta); // Linear: \( \delta (|\epsilon| - \frac{\delta}{2}) \)
    }

    // Step 4: Return total loss as negative log-likelihood
    // Note: Simplified; could scale by n or estimate robust σ²
    double nll = sum_loss;
    free(errors);
    return nll;
}


/**
 * Estimates AR coefficients using Conditional Maximum Likelihood Estimation (CMLE) with Newton-Raphson optimization.
 * 
 * The algorithm starts with Yule-Walker estimates, iteratively updates coefficients by minimizing the negative log-likelihood
 * using gradients and Hessians, and applies a line search to ensure convergence. It models the series as y_t = μ + Σφ_j y_{t-j} + ε_t,
 * optimizing φ and μ to maximize the likelihood of observed data given past values.
 * 
 * Used in ARIMA to estimate AR parameters more accurately than Yule-Walker alone. Exists to provide robust, likelihood-based AR estimates,
 * improving forecast precision over simpler autocorrelation-based methods.
 * 
 * @param series Input time series (assumed stationary).
 * @param length Length of the series.
 * @param p AR order.
 * @param phi Output array for AR coefficients [φ_1, ..., φ_p, μ].
 */
void estimateARWithCMLE(double series[], int length, int p, double phi[]) {
    // Step 1: Allocate temporary array for coefficients (p AR terms + intercept)
    double *theta = malloc((p + 1) * sizeof(double));
    if (!theta) { fprintf(stderr, "Memory allocation error in estimateARWithCMLE.\n"); exit(EXIT_FAILURE); }

    // Step 2: Initialize with Yule-Walker estimates
    // Why: Provides a fast, reasonable starting point for optimization
    yuleWalker(series, length, p, theta); // Initial φ_j guesses
    theta[p] = calculateMean(series, length); // Initial μ = mean(y)

    // Step 3: Newton-Raphson optimization loop
    for (int iter = 0; iter < MAX_NEWTON_ITER; iter++) {
        double grad[p + 1], hess[p + 1][p + 1];
        memset(grad, 0, sizeof(grad));
        memset(hess, 0, sizeof(hess));
        double *errors = calloc(length, sizeof(double));
        if (!errors) { free(theta); exit(EXIT_FAILURE); }

        // Step 4: Compute errors and build gradient/Hessian
        for (int t = p; t < length; t++) {
            double pred = theta[p]; // Prediction starts with intercept μ
            for (int j = 0; j < p; j++) pred += theta[j] * series[t - j - 1]; // Add AR terms: Σφ_j y_{t-j}
            errors[t] = series[t] - pred; // ε_t = y_t - (μ + Σφ_j y_{t-j})
            for (int i = 0; i < p; i++) {
                grad[i] += -errors[t] * series[t - i - 1]; // ∂(-logL)/∂φ_i = -Σε_t y_{t-i-1}
                for (int j = 0; j <= i; j++) hess[i][j] += series[t - i - 1] * series[t - j - 1]; // ∂²(-logL)/∂φ_i∂φ_j ≈ Σy_{t-i-1} y_{t-j-1}
            }
            grad[p] += -errors[t]; // ∂(-logL)/∂μ = -Σε_t
            hess[p][p] += 1.0;    // ∂²(-logL)/∂μ² ≈ 1 (simplified)
        }
        // Step 5: Fill symmetric Hessian
        // Why: Hessian is symmetric, so compute only lower triangle and mirror
        for (int i = 0; i < p; i++) for (int j = i + 1; j < p; j++) hess[j][i] = hess[i][j];

        // Step 6: Solve for update direction using normal equations
        // Formula: Δθ = H^(-1) * (-∇), where H is Hessian, ∇ is gradient
        double grad_mat[p + 1][1];
        for (int i = 0; i <= p; i++) grad_mat[i][0] = grad[i];
        double *delta = performMultivariateLinearRegression(p + 1, p + 1, hess, grad_mat);
        if (!delta) { free(errors); free(theta); exit(EXIT_FAILURE); }

        // Step 7: Line search to ensure likelihood decreases
        double step = 1.0;
        double old_nll = computeARNegLogLikelihood(theta, series, length, p);
        double new_theta[p + 1];
        for (int i = 0; i <= p; i++) new_theta[i] = theta[i] - step * delta[i]; // θ_new = θ - step * Δθ
        double new_nll = computeARNegLogLikelihood(new_theta, series, length, p);
        while (new_nll > old_nll && step > NEWTON_TOL) {
            step *= 0.5; // Reduce step size if likelihood increases
            for (int i = 0; i <= p; i++) new_theta[i] = theta[i] - step * delta[i];
            new_nll = computeARNegLogLikelihood(new_theta, series, length, p);
        }
        memcpy(theta, new_theta, (p + 1) * sizeof(double)); // Update coefficients
        free(delta);
        free(errors);
        if (step < NEWTON_TOL) break; // Converged if step size too small
    }

    // Step 8: Copy final estimates to output
    for (int i = 0; i < p; i++) phi[i] = theta[i]; // AR coefficients
    phi[p] = theta[p]; // Intercept
    free(theta);
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
    if (!theta) {
        fprintf(stderr, "Memory allocation failed for theta in estimateMAWithMLE.\n");
        return NULL;
    }

    // Step 1: Set initial guesses
    // Why: Provides a starting point for optimization using ACF-based estimates
    initialMAFromACF(residuals, length, q, theta); // θ_j = 0.5 * ACF(lag j+1)
    theta[q] = calculateMean(residuals, length);   // Initial intercept μ = mean(residuals)
    // Note: Scaling by 0.5 dampens ACF values to avoid unstable starting points

    // Step 2: Newton-Raphson optimization loop
    // Why: Iteratively refines θ to minimize negative log-likelihood
    for (int iter = 0; iter < MAX_NEWTON_ITER; iter++) {
        // Initialize gradient and Hessian arrays
        double grad[q + 1];
        double hess[q + 1][q + 1];
        memset(grad, 0, (q + 1) * sizeof(double));
        memset(hess, 0, (q + 1) * (q + 1) * sizeof(double));
        double *errors = calloc(length, sizeof(double));
        if (!errors) {
            fprintf(stderr, "Memory allocation failed for errors in estimateMAWithMLE.\n");
            free(theta);
            return NULL;
        }

        // Step 2.1: Compute errors and build gradient/Hessian
        // Why: Gradient and Hessian drive the Newton-Raphson update
        for (int t = q; t < length; t++) {
            double pred = theta[q]; // Prediction starts with intercept μ
            for (int j = 0; j < q && t - j - 1 >= 0; j++) {
                pred += theta[j] * errors[t - j - 1]; // Add MA terms: Σθ_j ε_{t-j}
            }
            errors[t] = residuals[t] - pred; // ε_t = residual_t - (μ + Σθ_j ε_{t-j})
            // Gradient computation for coefficients
            for (int i = 0; i < q; i++) {
                if (t - i - 1 >= 0) {
                    grad[i] += -errors[t] * errors[t - i - 1]; // ∂(-logL)/∂θ_i = -Σε_t ε_{t-i-1}
                    // Hessian approximation for coefficients
                    for (int j = 0; j <= i; j++) {
                        if (t - j - 1 >= 0) {
                            hess[i][j] += errors[t - i - 1] * errors[t - j - 1]; // H_ij ≈ Σε_{t-i-1} ε_{t-j-1}
                            hess[j][i] = hess[i][j]; // Symmetry: H_ji = H_ij
                        }
                    }
                }
            }
            grad[q] += -errors[t]; // ∂(-logL)/∂μ = -Σε_t (intercept gradient)
            hess[q][q] += 1.0;    // ∂²(-logL)/∂μ² ≈ 1 (simplified constant for intercept)
        }

        // Step 3: Solve for update direction
        // Why: Computes Δθ = H^(-1) * (-g) to adjust parameters
        double grad_mat[q + 1][1];
        for (int i = 0; i < q + 1; i++) {
            grad_mat[i][0] = grad[i]; // Convert gradient to matrix form for regression
        }
        double *delta = performMultivariateLinearRegression(q + 1, q + 1, hess, grad_mat); // Solve H * Δθ = g
        if (!delta) {
            fprintf(stderr, "Failed to compute update direction in estimateMAWithMLE.\n");
            free(errors);
            free(theta);
            return NULL;
        }

        // Step 4: Line search to ensure likelihood improves
        // Why: Prevents overshooting by adjusting step size
        double step_size = 1.0; // Initial step size
        double old_nll = computeMANegLogLikelihood(theta, residuals, length, q); // Current likelihood
        double new_theta[q + 1];
        for (int i = 0; i < q + 1; i++) {
            new_theta[i] = theta[i] - step_size * delta[i]; // Tentative update: θ_new = θ - step * Δθ
        }
        double new_nll = computeMANegLogLikelihood(new_theta, residuals, length, q); // New likelihood
        while (new_nll > old_nll && step_size > NEWTON_TOL) {
            step_size *= 0.5; // Halve step size if likelihood increases
            for (int i = 0; i < q + 1; i++) {
                new_theta[i] = theta[i] - step_size * delta[i]; // Retry with smaller step
            }
            new_nll = computeMANegLogLikelihood(new_theta, residuals, length, q);
        }

        // Step 5: Update theta with the successful step
        // Why: Applies the refined update to improve the estimate
        memcpy(theta, new_theta, (q + 1) * sizeof(double));
        free(delta);
        free(errors);
        if (step_size < NEWTON_TOL) {
            // Step 5.1: Early exit if convergence achieved
            // Why: Step size below tolerance indicates negligible improvement
            break;
        }
    }

    // Step 6: Check invertibility and adjust
    // Why: Ensures MA process is invertible (roots outside unit circle) for stability
    if (!checkRoots(theta, q, 0)) {
        fprintf(stderr, "Adjusting MA coefficients for invertibility.\n");
        for (int i = 0; i < q; i++) {
            theta[i] *= 0.95; // Scale coefficients to push roots outside unit circle
        }
    }

    // Step 7: Return the final estimates
    // Note: Caller must free the returned pointer
    return theta;
}

/**
 * Initializes MA coefficients using the Hannan-Rissanen method for ARIMA modeling.
 * 
 * The algorithm fits a high-order AR model to residuals using Yule-Walker, computes AR residuals, and regresses these against the
 * original residuals to estimate MA parameters. It approximates the MA process y_t = μ + ε_t + Σθ_j ε_{t-j} by using AR residuals
 * as proxies for past errors, providing a robust starting point for further optimization.
 * 
 * Used in ARIMA to bootstrap MA estimation before MLE refinement. Exists to improve convergence of MA parameter estimation by
 * leveraging a temporary AR fit, offering a more informed initial guess than simple ACF-based methods.
 * 
 * @param residuals Input residuals from AR fit or original series.
 * @param length Length of residuals.
 * @param q MA order.
 * @param theta Output array for initial MA coefficients [θ_1, ..., θ_q, μ].
 */
void hannanRissanenMAInit(double residuals[], int length, int q, double theta[]) {
    // Step 1: Set temporary AR order (p_temp = q + 2), capped at half the series length
    // Why: High-order AR approximates MA process; cap prevents overfitting
    int p_temp = q + 2;
    if (p_temp > length / 2) p_temp = length / 2;

    // Step 2: Allocate and fit temporary AR model
    double *ar_temp = malloc(p_temp * sizeof(double));
    if (!ar_temp) { fprintf(stderr, "Memory allocation error in hannanRissanenMAInit.\n"); exit(EXIT_FAILURE); }
    yuleWalker(residuals, length, p_temp, ar_temp); // Fit AR(p_temp) to residuals

    // Step 3: Compute AR residuals as error proxies
    double *ar_resid = malloc((length - p_temp) * sizeof(double));
    if (!ar_resid) { free(ar_temp); exit(EXIT_FAILURE); }
    for (int t = p_temp; t < length; t++) {
        double pred = 0.0;
        for (int j = 0; j < p_temp; j++) pred += ar_temp[j] * residuals[t - j - 1]; // y_t = Σφ_j y_{t-j}
        ar_resid[t - p_temp] = residuals[t] - pred; // ε_t ≈ y_t - pred
    }

    // Step 4: Build design matrix X and response Y for MA regression
    // Why: Regress residuals on lagged AR residuals to estimate θ_j
    double X[length - p_temp][q], Y[length - p_temp][1];
    for (int t = p_temp; t < length; t++) {
        Y[t - p_temp][0] = residuals[t]; // y_t as response
        for (int j = 0; j < q; j++) X[t - p_temp][j] = (t - j - 1 >= p_temp) ? ar_resid[t - p_temp - j - 1] : 0.0; // X[t][j] = ε_{t-j-1}
    }

    // Step 5: Estimate MA coefficients via multivariate regression
    // Formula: θ = (X^T X)^(-1) X^T Y
    double *ma_est = performMultivariateLinearRegression(length - p_temp, q, X, Y);
    if (!ma_est) { free(ar_resid); free(ar_temp); exit(EXIT_FAILURE); }

    // Step 6: Copy estimates to output
    for (int i = 0; i < q; i++) theta[i] = ma_est[i]; // MA coefficients θ_j
    theta[q] = ma_est[q]; // Intercept μ
    free(ma_est);
    free(ar_resid);
    free(ar_temp);
}

/**
 * Dynamically selects AR and MA orders for ARIMA using ACF and PACF diagnostics.
 * 
 * The algorithm computes the ACF to identify MA order (q) based on significant lags, and iteratively computes PACF via Yule-Walker
 * to determine AR order (p) where partial correlations drop below a significance threshold (2/√n). It ensures minimum orders of 1
 * if significant lags are found, approximating the series structure y_t = Σφ_j y_{t-j} + Σθ_k ε_{t-k} + ε_t.
 * 
 * Used in ARIMA to automate order selection, enhancing model fit. Exists to adaptively choose p and q based on data-driven
 * statistical significance, reducing manual tuning and improving forecast accuracy.
 * 
 * @param series Input time series.
 * @param length Length of the series.
 * @param maxLag Maximum lag to check.
 * @param finalAR Output pointer for AR order (p).
 * @param finalMA Output pointer for MA order (q).
 */
void selectOrdersWithFeedback(const double series[], int length, int maxLag, int *finalAR, int *finalMA) {
    // Step 1: Compute ACF to estimate MA order
    double acf[maxLag + 1];
    computeACF(series, length, maxLag, acf);
    int p = 0, q = 0;

    // Step 2: Identify significant ACF lags for q
    // Why: MA(q) has ACF significant up to lag q, then drops
    for (int i = 1; i <= maxLag; i++) {
        if (fabs(acf[i]) > 2.0 / sqrt(length)) { // Threshold: |r_k| > 2/√n (approx 95% CI)
            q = i; // Last significant lag suggests MA order
        } else {
            break; // Stop at first insignificant lag
        }
    }

    // Step 3: Compute PACF to estimate AR order
    // Why: AR(p) has PACF significant up to lag p, then drops
    double pacf[maxLag + 1];
    pacf[0] = 1.0; // PACF at lag 0 is always 1
    for (int k = 1; k <= maxLag; k++) {
        double phi[k + 1];
        yuleWalker(series, length, k, phi); // Fit AR(k) to get φ_k
        pacf[k] = phi[k - 1]; // PACF_k = φ_k from AR(k) fit
        if (fabs(pacf[k]) > 2.0 / sqrt(length)) { // Threshold: |φ_k| > 2/√n
            p = k; // Last significant lag suggests AR order
        } else {
            break; // Stop at first insignificant lag
        }
    }

    // Step 4: Set final orders, ensuring minimum of 1 if significant lags found
    // Why: Avoids degenerate models (p=0, q=0) when data suggests structure
    *finalAR = p > 0 ? p : 1;
    *finalMA = q > 0 ? q : 1;
}


/**
 * Detects and adjusts outliers in a time series using Median Absolute Deviation (MAD) for robust preprocessing.
 * 
 * Computes the median and MAD of the series, identifies outliers as points beyond \( k \cdot MAD / 0.6745 \) from the median
 * (where \( k = 2.5 \)), and caps them at the threshold to mitigate their impact on ARIMA estimation.
 * 
 * **Method Used**: 
 * - Sorts a copy of the series to find the median.
 * - Computes MAD as the median of absolute deviations from the median.
 * - Adjusts values exceeding \( median \pm k \cdot MAD / 0.6745 \) (scaled for normal consistency).
 * 
 * **Why This Method**: 
 * - **Robustness**: Median and MAD are less sensitive to outliers than mean and standard deviation, ideal for preprocessing noisy data like sampleData.
 * - **Simplicity**: Efficiently reduces extreme values without requiring complex iterative methods.
 * 
 * **Downsides and Limitations**:
 * - **Sorting**: \( O(n \log n) \) complexity due to qsort; could optimize with a linear-time median algorithm.
 * - **Threshold**: Fixed \( k = 2.5 \) may be too strict or lenient depending on data; could be configurable.
 * 
 * Used in `forecastARIMA` to preprocess the input series, enhancing model robustness against outliers.
 * 
 * @param series Input/output series (modified in-place).
 * @param length Length of series.
 * @return Number of outliers adjusted.
 */
int adjustOutliers(double series[], int length) {
    // Step 1: Allocate temporary array for sorting
    double *temp = malloc(length * sizeof(double));
    copyArray(series, temp, length);

    // Step 2: Compute median
    // Why: Robust central tendency measure
    qsort(temp, length, sizeof(double), (int (*)(const void*, const void*))strcmp); // Note: Replace strcmp with proper double comparison
    double median = (length % 2) ? temp[length / 2] : (temp[length / 2 - 1] + temp[length / 2]) / 2.0;

    // Step 3: Compute MAD (Median Absolute Deviation)
    // Formula: MAD = median(|x_i - median|)
    for (int i = 0; i < length; i++) temp[i] = fabs(series[i] - median);
    qsort(temp, length, sizeof(double), (int (*)(const void*, const void*))strcmp);
    double mad = (length % 2) ? temp[length / 2] : (temp[length / 2 - 1] + temp[length / 2]) / 2.0;
    double threshold = 2.5 * mad / 0.6745; // k=2.5, scaled for normal distribution consistency

    // Step 4: Adjust outliers by capping at threshold
    // Why: Reduces extreme values’ impact on parameter estimation
    int outliers = 0;
    for (int i = 0; i < length; i++) {
        double dev = fabs(series[i] - median);
        if (dev > threshold) {
            series[i] = (series[i] > median) ? median + threshold : median - threshold; // Cap at ±threshold
            outliers++;
        }
    }

    // Step 5: Clean up and return count
    free(temp);
    return outliers;
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
    DEBUG_PRINT("Entering forecastARIMA: seriesLength=%d, p=%d, d=%d, q=%d\n", seriesLength, p, d, q);

    // Step 1: Auto-select orders if any parameter is -1
    if (p == -1 || d == -1 || q == -1) {
        DEBUG_PRINT("Auto-selection triggered: p=%d, d=%d, q=%d\n", p, d, q);
        int auto_p, auto_q;
        int maxLag = 10; // Maximum lag for ACF/PACF analysis
        DEBUG_PRINT("Calling selectOrdersWithFeedback with maxLag=%d\n", maxLag);
        selectOrdersWithFeedback(series, seriesLength, maxLag, &auto_p, &auto_q);
        DEBUG_PRINT("Auto-selected p=%d, q=%d from ACF/PACF\n", auto_p, auto_q);

        double tStat, pValue;
        int auto_d = 0;
        double *tempSeries = malloc(seriesLength * sizeof(double));
        copyArray(series, tempSeries, seriesLength);
        int tempLength = seriesLength;
        DEBUG_PRINT("Starting ADF test loop for d, initial length=%d\n", tempLength);
        while (auto_d < 2 && !ADFTestExtendedAutoLag(tempSeries, tempLength, MODEL_CONSTANT_ONLY, &tStat, &pValue)) {
            DEBUG_PRINT("ADF test failed: tStat=%.4f, pValue=%.4f, auto_d=%d\n", tStat, pValue, auto_d);
            double *diff = differenceSeries(tempSeries, tempLength, 1);
            free(tempSeries);
            tempSeries = diff;
            tempLength--;
            auto_d++;
        }
        free(tempSeries);
        DEBUG_PRINT("Auto-selected d=%d after ADF test\n", auto_d);

        p = (p == -1) ? auto_p : p;
        d = (d == -1) ? auto_d : d;
        q = (q == -1) ? auto_q : q;
        DEBUG_PRINT("Final orders after auto-selection: p=%d, d=%d, q=%d\n", p, d, q);
        printf("Auto-selected orders: p=%d, d=%d, q=%d\n", p, d, q);
    }

    // Step 2: Validate data sufficiency
    DEBUG_PRINT("Validating data sufficiency: seriesLength=%d, p+d+q+1=%d\n", seriesLength, p + d + q + 1);
    if (seriesLength < p + d + q + 1) {
        fprintf(stderr, "Error: Insufficient data for ARIMA(%d,%d,%d). Need %d points, got %d.\n",
                p, d, q, p + d + q + 1, seriesLength);
        exit(EXIT_FAILURE);
    }

    // Step 3: Difference series to achieve stationarity
    int currentLength = seriesLength;
    DEBUG_PRINT("Differencing series: d=%d, initial length=%d\n", d, currentLength);
    double *currentSeries = ensureStationary(series, &currentLength, d);
    DEBUG_PRINT("After differencing: currentLength=%d\n", currentLength);
    if (p > currentLength / 2) {
        DEBUG_PRINT("Capping p: original p=%d, new p=%d (half currentLength)\n", p, currentLength / 2);
        p = currentLength / 2;
    }

    // Step 4: Estimate AR parameters if p > 0
    double *arEstimates = NULL;
    if (p > 0 && currentLength > p) {
        //DEBUG_PRINT("Estimating AR parameters: p=%d, length=%d\n", p, currentLength);
        arEstimates = malloc((p + 1) * sizeof(double));
        if (!arEstimates) { free(currentSeries); exit(EXIT_FAILURE); }
        estimateARWithCMLE(currentSeries, currentLength, p, arEstimates);
        // DEBUG_PRINT("AR estimates: ");
        for (int i = 0; i < p; i++) DEBUG_PRINT("phi[%d]=%.4f, ", i, arEstimates[i]);
        DEBUG_PRINT("mu=%.4f\n", arEstimates[p]);
        if (!checkRoots(arEstimates, p, 1)) {
            DEBUG_PRINT("Adjusting AR coefficients for stationarity\n");
            fprintf(stderr, "Warning: AR coefficients adjusted.\n");
            for (int i = 0; i < p; i++) arEstimates[i] *= 0.95;
            DEBUG_PRINT("Adjusted AR estimates: ");
            for (int i = 0; i < p; i++) DEBUG_PRINT("phi[%d]=%.4f, ", i, arEstimates[i]);
            DEBUG_PRINT("mu=%.4f\n", arEstimates[p]);
        }
    } else {
        DEBUG_PRINT("Skipping AR estimation: p=%d, length=%d\n", p, currentLength);
    }

    // Step 5: Compute residuals for MA estimation
    double *arResiduals = NULL;
    if (q > 0 && p > 0 && currentLength > p) {
        //DEBUG_PRINT("Computing AR residuals: length=%d, p=%d\n", currentLength, p);
        arResiduals = malloc(sizeof(double) * (currentLength - p));
        if (!arResiduals) { free(currentSeries); if (arEstimates) free(arEstimates); exit(EXIT_FAILURE); }
        for (int t = p; t < currentLength; t++) {
            double pred = arEstimates[p];
            for (int j = 0; j < p; j++) pred += arEstimates[j] * currentSeries[t - j - 1];
            arResiduals[t - p] = currentSeries[t] - pred;
        }
        //DEBUG_PRINT("First few AR residuals: %.4f, %.4f, %.4f\n", arResiduals[0], arResiduals[1], arResiduals[2]);
    } else if (q > 0) {
        //DEBUG_PRINT("No AR, using series as residuals: length=%d\n", currentLength);
        arResiduals = malloc(sizeof(double) * currentLength);
        if (!arResiduals) { free(currentSeries); if (arEstimates) free(arEstimates); exit(EXIT_FAILURE); }
        copyArray(currentSeries, arResiduals, currentLength);
    }

    // Step 6: Estimate MA parameters if q > 0
    double *maEstimates = NULL;
    if (q > 0 && arResiduals && currentLength - p > q) {
        DEBUG_PRINT("Estimating MA parameters: q=%d, residual length=%d\n", q, currentLength - p);
        maEstimates = estimateMAWithMLE(arResiduals, currentLength - p, q);
        DEBUG_PRINT("MA estimates: ");
        for (int i = 0; i < q; i++) DEBUG_PRINT("theta[%d]=%.4f, ", i, maEstimates[i]);
        DEBUG_PRINT("mu=%.4f\n", maEstimates[q]);
    } else {
        DEBUG_PRINT("Skipping MA estimation: q=%d, residuals available=%s\n", q, arResiduals ? "yes" : "no");
    }

    // Step 7: Allocate forecast array and initialize past errors
    double *forecast = malloc(sizeof(double) * FORECAST_ARRAY_SIZE);
    if (!forecast) { free(currentSeries); if (arEstimates) free(arEstimates); if (maEstimates) free(maEstimates); if (arResiduals) free(arResiduals); exit(EXIT_FAILURE); }
    double pastErrors[q];
    memset(pastErrors, 0, q * sizeof(double));
    if (q > 0 && arResiduals) {
        DEBUG_PRINT("Initializing past errors for MA: q=%d\n", q);
        for (int i = 0; i < q && i < currentLength - p; i++) {
            pastErrors[i] = arResiduals[currentLength - p - 1 - i];
            DEBUG_PRINT("pastErrors[%d]=%.4f\n", i, pastErrors[i]);
        }
    }

    // Step 8: Compute one-step forecast
    double oneStep = arEstimates ? arEstimates[p] : 0.0;
    if (p > 0) {
        for (int j = 0; j < p; j++) oneStep += arEstimates[j] * currentSeries[currentLength - j - 1];
        //DEBUG_PRINT("AR contribution to one-step: %.4f\n", oneStep);
    } else {
        oneStep = currentSeries[currentLength - 1];
       //DEBUG_PRINT("No AR, using last value: %.4f\n", oneStep);
    }
    if (q > 0 && maEstimates) {
        for (int j = 0; j < q; j++) oneStep += maEstimates[j] * pastErrors[j];
        //DEBUG_PRINT("MA contribution added, one-step: %.4f\n", oneStep);
    }
    forecast[0] = oneStep;
    //DEBUG_PRINT("One-step forecast: %.4f\n", forecast[0]);

    // Step 9: Generate multi-step forecasts recursively
    DEBUG_PRINT("Generating multi-step forecasts up to horizon=%d\n", FORECAST_HORIZON);
    for (int h = 1; h < FORECAST_HORIZON; h++) {
        double f = arEstimates ? arEstimates[p] : 0.0;
        for (int j = 0; j < p; j++) {
            double value = (h - j - 1 >= 0) ? forecast[h - j - 1] : currentSeries[currentLength - j - 1];
            f += arEstimates[j] * value;
        }
        if (q > 0 && maEstimates) {
            for (int j = 0; j < q; j++) {
                double error = (h - j - 1 >= 0) ? 0.0 : pastErrors[j];
                f += maEstimates[j] * error;
            }
        }
        forecast[h] = f;
        DEBUG_PRINT("Forecast at step %d: %.4f\n", h, forecast[h]);
    }

    // Step 10: Compute forecast variance
    //DEBUG_PRINT("Computing forecast variance\n");
    double psi[p + q];
    memset(psi, 0, sizeof(psi));
    for (int i = 0; i < p; i++) psi[i] = arEstimates ? arEstimates[i] : 0.0;
    for (int i = 0; i < q; i++) psi[p + i] = maEstimates ? maEstimates[i] : 0.0;
    double sigma2 = 0.0;
    if (arResiduals) {
        double sum_sq = 0.0;
        int n = currentLength - (p > 0 ? p : 0);
        for (int i = 0; i < n; i++) sum_sq += arResiduals[i] * arResiduals[i];
        sigma2 = sum_sq / n;
        DEBUG_PRINT("Residual variance (sigma^2): %.4f\n", sigma2);
    }
    forecast[FORECAST_HORIZON] = sigma2;
    double var = sigma2;
    for (int h = 1; h < FORECAST_HORIZON; h++) {
        for (int j = 0; j < h && j < p + q; j++) var += sigma2 * psi[j] * psi[j];
        //DEBUG_PRINT("Cumulative variance at step %d: %.4f\n", h, var);
    }
    forecast[FORECAST_HORIZON + 1] = var;

    // Step 11: Integrate forecasts if differenced
    if (d > 0) {
        double recoveryValue = series[seriesLength - 1];
        DEBUG_PRINT("Integrating forecasts: d=%d, recoveryValue=%.4f\n", d, recoveryValue);
        double *integrated = integrateSeries(forecast, recoveryValue, FORECAST_HORIZON);
        memcpy(forecast, integrated, FORECAST_HORIZON * sizeof(double));
        free(integrated);
        DEBUG_PRINT("Integrated forecasts: ");
        for (int i = 0; i < FORECAST_HORIZON; i++) DEBUG_PRINT("%.4f ", forecast[i]);
        DEBUG_PRINT("\n");
    }

    // Step 12: Clean up and return
    DEBUG_PRINT("Cleaning up and returning forecast\n");
    free(currentSeries);
    if (arEstimates) free(arEstimates);
    if (maEstimates) free(maEstimates);
    if (arResiduals) free(arResiduals);
    return forecast;
}

/**
 * Computes the autocorrelation function (ACF) of residuals and returns the maximum absolute value for diagnostic purposes.
 * 
 * Uses the standard ACF formula \( r_k = \frac{\sum_{t=1}^{n-k} (e_t - \bar{e})(e_{t+k} - \bar{e})}{\sum_{t=1}^{n} (e_t - \bar{e})^2} \)
 * to assess remaining correlation in ARIMA residuals, returning the largest \( |r_k| \) beyond lag 0 to indicate model fit quality.
 * 
 * **Method Used**: 
 * - Delegates to `computeACF` for efficiency.
 * - Scans ACF values from lag 1 to maxLag for the maximum absolute value.
 * 
 * **Why This Method**: 
 * - **Diagnostic**: Significant residual ACF suggests unmodeled structure, guiding ARIMA refinement.
 * - **Simplicity**: Reuses existing ACF computation, focusing on a key summary statistic.
 * 
 * **Downsides and Limitations**:
 * - **Stationarity**: Assumes residuals are stationary; non-stationary residuals skew results.
 * - **Threshold**: Max value alone lacks context; should be paired with significance bounds (e.g., 2/√n).
 * 
 * Used in `main` to evaluate ARIMA fit post-forecasting, aiding model validation.
 * 
 * @param residuals Input residuals from ARIMA fit.
 * @param length Length of residuals.
 * @param maxLag Maximum lag to compute ACF for.
 * @param acf Output array for ACF values (length maxLag + 1).
 * @return Maximum absolute ACF value beyond lag 0.
 */
double computeResidualACF(const double residuals[], int length, int maxLag, double acf[]) {
    // Step 1: Compute ACF up to maxLag
    // Why: Measures residual correlation structure
    computeACF(residuals, length, maxLag, acf);

    // Step 2: Find maximum absolute ACF value beyond lag 0
    // Why: Indicates largest remaining serial correlation
    double max_acf = 0.0;
    for (int i = 1; i <= maxLag; i++) {
        if (fabs(acf[i]) > max_acf) max_acf = fabs(acf[i]); // \( max(|r_k|) \) for k > 0
    }
    return max_acf;
}

/**
 * Computes the Ljung-Box test statistic to assess if ARIMA residuals are white noise.
 * 
 * Calculates \( Q = n(n+2) \sum_{k=1}^{m} \frac{r_k^2}{n-k} \) from residual ACF, where large values suggest significant
 * autocorrelation, rejecting the null hypothesis of white noise residuals. Used as a diagnostic in ARIMA modeling.
 * 
 * **Method Used**: 
 * - Computes ACF via `computeACF`.
 * - Sums weighted squared autocorrelations, adjusted for sample size reduction per lag.
 * 
 * **Why This Method**: 
 * - **Statistical Test**: Standard metric for residual independence, critical for ARIMA validation.
 * - **Efficiency**: Leverages existing ACF function, minimizing additional computation.
 * 
 * **Downsides and Limitations**:
 * - **Approximation**: Assumes asymptotic chi-squared distribution; less reliable for small \( n \).
 * - **Lag Choice**: Fixed maxLag (e.g., 10) may miss longer-term correlations or overfit short series.
 * 
 * Used in `main` to quantify residual whiteness post-forecasting, enhancing model evaluation.
 * 
 * @param residuals Input residuals from ARIMA fit.
 * @param length Length of residuals.
 * @param maxLag Maximum lag to test.
 * @return Ljung-Box Q statistic (large values suggest non-white residuals).
 */
double computeLjungBox(const double residuals[], int length, int maxLag) {
    // Step 1: Compute ACF up to maxLag
    double acf[maxLag + 1];
    computeACF(residuals, length, maxLag, acf);

    // Step 2: Compute Ljung-Box statistic
    // Formula: \( Q = n(n+2) \sum_{k=1}^{m} \frac{r_k^2}{n-k} \)
    double Q = 0.0;
    for (int k = 1; k <= maxLag; k++) {
        Q += (acf[k] * acf[k]) / (length - k); // Weighted term: \( \frac{r_k^2}{n-k} \)
    }
    Q *= length * (length + 2); // Scale by \( n(n+2) \)

    // Step 3: Return Q statistic
    return Q;
}

#define SERIES_LENGTH 175         // Length of sampleData subset for forecasting
#define DEFAULT_AR_ORDER 2        // Default AR order for diagnostics
#define DEFAULT_DIFF_ORDER 1      // Default differencing order
#define DEFAULT_MA_ORDER 4        // Default MA order for diagnostics
#define AUTO_MAX_LAG 10           // Max lag for ACF/PACF in auto-selection
#define DIAGNOSTIC_LAG 10         // Max lag for diagnostics (ACF, Ljung-Box)
#define ACF_ARRAY_SIZE (DIAGNOSTIC_LAG + 1) // ACF array size (lag 0 to 10)

double* preprocessSeries(double *series, int length) {
    double *adjusted = malloc(length * sizeof(double));
    copyArray(series, adjusted, length);
    adjustOutliers(adjusted, length);
    return adjusted;
}

double* computeForecast(double *series, int length, int p, int d, int q) {
    return forecastARIMA(series, length, p, d, q);
}


/**
 * @brief Computes and displays diagnostic metrics for an ARIMA model fit.
 *
 * @details Evaluates the ARIMA model by generating residuals, computing autocorrelation function (ACF), 
 * and performing the Ljung-Box test, then prints forecasts and diagnostics to assess model adequacy.
 *
 * **Steps Explained**:
 * - **Step 1: Differencing**: Applies \( d \) differences to the series to ensure stationarity using 
 *   `ensureStationary`, adjusting the length accordingly.
 *   
 * - **Step 2: AR Estimation**: Fits an AR(p) model to the differenced series with `estimateARWithCMLE`, 
 *   producing coefficients \( \phi_1, ..., \phi_p, \mu \).
 * 
 * - **Step 3: AR Residuals**: Computes residuals from the AR fit by subtracting predictions from observations.
 *  
 * - **Step 4: MA Estimation**: Fits an MA(q) model to the AR residuals with `estimateMAWithMLE`, yielding 
 *   coefficients \( \theta_1, ..., \theta_q, \mu \).
 * 
 * - **Step 5: Final Residuals**: Calculates final residuals by adjusting AR residuals with the MA model.
 * 
 * - **Step 6: Diagnostics**: Computes the maximum absolute ACF beyond lag 0 and the Ljung-Box Q statistic 
 *   to assess residual whiteness.
 *   
 * - **Step 7: Output**: Prints the forecast, 1-step and cumulative variances, max ACF, and Ljung-Box statistic.
 * 
 * - **Step 8: Cleanup**: Frees all allocated memory to prevent leaks.
 *
 * @param series Input time series array (post-preprocessing).
 * @param length Original length of the series.
 * @param forecast Forecast array from `computeForecast`.
 * @param p AR order for diagnostics.
 * @param d Differencing order for diagnostics.
 * @param q MA order for diagnostics.
 */
void computeDiagnostics(double *series, int length, double *forecast, int p, int d, int q) {
    // Step 1: Apply differencing to ensure stationarity
    int adjLength = length;
    double *current = ensureStationary(series, &adjLength, d);

    // Step 2: Estimate AR parameters
    double *arEst = malloc((p + 1) * sizeof(double));
    estimateARWithCMLE(current, adjLength, p, arEst);

    // Step 3: Compute AR residuals
    double *arRes = malloc((adjLength - p) * sizeof(double));
    for (int t = p; t < adjLength; t++) {
        double pred = arEst[p]; // Start with intercept
        for (int j = 0; j < p; j++) pred += arEst[j] * current[t - j - 1]; // Add AR terms
        arRes[t - p] = current[t] - pred; // Residual = actual - predicted
    }

    // Step 4: Estimate MA parameters on AR residuals
    double *maEst = estimateMAWithMLE(arRes, adjLength - p, q);

    // Step 5: Compute final residuals with MA adjustment
    double *res = malloc((adjLength - p) * sizeof(double));
    for (int t = q; t < adjLength; t++) {
        double pred = maEst[q]; // Start with MA intercept
        for (int j = 0; j < q && t - j - 1 >= p; j++) pred += maEst[j] * arRes[t - j - 1 - p]; // Add MA terms
        res[t - p] = arRes[t - p] - pred; // Final residual
    }

    // Step 6: Compute diagnostic metrics
    double acf[ACF_ARRAY_SIZE];
    double max_acf = computeResidualACF(res, adjLength - p, DIAGNOSTIC_LAG, acf);
    double lb_stat = computeLjungBox(res, adjLength - p, DIAGNOSTIC_LAG);

    // Step 7: Print forecast and diagnostics
    printf("ARIMA Forecast:\n");
    for (int i = 0; i < FORECAST_HORIZON; i++) printf("%.4f ", forecast[i]);
    printf("\n1-step forecast variance: %.4f\n", forecast[FORECAST_HORIZON]);
    printf("Cumulative forecast variance at step %d: %.4f\n", FORECAST_HORIZON - 1, forecast[FORECAST_HORIZON + 1]);
    printf("Max residual ACF (lag 1-%d): %.4f\n", DIAGNOSTIC_LAG, max_acf);
    printf("Ljung-Box Q statistic (lag %d): %.4f\n", DIAGNOSTIC_LAG, lb_stat);

    // Step 8: Free allocated memory
    free(current);
    free(arEst);
    free(arRes);
    free(maEst);
    free(res);
}


/*==================== Main Function ====================*/
int main(void) {
    
    double sampleData[] = {10.544653, 10.688583, 10.666841, 10.662732, 10.535033, 10.612065, 10.577628, 10.524487, 10.511290, 10.520899, 10.605484, 10.506456, 10.693456, 10.667562, 10.640863, 10.553473, 10.684760, 10.752397, 10.671068, 10.667091, 10.641893, 10.625706, 10.701795, 10.607544, 10.689169, 10.695256, 10.717050, 10.677475, 10.691141, 10.730298, 10.732664, 10.710082, 10.713123, 10.759815, 10.696599, 10.663845, 10.716597, 10.780855, 10.795759, 10.802620, 10.720496, 10.753401, 10.709436, 10.746909, 10.737377, 10.754609, 10.765248, 10.692602, 10.837926, 10.755324, 10.756213, 10.843190, 10.862529, 10.751269, 10.902390, 10.817731, 10.859796, 10.887362, 10.835401, 10.824412, 10.860767, 10.819504, 10.907496, 10.831528, 10.821727, 10.830010, 10.915317, 10.858694, 10.921139, 10.927524, 10.894352, 10.889785, 10.956356, 10.938758, 11.093567, 10.844841, 11.094493, 11.035941, 10.982765, 11.071057, 10.996308, 11.099276, 11.142057, 11.137176, 11.157537, 11.007247, 11.144075, 11.183029, 11.172096, 11.164571, 11.192833, 11.227109, 11.141589, 11.311490, 11.239783, 11.295933, 11.199566, 11.232262, 11.333208, 11.337874, 11.322334, 11.288216, 11.280459, 11.247973, 11.288277, 11.415095, 11.297583, 11.360763, 11.288338, 11.434631, 11.456051, 11.578981, 11.419166, 11.478404, 11.660141, 11.544303, 11.652028, 11.638368, 11.651792, 11.621518, 11.763853, 11.760687, 11.771138, 11.678104, 11.783163, 11.932094, 11.948678, 11.962627, 11.937934, 12.077570, 11.981595, 12.096366, 12.032683, 12.094221, 11.979764, 12.217793, 12.235930, 12.129859, 12.411867, 12.396301, 12.413920, 12.445867, 12.480462, 12.470674, 12.537774, 12.562252, 12.810248, 12.733546, 12.861890, 12.918012, 13.033087, 13.245610, 13.184196, 13.414342, 13.611838, 13.626345, 13.715446, 13.851129, 14.113374, 14.588537, 14.653982, 15.250756, 15.618371, 16.459558, 18.144264, 23.523062, 40.229511, 38.351265, 38.085281, 37.500885, 37.153946, 36.893066, 36.705956, 36.559536, 35.938847, 36.391586, 36.194046, 36.391586, 36.119102, 35.560543, 35.599018, 34.958851, 35.393860, 34.904797, 35.401318, 34.863518, 34.046680, 34.508522, 34.043182, 34.704235, 33.556644, 33.888481, 33.533638, 33.452129, 32.930935, 32.669731, 32.772537, 32.805634, 32.246761, 32.075809, 31.864927, 31.878294, 32.241131, 31.965626, 31.553604, 30.843288, 30.784569, 31.436094, 31.170496, 30.552132, 30.500242, 30.167421, 29.911989, 29.586046, 29.478958, 29.718994, 29.611095, 29.557945, 28.463432, 29.341291, 28.821512, 28.447210, 27.861872, 27.855633, 27.910660, 28.425800, 27.715517, 27.617193, 27.093372, 26.968832, 26.977205, 27.170172, 26.251677, 26.633236, 26.224941, 25.874708, 25.593761, 26.392395, 24.904768, 25.331600, 24.530737, 25.074808, 25.310865, 24.337013, 24.442986, 24.500193, 24.130409, 24.062714, 24.064592, 23.533037, 23.977909, 22.924667, 22.806379, 23.130791, 22.527645, 22.570505, 22.932512, 22.486126, 22.594856, 22.383926, 22.115181, 22.105082, 21.151754, 21.074114, 21.240192, 20.977468, 20.771507, 21.184586, 20.495111, 20.650751, 20.656075, 20.433039, 20.005697, 20.216360, 19.982117, 19.703951, 19.572884, 19.332155, 19.544645, 18.666328, 19.219872, 18.934229, 19.186989, 18.694986, 18.096903, 18.298306, 17.704309, 18.023785, 18.224157, 18.182484, 17.642824, 17.739542, 17.474176, 17.270575, 17.604120, 17.631210, 16.639175, 17.107626, 17.024216, 16.852285, 16.780111, 16.838861, 16.539309, 16.092861, 16.131529, 16.221350, 16.087164, 15.821659, 15.695448, 15.693087, 16.047991, 15.682863, 15.724131, 15.263708, 15.638486, 15.443835, 15.602257, 15.122874, 14.918172, 14.968882, 14.843689, 14.861169, 15.052527, 15.056897, 14.690192, 14.686479, 14.567565, 14.365212, 14.253309, 14.289158, 14.227124, 14.069589, 14.074703, 13.869432, 13.861959, 13.782178, 13.882711, 13.908362, 13.727641, 13.600214, 13.594969, 13.535290, 13.602018, 13.502626, 13.579159, 13.207825, 13.426789, 13.178141, 13.286413, 12.958746, 13.189507, 13.079733, 13.138372, 12.986096, 12.854589, 12.858962, 12.903029, 12.852099, 12.644394, 12.558786, 12.636994};

    int dataLength = sizeof(sampleData) / sizeof(sampleData[0]);

    double *adjustedSeries = preprocessSeries(sampleData, dataLength);
    double *forecast = computeForecast(adjustedSeries, SERIES_LENGTH, -1, -1, -1);
    computeDiagnostics(adjustedSeries, dataLength, forecast, DEFAULT_AR_ORDER, DEFAULT_DIFF_ORDER, DEFAULT_MA_ORDER);

    free(adjustedSeries);
    free(forecast);
    return 0;
    return 0;
}

