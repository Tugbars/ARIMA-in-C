/**
 * @defgroup LinearRegressionFunctions Linear Regression Functions
 * @brief Functions used to estimate model parameters via least‐squares.
 *
 * These functions provide implementations for several regression routines:
 *
 * - **Univariate Linear Regression (LR1):**
 *   - **LR1(double arr_x[], double arr_y[], int length):**
 *     Computes the slope (β) and intercept from regressing y on x. It first centers the data
 *     (by subtracting the means), computes the standard deviations, and then calculates the Pearson
 *     correlation coefficient. The slope is computed using the relationship:
 *     \f[
 *         \beta = \mathrm{Corr} \times \frac{S_y}{S_x}
 *     \f]
 *     and the intercept as:
 *     \f[
 *         \mathrm{intercept} = \bar{y} - \beta\, \bar{x}
 *     \f]
 *   - **LR1_Pred(...):**
 *     Given the estimated slope and intercept, this function produces the predicted values for a
 *     provided predictor array.
 *
 * - **Bivariate Linear Regression (LR2):**
 *   - **LR2(double arr_x1[], double arr_x2[], double arr_y[], int length):**
 *     Computes regression estimates when two predictors are available. The function calculates sums,
 *     products, and uses closed‐form formulas (based on covariances) to obtain β₁, β₂, and the intercept.
 *   - **LR2_Pred(...):**
 *     Produces predictions given two predictor arrays and their corresponding estimated coefficients.
 *
 * - **Multivariate Regression (LRM):**
 *   - **LRM(int m, int n, double X[][n], double Y[][1]):**
 *     For a model with n predictors and m observations, this function first normalizes the design
 *     matrix (by subtracting column means). It then forms the normal equations (XᵀX) and solves for
 *     the coefficients using matrix inversion (selecting an appropriate routine for a 3×3 or 4×4 matrix).
 *     The intercept is computed separately using the means of X and Y.
 *   - **LRM_Pred(...):**
 *     Uses the estimated coefficients from LRM to compute predicted values for a given matrix of predictors.
 *
 * - **Correlation Function (Corr):**
 *   - **Corr(double arr_x[], double arr_y[], int length):**
 *     Computes the Pearson correlation coefficient between two arrays.
 *
 * @note Detailed parameter and return-value descriptions for each function are provided
 * in the corresponding function documentation.
 *
 * @{
 */

/* (Place the function documentation for LR1, LR1_Pred, LR2, LR2_Pred, LRM, LRM_Pred, and Corr here.) */

/** @} */ // end of LinearRegressionFunctions group

/**
 * @defgroup StationarityTesting Stationarity Testing and Data Preprocessing
 * @brief Functions for differencing a time series to achieve stationarity and for drift adjustment.
 *
 * Before applying a forecasting model it is common practice to difference the series if it is non‐stationary.
 *
 * - **DFTest (Dickey–Fuller Test):**
 *   - **DFTest(double arr[], double arr_recov[], int length):**
 *     This function iteratively applies a regression test to determine the order of differencing (d)
 *     required to achieve stationarity. In each iteration, the function:
 *       - Creates a “lagged” version and a “lead” version (one time step ahead) of the series.
 *       - Uses LR1 to regress the lead on the lagged series.
 *       - Checks whether the estimated coefficient is sufficiently close to unity (within a tolerance
 *         of approximately ±1.001). If not, the series is differenced (by subtracting the previous value)
 *         and the test is repeated.
 *     The function returns the order of differencing applied (d) and stores recovery information in
 *     the array `arr_recov` so that the drift can be added back later.
 *
 * - **Drift Adjustment:**
 *   - **Drift(double arr[], double arr_stry[], int length, int d):**
 *     After differencing, if d > 0, this function adjusts the series to “recover” its original level.
 *     Forecasts generated on the differenced scale need to be cumulatively summed (or otherwise adjusted)
 *     using the saved drift information.
 *
 * @note Detailed descriptions for parameters and processing steps are available in the function documentation.
 *
 * @{
 */

/* (Place the function documentation for DFTest and Drift here.) */

/** @} */ // end of StationarityTesting group

/**
 * @defgroup DiagnosticMatrix Autocorrelation/Diagnostic Matrix (EAFMatrix)
 * @brief Functions for computing diagnostic matrices based on extended autocorrelations.
 *
 * The diagnostic matrix is used for model selection. In particular:
 *
 * - **EAFMatrix(double arr[], double arr_eaf[][3], int length):**
 *   This function computes an extended autocorrelation function matrix as follows:
 *     - It first computes correlations between the series and its various leads/lags (typically for lags 1, 2, and 3).
 *     - Then, it fits an AR(1) model (using LR1) and computes the prediction errors.
 *     - The autocorrelations of these error series are computed.
 *     - The process is repeated for an AR(2) model.
 *   The resulting 3×3 matrix (`arr_eaf`) encapsulates diagnostic information about the series and is used
 *   later in selecting the most appropriate forecasting model.
 *
 * @note More detailed parameter explanations are provided in the function’s documentation.
 *
 * @{
 */

/* (Place the function documentation for EAFMatrix here.) */

/** @} */ // end of DiagnosticMatrix group

/**
 * @defgroup ErrorMetrics Error Metrics
 * @brief Functions to compute forecast error metrics.
 *
 * Two primary error metrics are computed:
 *
 * - **MAPE (Mean Absolute Percentage Error):**
 *   - **MAPE(double arr_y[], double arr_y_cap[], int length):**
 *     Computes the average absolute percentage difference between forecasted values and actual values.
 *
 * - **MAE (Mean Absolute Error):**
 *   - **MAE(double arr_y[], double arr_y_cap[], int length):**
 *     Computes the average absolute error between forecasted values and actual values.
 *
 * These metrics help in assessing the quality of forecasts produced by various models.
 *
 * @note Refer to each function’s detailed documentation for parameter and return descriptions.
 *
 * @{
 */

/* (Place the function documentation for MAPE and MAE here.) */

/** @} */ // end of ErrorMetrics group

/**
 * @file forecasting.c
 * @brief Refactored time series forecasting code with modular design,
 *        improved error handling, and detailed inline documentation.
 */

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
 * @brief Converts a 1D array into a 2D matrix with one column.
 *
 * @param numElements The number of elements.
 * @param array The input array.
 * @param matrix Output 2D matrix (numElements x 1).
 */
void arrayToMatrix(int numElements, const double array[], double matrix[][1])
{
  for (int i = 0; i < numElements; i++)
  {
    matrix[i][0] = array[i];
  }
}

/**
 * @brief Converts a 2D matrix (with one column) to a 1D array.
 *
 * @param numRows The number of rows.
 * @param matrix Input 2D matrix.
 * @param array Output array.
 */
void matrixToArray(int numRows, double matrix[][1], double array[])
{
  for (int i = 0; i < numRows; i++)
  {
    array[i] = matrix[i][0];
  }
}

/**
 * @brief Column-binds three arrays into a 2D matrix.
 *
 * @param numRows The number of rows.
 * @param array1 First array (becomes column 0).
 * @param array2 Second array (becomes column 1).
 * @param array3 Third array (becomes column 2).
 * @param result Output 2D matrix with 3 columns.
 */
void columnBind3(int numRows, double array1[], double array2[], double array3[], double result[][3])
{
  for (int i = 0; i < numRows; i++)
  {
    result[i][0] = array1[i];
    result[i][1] = array2[i];
    result[i][2] = array3[i];
  }
}

/**
 * @brief Column-binds four arrays into a 2D matrix.
 *
 * @param numRows The number of rows.
 * @param array1 First array.
 * @param array2 Second array.
 * @param array3 Third array.
 * @param array4 Fourth array.
 * @param result Output 2D matrix with 4 columns.
 */
void columnBind4(int numRows, double array1[], double array2[], double array3[], double array4[], double result[][4])
{
  for (int i = 0; i < numRows; i++)
  {
    result[i][0] = array1[i];
    result[i][1] = array2[i];
    result[i][2] = array3[i];
    result[i][3] = array4[i];
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

/**
 * @brief Performs LU decomposition with partial pivoting.
 *
 * This function decomposes a square matrix A (n x n) into a product of a lower
 * triangular matrix L and an upper triangular matrix U (stored in the same matrix),
 * while computing a pivot array that records the row exchanges.
 *
 * Partial pivoting is employed to enhance numerical stability, especially for
 * nearly singular matrices. This robust method is critical in ARIMA computations
 * where the normal equations may produce ill-conditioned matrices.
 *
 * @param n The dimension of the square matrix.
 * @param A The input matrix, which is overwritten with the combined L and U factors.
 * @param pivot An output array holding the pivot indices.
 * @return 0 on success, nonzero if A is singular.
 */
int luDecomposition(int n, double A[n][n], int pivot[n])
{
  // Initialize the pivot vector with row indices.
  for (int i = 0; i < n; i++)
  {
    pivot[i] = i;
  }
  for (int k = 0; k < n; k++)
  {
    // Partial pivoting: find the row with the maximum absolute value in column k.
    double max = fabs(A[k][k]);
    int maxIndex = k;
    for (int i = k + 1; i < n; i++)
    {
      if (fabs(A[i][k]) > max)
      {
        max = fabs(A[i][k]);
        maxIndex = i;
      }
    }
    DEBUG_PRINT("LU Decomposition: k=%d, maxIndex=%d, max=%lf\n", k, maxIndex, max);
    if (max < SINGULARITY_THRESHOLD)
    {
      DEBUG_PRINT("LU Decomposition: Matrix is singular at k=%d\n", k);
      return -1; // Singular matrix
    }
    // Swap rows k and maxIndex if needed.
    if (maxIndex != k)
    {
      for (int j = 0; j < n; j++)
      {
        double temp = A[k][j];
        A[k][j] = A[maxIndex][j];
        A[maxIndex][j] = temp;
      }
      int temp = pivot[k];
      pivot[k] = pivot[maxIndex];
      pivot[maxIndex] = temp;
      DEBUG_PRINT("LU Decomposition: Swapped rows %d and %d\n", k, maxIndex);
    }
    // LU update: compute multipliers and update the submatrix.
    for (int i = k + 1; i < n; i++)
    {
      A[i][k] /= A[k][k];
      for (int j = k + 1; j < n; j++)
      {
        A[i][j] -= A[i][k] * A[k][j];
      }
    }
    DEBUG_PRINT("LU Decomposition: Completed column %d\n", k);
  }
  return 0;
}

/**
 * @brief Solves the system LUx = b using forward and back substitution.
 *
 * Given an LU-decomposed matrix (with pivoting) and a right-hand side vector b,
 * this function computes the solution vector x.
 *
 * The solution is obtained in two steps:
 *   - Forward substitution: Solve Ly = Pb for y.
 *   - Back substitution: Solve Ux = y for x.
 *
 * @param n The dimension of the matrix.
 * @param LU The LU-decomposed matrix (combined L and U).
 * @param pivot The pivot indices from LU decomposition.
 * @param b The right-hand side vector.
 * @param x Output solution vector.
 */
void luSolve(int n, double LU[n][n], int pivot[n], double b[n], double x[n])
{
  double y[n];
  // Forward substitution: solve Ly = Pb.
  for (int i = 0; i < n; i++)
  {
    y[i] = b[pivot[i]]; // Apply pivoting.
    for (int j = 0; j < i; j++)
    {
      y[i] -= LU[i][j] * y[j];
    }
    DEBUG_PRINT("luSolve Forward: i=%d, y[%d]=%lf\n", i, i, y[i]);
  }
  // Back substitution: solve Ux = y.
  for (int i = n - 1; i >= 0; i--)
  {
    x[i] = y[i];
    for (int j = i + 1; j < n; j++)
    {
      x[i] -= LU[i][j] * x[j];
    }
    x[i] /= LU[i][i];
    DEBUG_PRINT("luSolve Backward: i=%d, x[%d]=%lf\n", i, i, x[i]);
  }
}

/**
 * @brief Inverts an n x n matrix using LU decomposition.
 *
 * This function first copies the input matrix into a temporary matrix, then performs
 * LU decomposition with partial pivoting. It then solves n linear systems (one per column)
 * to compute the inverse column-by-column.
 *
 * This LU-based inversion method is preferred over the adjugate method for its improved
 * numerical stability, particularly when matrices are nearly singular.
 *
 * @param n The dimension of the matrix.
 * @param A The input matrix (n x n).
 * @param inverse The output inverse matrix (n x n).
 */
void invertMatrixLU(int n, double A[n][n], double inverse[n][n])
{
  int pivot[n];
  double LU[n][n];
  // Copy A into LU to preserve the original matrix.
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      LU[i][j] = A[i][j];
    }
  }
  if (luDecomposition(n, LU, pivot) != 0)
  {
    fprintf(stderr, "Matrix is singular\n");
    exit(EXIT_FAILURE);
  }
  double b[n], x[n];
  // Solve for each column of the inverse.
  for (int j = 0; j < n; j++)
  {
    // Set b to the jth unit vector.
    for (int i = 0; i < n; i++)
    {
      b[i] = 0.0;
    }
    b[j] = 1.0;
    luSolve(n, LU, pivot, b, x);
    for (int i = 0; i < n; i++)
    {
      inverse[i][j] = x[i];
    }
    DEBUG_PRINT("invertMatrixLU: Solved column %d\n", j);
  }
}

/**
 * @brief Wrapper for inverting a 3x3 matrix using LU decomposition.
 *
 * This wrapper ensures that only 3x3 matrices are processed. It calls the LU-based inversion
 * routine and enforces that the dimension is exactly 3.
 *
 * @param n The dimension (must be 3).
 * @param matrix The input 3x3 matrix.
 * @param inverseMatrix The output 3x3 inverse matrix.
 */
void invert3x3Matrix(int n, double matrix[][n], double inverseMatrix[][n])
{
  if (n != 3)
  {
    fprintf(stderr, "invert3x3Matrix error: Expected n == 3, got n == %d\n", n);
    exit(EXIT_FAILURE);
  }
  invertMatrixLU(3, matrix, inverseMatrix);
  DEBUG_PRINT("invert3x3Matrix: Inversion complete.\n");
}

/**
 * @brief Wrapper for inverting a 4x4 matrix using LU decomposition.
 *
 * This wrapper calls the LU-based inversion routine for 4x4 matrices and provides
 * the expected interface for legacy code.
 *
 * @param matrix The input 4x4 matrix.
 * @param inverse The output 4x4 inverse matrix.
 */
void invert4x4Matrix(double matrix[4][4], double inverse[4][4])
{
  invertMatrixLU(4, matrix, inverse);
  DEBUG_PRINT("invert4x4Matrix: Inversion complete.\n");
}

/**
 * @brief Computes the gradient for an MA(1) model's objective function.
 *
 * For a moving average model of order 1 (MA(1)), the model is defined as:
 *   y[i] = theta * lag[i] + c
 * and the objective function is the sum of squared errors:
 *   J = sum_{i=0}^{n-1} (target[i] - (theta * lag[i] + c))^2.
 *
 * The gradients with respect to theta and c are computed as:
 *   dJ/d(theta) = -2 * sum_{i=0}^{n-1} (target[i] - (theta * lag[i] + c)) * lag[i],
 *   dJ/d(c)     = -2 * sum_{i=0}^{n-1} (target[i] - (theta * lag[i] + c)).
 *
 * DEBUG_PRINT statements output the error and the cumulative gradient for each parameter.
 *
 * @param target Pointer to the target vector.
 * @param lag Pointer to the lagged input vector.
 * @param n Number of observations.
 * @param params Current parameter estimates (theta and c).
 * @param grad Output gradient vector (length 2).
 */
void computeMA1Gradient(const double *target, const double *lag, int n, const double params[2], double grad[2])
{
  grad[0] = 0.0;
  grad[1] = 0.0;
  for (int i = 0; i < n; i++)
  {
    double pred = params[0] * lag[i] + params[1];
    double error = target[i] - pred;
    grad[0] += -2.0 * error * lag[i];
    grad[1] += -2.0 * error;
    DEBUG_PRINT("MA1Grad: i=%d, pred=%lf, error=%lf, grad0_partial=%lf, grad1_partial=%lf\n",
                i, pred, error, grad[0], grad[1]);
  }
  DEBUG_PRINT("MA1Grad: Final grad[0]=%lf, grad[1]=%lf\n", grad[0], grad[1]);
}

/**
 * @brief Computes the gradient for an MA(2) model's objective function.
 *
 * For a moving average model of order 2 (MA(2)), the model is defined as:
 *   y[i] = theta1 * lag1[i] + theta2 * lag2[i] + c
 * and the objective function is:
 *   J = sum_{i=0}^{n-1} (target[i] - (theta1 * lag1[i] + theta2 * lag2[i] + c))^2.
 *
 * The gradients are computed as:
 *   dJ/d(theta1) = -2 * sum (target[i] - (theta1 * lag1[i] + theta2 * lag2[i] + c)) * lag1[i],
 *   dJ/d(theta2) = -2 * sum (target[i] - (theta1 * lag1[i] + theta2 * lag2[i] + c)) * lag2[i],
 *   dJ/d(c)      = -2 * sum (target[i] - (theta1 * lag1[i] + theta2 * lag2[i] + c)).
 *
 * DEBUG_PRINT statements output the error and the cumulative gradient for each parameter.
 *
 * @param target Pointer to the target vector.
 * @param lag1 Pointer to the first lagged input vector.
 * @param lag2 Pointer to the second lagged input vector.
 * @param n Number of observations.
 * @param params Current parameter estimates (theta1, theta2, c).
 * @param grad Output gradient vector (length 3).
 */
void computeMA2Gradient(const double *target, const double *lag1, const double *lag2, int n, const double params[3], double grad[3])
{
  grad[0] = 0.0;
  grad[1] = 0.0;
  grad[2] = 0.0;
  for (int i = 0; i < n; i++)
  {
    double pred = params[0] * lag1[i] + params[1] * lag2[i] + params[2];
    double error = target[i] - pred;
    grad[0] += -2.0 * error * lag1[i];
    grad[1] += -2.0 * error * lag2[i];
    grad[2] += -2.0 * error;
    DEBUG_PRINT("MA2Grad: i=%d, pred=%lf, error=%lf, grad0_partial=%lf, grad1_partial=%lf, grad2_partial=%lf\n",
                i, pred, error, grad[0], grad[1], grad[2]);
  }
  DEBUG_PRINT("MA2Grad: Final grad[0]=%lf, grad[1]=%lf, grad[2]=%lf\n", grad[0], grad[1], grad[2]);
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

/**
 * @brief Performs multivariate linear regression.
 *
 * @param numObservations The number of observations (rows).
 * @param numPredictors The number of predictors (columns).
 * @param X The design matrix.
 * @param Y The response vector (as a matrix with one column).
 * @return Pointer to a dynamically allocated array containing [beta coefficients..., intercept].
 *
 * This function normalizes the design matrix, computes the normal equations (XᵀX),
 * inverts the matrix (using either a 3x3 or 4x4 inversion routine), and then computes
 * the coefficients. The intercept is calculated separately.
 */
double *performMultivariateLinearRegression(int numObservations, int numPredictors, double X[][numPredictors], double Y[][1])
{
  double X_normalized[numObservations][numPredictors], Y_normalized[numObservations][1];
  double X_means[1][numPredictors], Y_mean[1][1];
  double Xt[numPredictors][numObservations], XtX[numPredictors][numPredictors], XtX_inv[numPredictors][numPredictors];
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

  // Invert XtX.
  if (numPredictors == 3)
  {
    invert3x3Matrix(3, XtX, XtX_inv);
  }
  else
  {
    invert4x4Matrix(XtX, XtX_inv);
  }

  // Compute XtX_inv_Xt = XtX_inv * Xt.
  matrixMultiply(numPredictors, numPredictors, numObservations, XtX_inv, Xt, XtX_inv_Xt);

  // Compute beta = XtX_inv_Xt * Y_normalized.
  matrixMultiply(numPredictors, numObservations, 1, XtX_inv_Xt, Y_normalized, beta);
  for (int i = 0; i < numPredictors; i++)
  {
    estimates[i] = beta[i][0];
  }

  // Compute intercept from the means.
  for (int j = 0; j < numPredictors; j++)
  {
    double col[numObservations];
    for (int i = 0; i < numObservations; i++)
    {
      col[i] = X[i][j];
    }
    X_means[0][j] = calculateMean(col, numObservations);
  }
  {
    double yCol[numObservations];
    for (int i = 0; i < numObservations; i++)
    {
      yCol[i] = Y[i][0];
    }
    Y_mean[0][0] = calculateMean(yCol, numObservations);
  }
  double intercept = Y_mean[0][0];
  for (int i = 0; i < numPredictors; i++)
  {
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

#define INITIAL_MA_LEARNING_RATE 0.001
#define MIN_MA_LEARNING_RATE 1e-6

/**
 * @brief Estimates MA(1) parameters (theta and intercept) using adaptive gradient descent.
 *
 * This function minimizes the sum of squared errors for an MA(1) model of the form
 *   y[i] = theta * lag[i] + c,
 * with objective
 *   J = sum_{i=0}^{n-1} (target[i] - (theta*lag[i] + c))^2.
 *
 * The gradient is computed for both parameters:
 *   dJ/d(theta) = -2 * sum (target[i] - (theta*lag[i] + c)) * lag[i]
 *   dJ/d(c)     = -2 * sum (target[i] - (theta*lag[i] + c))
 *
 * An adaptive learning rate is used:
 *   - If the proposed update reduces J, the learning rate is increased (multiplied by 1.1)
 *   - Otherwise, the learning rate is decreased (multiplied by 0.5)
 *
 * Debug prints trace the gradient norm, learning rate, and objective value.
 *
 * @param target Pointer to the target vector.
 * @param lag Pointer to the lagged error vector.
 * @param n Number of observations.
 * @param theta Output pointer for the estimated MA coefficient.
 * @param c Output pointer for the estimated intercept.
 */
void estimateMA1Parameters(const double *target, const double *lag, int n, double *theta, double *c)
{
  double maParams[2] = {0.0, 0.0}; // [theta, c] initial guess: 0,0
  double learningRate = INITIAL_MA_LEARNING_RATE;
  int iter = 0;
  double currentObjective = 0.0, newObjective = 0.0;
  double grad[2];

  // Compute initial objective J = sum (target - (theta * lag + c))^2
  for (int i = 0; i < n; i++)
  {
    double pred = maParams[0] * lag[i] + maParams[1];
    double error = target[i] - pred;
    currentObjective += error * error;
  }
  DEBUG_PRINT("MA1 initial objective: %lf\n", currentObjective);

  // Adaptive gradient descent loop
  while (iter < MAX_ITERATIONS)
  {
    // Compute gradient for current parameters
    grad[0] = 0.0;
    grad[1] = 0.0;
    computeMA1Gradient(target, lag, n, maParams, grad);
    double gradNorm = sqrt(grad[0] * grad[0] + grad[1] * grad[1]);
    DEBUG_PRINT("MA1 iter %d: grad norm = %lf, learningRate = %lf, currentObjective = %lf\n", iter, gradNorm, learningRate, currentObjective);
    if (gradNorm < CONVERGENCE_TOLERANCE)
      break;

    // Propose an update in the negative gradient direction
    double proposedParams[2];
    proposedParams[0] = maParams[0] - learningRate * grad[0];
    proposedParams[1] = maParams[1] - learningRate * grad[1];

    // Compute the new objective with the proposed parameters
    newObjective = 0.0;
    for (int i = 0; i < n; i++)
    {
      double pred = proposedParams[0] * lag[i] + proposedParams[1];
      double error = target[i] - pred;
      newObjective += error * error;
    }
    DEBUG_PRINT("MA1 iter %d: newObjective = %lf\n", iter, newObjective);

    // If improvement is observed, accept the update and slightly increase the learning rate.
    if (newObjective < currentObjective)
    {
      maParams[0] = proposedParams[0];
      maParams[1] = proposedParams[1];
      currentObjective = newObjective;
      learningRate *= 1.1;
    }
    else
    {
      // Otherwise, reduce the learning rate and try again.
      learningRate *= 0.5;
      if (learningRate < MIN_MA_LEARNING_RATE)
      {
        DEBUG_PRINT("MA1 iter %d: learningRate below minimum threshold, breaking\n", iter);
        break;
      }
    }
    iter++;
  }

  DEBUG_PRINT("MA1 final iter %d: theta = %lf, intercept = %lf, objective = %lf\n", iter, maParams[0], maParams[1], currentObjective);
  *theta = maParams[0];
  *c = maParams[1];
}

/**
 * @brief Estimates MA(2) parameters (theta1, theta2, and intercept) using adaptive gradient descent.
 *
 * For an MA(2) model of the form
 *   y[i] = theta1 * lag1[i] + theta2 * lag2[i] + c,
 * the objective is:
 *   J = sum_{i=0}^{n-1} (target[i] - (theta1*lag1[i] + theta2*lag2[i] + c))^2.
 *
 * The gradients are:
 *   dJ/d(theta1) = -2 * sum (target[i] - (theta1*lag1[i] + theta2*lag2[i] + c)) * lag1[i],
 *   dJ/d(theta2) = -2 * sum (target[i] - (theta1*lag1[i] + theta2*lag2[i] + c)) * lag2[i],
 *   dJ/d(c)      = -2 * sum (target[i] - (theta1*lag1[i] + theta2*lag2[i] + c)).
 *
 * An adaptive learning rate is used similarly as in MA(1) estimation.
 *
 * @param target Pointer to the target vector.
 * @param lag1 Pointer to the first lagged error vector.
 * @param lag2 Pointer to the second lagged error vector.
 * @param n Number of observations.
 * @param theta1 Output pointer for the first MA coefficient.
 * @param theta2 Output pointer for the second MA coefficient.
 * @param c Output pointer for the intercept.
 */
void estimateMA2Parameters(const double *target, const double *lag1, const double *lag2, int n,
                           double *theta1, double *theta2, double *c)
{
  double maParams[3] = {0.0, 0.0, 0.0}; // [theta1, theta2, c] initial guess
  double learningRate = INITIAL_MA_LEARNING_RATE;
  int iter = 0;
  double currentObjective = 0.0, newObjective = 0.0;
  double grad[3];

  // Compute initial objective J = sum (target - (theta1*lag1 + theta2*lag2 + c))^2
  for (int i = 0; i < n; i++)
  {
    double pred = maParams[0] * lag1[i] + maParams[1] * lag2[i] + maParams[2];
    double error = target[i] - pred;
    currentObjective += error * error;
  }
  DEBUG_PRINT("MA2 initial objective: %lf\n", currentObjective);

  // Adaptive gradient descent loop
  while (iter < MAX_ITERATIONS)
  {
    grad[0] = grad[1] = grad[2] = 0.0;
    computeMA2Gradient(target, lag1, lag2, n, maParams, grad);
    double gradNorm = sqrt(grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2]);
    DEBUG_PRINT("MA2 iter %d: grad norm = %lf, learningRate = %lf, currentObjective = %lf\n", iter, gradNorm, learningRate, currentObjective);
    if (gradNorm < CONVERGENCE_TOLERANCE)
      break;

    double proposedParams[3];
    proposedParams[0] = maParams[0] - learningRate * grad[0];
    proposedParams[1] = maParams[1] - learningRate * grad[1];
    proposedParams[2] = maParams[2] - learningRate * grad[2];

    newObjective = 0.0;
    for (int i = 0; i < n; i++)
    {
      double pred = proposedParams[0] * lag1[i] + proposedParams[1] * lag2[i] + proposedParams[2];
      double error = target[i] - pred;
      newObjective += error * error;
    }
    DEBUG_PRINT("MA2 iter %d: newObjective = %lf\n", iter, newObjective);

    if (newObjective < currentObjective)
    {
      // Accept the update if objective decreases.
      maParams[0] = proposedParams[0];
      maParams[1] = proposedParams[1];
      maParams[2] = proposedParams[2];
      currentObjective = newObjective;
      learningRate *= 1.1;
    }
    else
    {
      // Otherwise reduce the learning rate.
      learningRate *= 0.5;
      if (learningRate < MIN_MA_LEARNING_RATE)
      {
        DEBUG_PRINT("MA2 iter %d: learningRate below minimum threshold, breaking\n", iter);
        break;
      }
    }
    iter++;
  }

  DEBUG_PRINT("MA2 final iter %d: theta1 = %lf, theta2 = %lf, intercept = %lf, objective = %lf\n", iter, maParams[0], maParams[1], maParams[2], currentObjective);
  *theta1 = maParams[0];
  *theta2 = maParams[1];
  *c = maParams[2];
}

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

/*========================================================================
  Forecasting Model: AR(1)-MA(1) Hybrid (Example)
========================================================================*/

/**
 * @brief Forecasts future values using an AR(1)-MA(1) hybrid model.
 *
 * @param series The input time series.
 * @param length The length of the series.
 * @return A dynamically allocated forecast array (FORECAST_HORIZON amount of forecasts and MAPE).
 *
 * This example model combines autoregressive and moving average components.
 * It involves an iterative procedure to update the MA parameters until convergence.
 * (For brevity, this example uses a univariate update to refine two parameters.)
 */
double *forecastAR1MA1(double series[], int length)
{
  // Build differenced series for AR estimation
  int diffLength = length - 2;
  double diffSeries[diffLength], diffSeriesLag1[diffLength], diffSeriesLag2[diffLength];
  calculateLead(series, diffSeries, length, 2);
  double tempArray[length];
  calculateLead(series, tempArray, length, 1);
  calculateLag(tempArray, diffSeriesLag1, length, 1);
  calculateLag(series, diffSeriesLag2, length, 2);

  // Estimate AR parameters via bivariate regression on the differenced series
  double *ar2Estimates = performBivariateLinearRegression(diffSeriesLag1, diffSeriesLag2, diffSeries, diffLength);
  double ar2Beta1 = ar2Estimates[0],
         ar2Beta2 = ar2Estimates[1],
         ar2Intercept = ar2Estimates[2];
  free(ar2Estimates);

  // Compute AR predictions and errors on the differenced series
  double diffSeriesPred[diffLength];
  predictBivariate(diffSeriesLag1, diffSeriesLag2, diffSeriesPred, ar2Beta1, ar2Beta2, ar2Intercept, diffLength);
  double diffError[diffLength];
  calculateArrayDifference(diffSeries, diffSeriesPred, diffError, diffLength);

  // Prepare arrays for MA estimation: shift diffSeries to create "arComponent" and error lag
  int maLength = diffLength - 1;
  double arComponent[maLength], errorLag[maLength];
  calculateLead(diffSeries, arComponent, diffLength, 1);
  calculateLag(diffError, errorLag, diffLength, 1);

  // Now use adaptive MA(1) estimation:
  double estimatedTheta, estimatedC;
  estimateMA1Parameters(errorLag, arComponent, maLength, &estimatedTheta, &estimatedC);

  // Generate a simple recursive forecast (here we produce FORECAST_HORIZON amount of forecast values)
  double *forecast = malloc(sizeof(double) * FORECAST_ARRAY_SIZE);
  if (!forecast)
    exit(EXIT_FAILURE);
  // For example, use the last error value to produce a one‐step forecast:
  double lastError = errorLag[maLength - 1];
  forecast[0] = estimatedTheta * lastError + estimatedC;
  // Here we simply replicate the forecast recursively (adjust as needed)
  for (int i = 1; i < FORECAST_HORIZON; i++)
  {
    forecast[i] = forecast[i - 1];
  }
  // Compute and store a placeholder MAPE in forecast[FORECAST_HORIZON]
  double mapeValue = calculateMAPE(arComponent, diffSeriesPred, maLength);
  forecast[FORECAST_HORIZON] = mapeValue;
  return forecast;
}

/**
 * @brief Forecasts future values using an AR(1)–MA(2) hybrid model.
 *
 * @param series The input time series.
 * @param seriesLength The number of observations in the series.
 * @return A pointer to a dynamically allocated forecast array (first FORECAST_HORIZON forecasts; element FORECAST_HORIZON holds MAPE).
 *
 * This function performs the following steps:
 *   1. Constructs a differenced series for the AR component.
 *   2. Builds a design matrix from several lagged versions of the differenced series and estimates AR parameters.
 *   3. Computes AR prediction errors.
 *   4. Prepares a design matrix for MA estimation.
 *   5. Iteratively refines MA parameters.
 *   6. Generates recursive forecasts using the hybrid model.
 */
double *forecastAR1MA2(double series[], int seriesLength)
{
  // Step 1: Build differenced series for AR estimation.
  int arDataLength = seriesLength - 3;
  double diffSeries[arDataLength], lag1Diff[arDataLength], lag2Diff[arDataLength], lag3Diff[arDataLength];
  calculateLeadWithLag(series, diffSeries, seriesLength, 0, 3);
  calculateLeadWithLag(series, lag1Diff, seriesLength, 1, 2);
  calculateLeadWithLag(series, lag2Diff, seriesLength, 2, 1);
  calculateLeadWithLag(series, lag3Diff, seriesLength, 3, 0);

  // Step 2: Estimate AR parameters using multivariate regression.
  int numARPredictors = 3;
  double arDesign[arDataLength][3];
  double arResponse[arDataLength][1];
  arrayToMatrix(arDataLength, diffSeries, arResponse);
  columnBind3(arDataLength, lag1Diff, lag2Diff, lag3Diff, arDesign);
  double *arEstimates = performMultivariateLinearRegression(arDataLength, numARPredictors, arDesign, arResponse);
  double phi1 = arEstimates[0],
         phi2 = arEstimates[1],
         phi3 = arEstimates[2],
         AR_intercept = arEstimates[3];
  free(arEstimates);

  // Step 3: Compute AR prediction errors.
  double arPredicted[arDataLength][1];
  for (int i = 0; i < arDataLength; i++)
  {
    arPredicted[i][0] = arDesign[i][0] * phi1 + arDesign[i][1] * phi2 + arDesign[i][2] * phi3 + AR_intercept;
  }
  double arError[arDataLength];
  for (int i = 0; i < arDataLength; i++)
  {
    arError[i] = arResponse[i][0] - arPredicted[i][0];
  }

  // Step 4: Prepare data for MA estimation.
  int maDataLength = arDataLength - 2;
  double targetSeries[maDataLength], targetLag[maDataLength];
  double errorLag1[maDataLength], errorLag2[maDataLength];
  calculateLeadWithLag(diffSeries, targetSeries, arDataLength, 0, 2);
  calculateLeadWithLag(diffSeries, targetLag, arDataLength, 1, 1);
  calculateLeadWithLag(arError, errorLag1, arDataLength, 1, 1);
  calculateLeadWithLag(arError, errorLag2, arDataLength, 2, 0);

  // Step 5: Instead of using regression, apply adaptive MA(2) estimation:
  double estimatedTheta1, estimatedTheta2, estimatedMAIntercept;
  estimateMA2Parameters(targetSeries, errorLag1, errorLag2, maDataLength,
                        &estimatedTheta1, &estimatedTheta2, &estimatedMAIntercept);

  // Step 6: Generate recursive forecasts.
  double forecastMAPE = calculateMAPE(targetSeries, targetSeries, maDataLength); // placeholder
  double *forecast = malloc(sizeof(double) * FORECAST_ARRAY_SIZE);
  if (!forecast)
    exit(EXIT_FAILURE);
  double lastValue = targetSeries[maDataLength - 1];
  for (int i = 0; i < FORECAST_HORIZON; i++)
  {
    forecast[i] = estimatedTheta1 * lastValue +
                  estimatedTheta2 * errorLag2[maDataLength - 1] +
                  estimatedMAIntercept;
    lastValue = forecast[i];
  }
  forecast[FORECAST_HORIZON] = forecastMAPE;
  return forecast;
}

/*========================================================================
  Forecasting Model: AR(2)-MA(1) and AR(2)-MA(2)
========================================================================*/

double *forecastAR2MA1(double series[], int seriesLength)
{
  int numPredictors = 2; // For AR(2)
  int arDataLength = seriesLength - 2;
  double diffSeries[arDataLength];
  calculateLead(series, diffSeries, seriesLength, 1);

  double arLag1[arDataLength - 1], arLag2[arDataLength - 1], arResponse[arDataLength - 1];
  calculateLag(diffSeries, arResponse, arDataLength, 1);
  calculateLead(diffSeries, arLag1, arDataLength, 1);
  calculateLead(diffSeries, arLag2, arDataLength, 2);

  double *arEstimates = performBivariateLinearRegression(arLag1, arLag2, arResponse, arDataLength - 1);
  double phi1 = arEstimates[0],
         phi2 = arEstimates[1],
         AR_intercept = arEstimates[2];
  free(arEstimates);

  double arPred[arDataLength - 1];
  for (int i = 0; i < arDataLength - 1; i++)
  {
    arPred[i] = arLag1[i] * phi1 + arLag2[i] * phi2 + AR_intercept;
  }
  double arError[arDataLength - 1];
  for (int i = 0; i < arDataLength - 1; i++)
  {
    arError[i] = arResponse[i] - arPred[i];
  }

  int maDataLength = (arDataLength - 1) - 1;
  double maTarget[maDataLength], maLag[maDataLength];
  calculateLead(arError, maTarget, arDataLength - 1, 0);
  calculateLag(arError, maLag, arDataLength - 1, 1);

  // Use adaptive MA(1) estimation:
  double estimatedTheta, estimatedMAIntercept;
  estimateMA1Parameters(maLag, maTarget, maDataLength, &estimatedTheta, &estimatedMAIntercept);

  double *forecast = malloc(sizeof(double) * FORECAST_ARRAY_SIZE);
  if (!forecast)
    exit(EXIT_FAILURE);
  double lastARValue = arResponse[arDataLength - 2];
  double lastError = arError[arDataLength - 2];
  for (int i = 0; i < FORECAST_HORIZON; i++)
  {
    forecast[i] = lastARValue * phi1 + lastARValue * phi2 +
                  estimatedMAIntercept + estimatedTheta * lastError;
    lastARValue = forecast[i];
  }
  forecast[FORECAST_HORIZON] = calculateMAPE(arResponse, arPred, arDataLength - 1);
  return forecast;
}

/*========================================================================
  Forecasting Model: AR(2)-MA(2) Hybrid
========================================================================*/

double *forecastAR2MA2(double series[], int seriesLength)
{
  int numPredictors = 2; // For AR(2)
  int arDataLength = seriesLength - 2;
  double diffSeries[arDataLength];
  calculateLead(series, diffSeries, seriesLength, 1);

  double arLag1[arDataLength - 1], arLag2[arDataLength - 1], arResponse[arDataLength - 1];
  calculateLag(diffSeries, arResponse, arDataLength, 1);
  calculateLead(diffSeries, arLag1, arDataLength, 1);
  calculateLead(diffSeries, arLag2, arDataLength, 2);

  double *arEstimates = performBivariateLinearRegression(arLag1, arLag2, arResponse, arDataLength - 1);
  double phi1 = arEstimates[0],
         phi2 = arEstimates[1],
         AR_intercept = arEstimates[2];
  free(arEstimates);

  double arPred[arDataLength - 1];
  for (int i = 0; i < arDataLength - 1; i++)
  {
    arPred[i] = arLag1[i] * phi1 + arLag2[i] * phi2 + AR_intercept;
  }
  double arError[arDataLength - 1];
  for (int i = 0; i < arDataLength - 1; i++)
  {
    arError[i] = arResponse[i] - arPred[i];
  }

  int maDataLength = (arDataLength - 1) - 1;
  double maTarget[maDataLength], maLag1[maDataLength], maLag2[maDataLength];
  calculateLead(arError, maTarget, arDataLength - 1, 0);
  calculateLag(arError, maLag1, arDataLength - 1, 1);
  calculateLag(arError, maLag2, arDataLength - 1, 2);

  // Use adaptive MA(2) estimation:
  double estimatedTheta1, estimatedTheta2, estimatedMAIntercept;
  estimateMA2Parameters(maTarget, maLag1, maLag2, maDataLength,
                        &estimatedTheta1, &estimatedTheta2, &estimatedMAIntercept);

  double *forecast = malloc(sizeof(double) * FORECAST_ARRAY_SIZE);
  if (!forecast)
    exit(EXIT_FAILURE);
  double lastValue = maTarget[maDataLength - 1];
  for (int i = 0; i < FORECAST_HORIZON; i++)
  {
    forecast[i] = lastValue * phi1 + lastValue * phi2 +
                  estimatedMAIntercept + estimatedTheta1 * maLag1[maDataLength - 1] +
                  estimatedTheta2 * maLag2[maDataLength - 1];
    lastValue = forecast[i];
  }
  forecast[FORECAST_HORIZON] = calculateMAPE(maTarget, maTarget, maDataLength); // placeholder
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

/*========================================================================
  Main Function (for testing purposes)
========================================================================*/

int main(void)
{
  double sampleData[] = {10.544653, 10.688583, 10.666841, 10.662732, 10.535033, 10.612065, 10.577628, 10.524487, 10.511290, 10.520899, 10.605484, 10.506456, 10.693456, 10.667562, 10.640863, 10.553473, 10.684760, 10.752397, 10.671068, 10.667091, 10.641893, 10.625706, 10.701795, 10.607544, 10.689169, 10.695256, 10.717050, 10.677475, 10.691141, 10.730298, 10.732664, 10.710082, 10.713123, 10.759815, 10.696599, 10.663845, 10.716597, 10.780855, 10.795759, 10.802620, 10.720496, 10.753401, 10.709436, 10.746909, 10.737377, 10.754609, 10.765248, 10.692602, 10.837926, 10.755324, 10.756213, 10.843190, 10.862529, 10.751269, 10.902390, 10.817731, 10.859796, 10.887362, 10.835401, 10.824412, 10.860767, 10.819504, 10.907496, 10.831528, 10.821727, 10.830010, 10.915317, 10.858694, 10.921139, 10.927524, 10.894352, 10.889785, 10.956356, 10.938758, 11.093567, 10.844841, 11.094493, 11.035941, 10.982765, 11.071057, 10.996308, 11.099276, 11.142057, 11.137176, 11.157537, 11.007247, 11.144075, 11.183029, 11.172096, 11.164571, 11.192833, 11.227109, 11.141589, 11.311490, 11.239783, 11.295933, 11.199566, 11.232262, 11.333208, 11.337874, 11.322334, 11.288216, 11.280459, 11.247973, 11.288277, 11.415095, 11.297583, 11.360763, 11.288338, 11.434631, 11.456051, 11.578981, 11.419166, 11.478404, 11.660141, 11.544303, 11.652028, 11.638368, 11.651792, 11.621518, 11.763853, 11.760687, 11.771138, 11.678104, 11.783163, 11.932094, 11.948678, 11.962627, 11.937934, 12.077570, 11.981595, 12.096366, 12.032683, 12.094221, 11.979764, 12.217793, 12.235930, 12.129859, 12.411867, 12.396301, 12.413920, 12.445867, 12.480462, 12.470674, 12.537774, 12.562252, 12.810248, 12.733546, 12.861890, 12.918012, 13.033087, 13.245610, 13.184196, 13.414342, 13.611838, 13.626345, 13.715446, 13.851129, 14.113374, 14.588537, 14.653982, 15.250756, 15.618371, 16.459558, 18.144264, 23.523062, 40.229511, 38.351265, 38.085281, 37.500885, 37.153946, 36.893066, 36.705956, 36.559536, 35.938847, 36.391586, 36.194046, 36.391586, 36.119102, 35.560543, 35.599018, 34.958851, 35.393860, 34.904797, 35.401318, 34.863518, 34.046680, 34.508522, 34.043182, 34.704235, 33.556644, 33.888481, 33.533638, 33.452129, 32.930935, 32.669731, 32.772537, 32.805634, 32.246761, 32.075809, 31.864927, 31.878294, 32.241131, 31.965626, 31.553604, 30.843288, 30.784569, 31.436094, 31.170496, 30.552132, 30.500242, 30.167421, 29.911989, 29.586046, 29.478958, 29.718994, 29.611095, 29.557945, 28.463432, 29.341291, 28.821512, 28.447210, 27.861872, 27.855633, 27.910660, 28.425800, 27.715517, 27.617193, 27.093372, 26.968832, 26.977205, 27.170172, 26.251677, 26.633236, 26.224941, 25.874708, 25.593761, 26.392395, 24.904768, 25.331600, 24.530737, 25.074808, 25.310865, 24.337013, 24.442986, 24.500193, 24.130409, 24.062714, 24.064592, 23.533037, 23.977909, 22.924667, 22.806379, 23.130791, 22.527645, 22.570505, 22.932512, 22.486126, 22.594856, 22.383926, 22.115181, 22.105082, 21.151754, 21.074114, 21.240192, 20.977468, 20.771507, 21.184586, 20.495111, 20.650751, 20.656075, 20.433039, 20.005697, 20.216360, 19.982117, 19.703951, 19.572884, 19.332155, 19.544645, 18.666328, 19.219872, 18.934229, 19.186989, 18.694986, 18.096903, 18.298306, 17.704309, 18.023785, 18.224157, 18.182484, 17.642824, 17.739542, 17.474176, 17.270575, 17.604120, 17.631210, 16.639175, 17.107626, 17.024216, 16.852285, 16.780111, 16.838861, 16.539309, 16.092861, 16.131529, 16.221350, 16.087164, 15.821659, 15.695448, 15.693087, 16.047991, 15.682863, 15.724131, 15.263708, 15.638486, 15.443835, 15.602257, 15.122874, 14.918172, 14.968882, 14.843689, 14.861169, 15.052527, 15.056897, 14.690192, 14.686479, 14.567565, 14.365212, 14.253309, 14.289158, 14.227124, 14.069589, 14.074703, 13.869432, 13.861959, 13.782178, 13.882711, 13.908362, 13.727641, 13.600214, 13.594969, 13.535290, 13.602018, 13.502626, 13.579159, 13.207825, 13.426789, 13.178141, 13.286413, 12.958746, 13.189507, 13.079733, 13.138372, 12.986096, 12.854589, 12.858962, 12.903029, 12.852099, 12.644394, 12.558786, 12.636994};

  int dataLength = sizeof(sampleData) / sizeof(sampleData[0]);

  // Validate input length.
  assert(dataLength > 3 && "Series length must be greater than 3 for AR estimation.");

  double meanValue = calculateMean(sampleData, dataLength);
  printf("Mean = %lf\n", meanValue);

  // Univariate regression test.
  double predictor[] = {1, 2, 3, 4, 5};
  double response[] = {2, 4, 6, 8, 10};
  double *lr1Estimates = performUnivariateLinearRegression(predictor, response, 5);
  printf("Univariate Regression: Slope = %lf, Intercept = %lf\n", lr1Estimates[0], lr1Estimates[1]);
  free(lr1Estimates);

  // Forecast using AR(1)
  double *ar1Forecast = forecastAR1(sampleData, 183);
  printf("AR(1) Forecast: ");
  for (int i = 0; i < 17; i++)
  {
    printf("%lf ", ar1Forecast[i]);
  }
  printf("\n");
  free(ar1Forecast);

  /* --- Compute AR(1) residuals and run Ljung–Box test --- */
  int newLength = dataLength - 1;
  double predictions[newLength];
  double *regEst = performUnivariateLinearRegression(sampleData, sampleData + 1, newLength);
  double phi = regEst[0];
  double intercept = regEst[1];
  predictUnivariate(sampleData, predictions, phi, intercept, newLength);
  double residuals[newLength];
  calculateArrayDifference(sampleData + 1, predictions, residuals, newLength);
  free(regEst);
  // For AR(1), one parameter is estimated; test using 10 lags.
  checkResidualDiagnostics(residuals, newLength, 10, 1);

  // Forecast using AR(1)-MA(2) hybrid model.
  double *ar1ma2Forecast = forecastAR1MA1(sampleData, 183);
  printf("AR(1)-MA(1) Forecast: ");
  for (int i = 0; i < 17; i++)
  {
    printf("%lf ", ar1ma2Forecast[i]);
  }
  printf("\n");
  free(ar1ma2Forecast);

  // Additional forecasting models (e.g., AR(2)-MA(1), AR(2)-MA(2)) can be tested here.

  return 0;
}
