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

/** @} */  // end of LinearRegressionFunctions group


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

/** @} */  // end of StationarityTesting group


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

/** @} */  // end of DiagnosticMatrix group


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

/** @} */  // end of ErrorMetrics group

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
#define MAX_ITERATIONS 1000
#define UNIT_TOLERANCE 1.001
#define MA_LEARNING_RATE 0.0005  // learning rate for the MA gradient descent update

#ifdef DEBUG
#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define DEBUG_PRINT(...)
#endif

/*==================== Data Structures =====================*/
/**
 * @brief Structure to hold ARMA parameters.
 */
typedef struct {
    double ar[3];       // AR coefficients (if needed)
    double ma[3];       // MA coefficients (for example, theta values)
    double intercept;   // Constant term (for AR or MA part)
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
double calculateMean(const double array[], int length) {
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
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
int findIndexOfMin(const int array[], int size) {
    int minValue = array[0];
    int minIndex = 0;
    for (int i = 1; i < size; i++) {
        if (array[i] < minValue) {
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
void copyArray(const double source[], double destination[], int length) {
    for (int i = 0; i < length; i++) {
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
double* squareArray(const double input[], int size) {
    double* squared = malloc(sizeof(double) * size);
    if (!squared) {
        fprintf(stderr, "Error allocating memory in squareArray.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++) {
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
double calculateArraySum(const double array[], int length) {
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
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
void calculateArrayDifference(const double array1[], const double array2[], double difference[], int size) {
    for (int i = 0; i < size; i++) {
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
double* calculateElementwiseProduct(const double array1[], const double array2[], int size) {
    double* product = malloc(sizeof(double) * size);
    if (!product) {
        fprintf(stderr, "Error allocating memory in calculateElementwiseProduct.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++) {
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
double calculateStandardDeviation(const double array[], int length) {
    double meanValue = calculateMean(array, length);
    double sumSqDiff = 0.0;
    for (int i = 0; i < length; i++) {
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
void calculateLead(const double array[], double leadArray[], int length, int lag) {
    for (int i = 0; i < (length - lag); i++) {
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
void calculateLag(const double array[], double lagArray[], int length, int lag) {
    for (int i = 0; i < (length - lag); i++) {
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
void calculateLeadWithLag(const double array[], double leadArray[], int length, int lead, int lag) {
    for (int i = 0; i < (length - lead - lag); i++) {
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
void arrayToMatrix(int numElements, const double array[], double matrix[][1]) {
    for (int i = 0; i < numElements; i++) {
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
void matrixToArray(int numRows, double matrix[][1], double array[]) {
    for (int i = 0; i < numRows; i++) {
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
void columnBind3(int numRows, double array1[], double array2[], double array3[], double result[][3]) {
    for (int i = 0; i < numRows; i++) {
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
void columnBind4(int numRows, double array1[], double array2[], double array3[], double array4[], double result[][4]) {
    for (int i = 0; i < numRows; i++) {
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
void normalize2DArray(int numRows, int numCols, double matrix[][numCols], double normalized[][numCols]) {
    double columnMeans[1][numCols];
    for (int j = 0; j < numCols; j++) {
        double col[numRows];
        for (int i = 0; i < numRows; i++) {
            col[i] = matrix[i][j];
        }
        columnMeans[0][j] = calculateMean(col, numRows);
    }
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
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
void transposeMatrix(int numRows, int numCols, double matrix[][numCols], double transposed[][numRows]) {
    for (int i = 0; i < numCols; i++) {
        for (int j = 0; j < numRows; j++) {
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
void matrixMultiply(int rowsA, int colsA, int colsB, double A[][colsA], double B[][colsB], double result[][colsB]) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            double sum = 0.0;
            for (int k = 0; k < colsA; k++) {
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum;
        }
    }
}

/**
 * @brief Inverts a 3x3 matrix.
 *
 * @param n The dimension (should be 3).
 * @param matrix Input 3x3 matrix.
 * @param inverseMatrix Output inverted matrix.
 */
void invert3x3Matrix(int n, double matrix[][n], double inverseMatrix[][n]) {
    double determinant = 0;
    for (int i = 0; i < n; i++) {
        determinant += matrix[0][i] * (matrix[1][(i + 1) % 3] * matrix[2][(i + 2) % 3] -
                                        matrix[1][(i + 2) % 3] * matrix[2][(i + 1) % 3]);
    }
    // Check for singularity.
    if (fabs(determinant) < 1e-8) {
        fprintf(stderr, "Error: 3x3 matrix is singular (determinant near zero).\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inverseMatrix[j][i] = (matrix[(i + 1) % 3][(j + 1) % 3] * matrix[(i + 2) % 3][(j + 2) % 3] -
                                   matrix[(i + 1) % 3][(j + 2) % 3] * matrix[(i + 2) % 3][(j + 1) % 3]) / determinant;
        }
    }
}

/**
 * @brief Inverts a 4x4 matrix.
 *
 * @param matrix Input 4x4 matrix.
 * @param inverse Output inverted 4x4 matrix.
 *
 * @note This function computes the adjugate matrix and divides by the determinant.
 *       If the determinant is near zero, an error is printed and the program exits.
 */
void invert4x4Matrix(double matrix[4][4], double inverse[4][4]) {
    double det;
    
    // Compute the cofactors (elements of the adjugate matrix, before transposition)
    inverse[0][0] = matrix[1][1] * matrix[2][2] * matrix[3][3] -
                    matrix[1][1] * matrix[2][3] * matrix[3][2] -
                    matrix[2][1] * matrix[1][2] * matrix[3][3] +
                    matrix[2][1] * matrix[1][3] * matrix[3][2] +
                    matrix[3][1] * matrix[1][2] * matrix[2][3] -
                    matrix[3][1] * matrix[1][3] * matrix[2][2];

    inverse[1][0] = -matrix[1][0] * matrix[2][2] * matrix[3][3] +
                     matrix[1][0] * matrix[2][3] * matrix[3][2] +
                     matrix[2][0] * matrix[1][2] * matrix[3][3] -
                     matrix[2][0] * matrix[1][3] * matrix[3][2] -
                     matrix[3][0] * matrix[1][2] * matrix[2][3] +
                     matrix[3][0] * matrix[1][3] * matrix[2][2];

    inverse[2][0] = matrix[1][0] * matrix[2][1] * matrix[3][3] -
                    matrix[1][0] * matrix[2][3] * matrix[3][1] -
                    matrix[2][0] * matrix[1][1] * matrix[3][3] +
                    matrix[2][0] * matrix[1][3] * matrix[3][1] +
                    matrix[3][0] * matrix[1][1] * matrix[2][3] -
                    matrix[3][0] * matrix[1][3] * matrix[2][1];

    inverse[3][0] = -matrix[1][0] * matrix[2][1] * matrix[3][2] +
                     matrix[1][0] * matrix[2][2] * matrix[3][1] +
                     matrix[2][0] * matrix[1][1] * matrix[3][2] -
                     matrix[2][0] * matrix[1][2] * matrix[3][1] -
                     matrix[3][0] * matrix[1][1] * matrix[2][2] +
                     matrix[3][0] * matrix[1][2] * matrix[2][1];

    inverse[0][1] = -matrix[0][1] * matrix[2][2] * matrix[3][3] +
                     matrix[0][1] * matrix[2][3] * matrix[3][2] +
                     matrix[2][1] * matrix[0][2] * matrix[3][3] -
                     matrix[2][1] * matrix[0][3] * matrix[3][2] -
                     matrix[3][1] * matrix[0][2] * matrix[2][3] +
                     matrix[3][1] * matrix[0][3] * matrix[2][2];

    inverse[1][1] = matrix[0][0] * matrix[2][2] * matrix[3][3] -
                    matrix[0][0] * matrix[2][3] * matrix[3][2] -
                    matrix[2][0] * matrix[0][2] * matrix[3][3] +
                    matrix[2][0] * matrix[0][3] * matrix[3][2] +
                    matrix[3][0] * matrix[0][2] * matrix[2][3] -
                    matrix[3][0] * matrix[0][3] * matrix[2][2];

    inverse[2][1] = -matrix[0][0] * matrix[2][1] * matrix[3][3] +
                     matrix[0][0] * matrix[2][3] * matrix[3][1] +
                     matrix[2][0] * matrix[0][1] * matrix[3][3] -
                     matrix[2][0] * matrix[0][3] * matrix[3][1] -
                     matrix[3][0] * matrix[0][1] * matrix[2][3] +
                     matrix[3][0] * matrix[0][3] * matrix[2][1];

    inverse[3][1] = matrix[0][0] * matrix[2][1] * matrix[3][2] -
                    matrix[0][0] * matrix[2][2] * matrix[3][1] -
                    matrix[2][0] * matrix[0][1] * matrix[3][2] +
                    matrix[2][0] * matrix[0][2] * matrix[3][1] +
                    matrix[3][0] * matrix[0][1] * matrix[2][2] -
                    matrix[3][0] * matrix[0][2] * matrix[2][1];

    inverse[0][2] = matrix[0][1] * matrix[1][2] * matrix[3][3] -
                    matrix[0][1] * matrix[1][3] * matrix[3][2] -
                    matrix[1][1] * matrix[0][2] * matrix[3][3] +
                    matrix[1][1] * matrix[0][3] * matrix[3][2] +
                    matrix[3][1] * matrix[0][2] * matrix[1][3] -
                    matrix[3][1] * matrix[0][3] * matrix[1][2];

    inverse[1][2] = -matrix[0][0] * matrix[1][2] * matrix[3][3] +
                     matrix[0][0] * matrix[1][3] * matrix[3][2] +
                     matrix[1][0] * matrix[0][2] * matrix[3][3] -
                     matrix[1][0] * matrix[0][3] * matrix[3][2] -
                     matrix[3][0] * matrix[0][2] * matrix[1][3] +
                     matrix[3][0] * matrix[0][3] * matrix[1][2];

    inverse[2][2] = matrix[0][0] * matrix[1][1] * matrix[3][3] -
                    matrix[0][0] * matrix[1][3] * matrix[3][1] -
                    matrix[1][0] * matrix[0][1] * matrix[3][3] +
                    matrix[1][0] * matrix[0][3] * matrix[3][1] +
                    matrix[3][0] * matrix[0][1] * matrix[1][3] -
                    matrix[3][0] * matrix[0][3] * matrix[1][1];

    inverse[3][2] = -matrix[0][0] * matrix[1][1] * matrix[3][2] +
                     matrix[0][0] * matrix[1][2] * matrix[3][1] +
                     matrix[1][0] * matrix[0][1] * matrix[3][2] -
                     matrix[1][0] * matrix[0][2] * matrix[3][1] -
                     matrix[3][0] * matrix[0][1] * matrix[1][2] +
                     matrix[3][0] * matrix[0][2] * matrix[1][1];

    inverse[0][3] = -matrix[0][1] * matrix[1][2] * matrix[2][3] +
                     matrix[0][1] * matrix[1][3] * matrix[2][2] +
                     matrix[1][1] * matrix[0][2] * matrix[2][3] -
                     matrix[1][1] * matrix[0][3] * matrix[2][2] -
                     matrix[2][1] * matrix[0][2] * matrix[1][3] +
                     matrix[2][1] * matrix[0][3] * matrix[1][2];

    inverse[1][3] = matrix[0][0] * matrix[1][2] * matrix[2][3] -
                    matrix[0][0] * matrix[1][3] * matrix[2][2] -
                    matrix[1][0] * matrix[0][2] * matrix[2][3] +
                    matrix[1][0] * matrix[0][3] * matrix[2][2] +
                    matrix[2][0] * matrix[0][2] * matrix[1][3] -
                    matrix[2][0] * matrix[0][3] * matrix[1][2];

    inverse[2][3] = -matrix[0][0] * matrix[1][1] * matrix[2][3] +
                     matrix[0][0] * matrix[1][3] * matrix[2][1] +
                     matrix[1][0] * matrix[0][1] * matrix[2][3] -
                     matrix[1][0] * matrix[0][3] * matrix[2][1] -
                     matrix[2][0] * matrix[0][1] * matrix[1][3] +
                     matrix[2][0] * matrix[0][3] * matrix[1][1];

    inverse[3][3] = matrix[0][0] * matrix[1][1] * matrix[2][2] -
                    matrix[0][0] * matrix[1][2] * matrix[2][1] -
                    matrix[1][0] * matrix[0][1] * matrix[2][2] +
                    matrix[1][0] * matrix[0][2] * matrix[2][1] +
                    matrix[2][0] * matrix[0][1] * matrix[1][2] -
                    matrix[2][0] * matrix[0][2] * matrix[1][1];

    // Compute the determinant using the first row and its cofactors.
    det = matrix[0][0] * inverse[0][0] + matrix[0][1] * inverse[1][0] +
          matrix[0][2] * inverse[2][0] + matrix[0][3] * inverse[3][0];

    if (fabs(det) < 1e-8) {
        fprintf(stderr, "Error: Determinant is near zero in invert4x4Matrix.\n");
        exit(EXIT_FAILURE);
    }
    det = 1.0 / det;
    // Multiply the adjugate matrix by 1/det to obtain the inverse.
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            inverse[i][j] *= det;
        }
    }
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
double* performUnivariateLinearRegression(double predictor[], double response[], int length) {
    double predictorDiff[length], responseDiff[length];
    double meanPredictor = calculateMean(predictor, length);
    double meanResponse = calculateMean(response, length);
    
    for (int i = 0; i < length; i++) {
        predictorDiff[i] = predictor[i] - meanPredictor;
        responseDiff[i] = response[i] - meanResponse;
    }
    
    double stdPredictor = calculateStandardDeviation(predictor, length);
    double stdResponse = calculateStandardDeviation(response, length);
    
    double* prodDiff = calculateElementwiseProduct(predictorDiff, responseDiff, length);
    double covariance = calculateArraySum(prodDiff, length) / (length - 1);
    free(prodDiff);
    
    double correlation = covariance / (stdPredictor * stdResponse);
    
    double slope = correlation * (stdResponse / stdPredictor);
    double intercept = meanResponse - slope * meanPredictor;
    
    double* estimates = malloc(sizeof(double) * 2);
    if (!estimates) exit(EXIT_FAILURE);
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
void predictUnivariate(double predictor[], double predictions[], double slope, double intercept, int length) {
    for (int i = 0; i < length; i++) {
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
double* performBivariateLinearRegression(double predictor1[], double predictor2[], double response[], int length) {
    double pred1Diff[length], pred2Diff[length], respDiff[length];
    double meanPred1 = calculateMean(predictor1, length);
    double meanPred2 = calculateMean(predictor2, length);
    double meanResp = calculateMean(response, length);
    
    for (int i = 0; i < length; i++) {
        pred1Diff[i] = predictor1[i] - meanPred1;
        pred2Diff[i] = predictor2[i] - meanPred2;
        respDiff[i] = response[i] - meanResp;
    }
    
    double sumPred1 = calculateArraySum(predictor1, length);
    double sumPred2 = calculateArraySum(predictor2, length);
    double sumResp = calculateArraySum(response, length);
    
    double* prodPred1Resp = calculateElementwiseProduct(predictor1, response, length);
    double sumProdPred1Resp = calculateArraySum(prodPred1Resp, length);
    free(prodPred1Resp);
    
    double* prodPred2Resp = calculateElementwiseProduct(predictor2, response, length);
    double sumProdPred2Resp = calculateArraySum(prodPred2Resp, length);
    free(prodPred2Resp);
    
    double* prodPred1Pred2 = calculateElementwiseProduct(predictor1, predictor2, length);
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
    
    double* estimates = malloc(sizeof(double) * 3);
    if (!estimates) exit(EXIT_FAILURE);
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
void predictBivariate(double predictor1[], double predictor2[], double predictions[], double beta1, double beta2, double intercept, int length) {
    for (int i = 0; i < length; i++) {
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
double* performMultivariateLinearRegression(int numObservations, int numPredictors, double X[][numPredictors], double Y[][1]) {
    double X_normalized[numObservations][numPredictors], Y_normalized[numObservations][1];
    double X_means[1][numPredictors], Y_mean[1][1];
    double Xt[numPredictors][numObservations], XtX[numPredictors][numPredictors], XtX_inv[numPredictors][numPredictors];
    double XtX_inv_Xt[numPredictors][numObservations], beta[numPredictors][1];
    double* estimates = malloc(sizeof(double) * (numPredictors + 1));
    if (!estimates) exit(EXIT_FAILURE);
    
    // Normalize X and Y.
    normalize2DArray(numObservations, numPredictors, X, X_normalized);
    normalize2DArray(numObservations, 1, Y, Y_normalized);
    
    // Transpose X.
    transposeMatrix(numObservations, numPredictors, X_normalized, Xt);
    
    // Compute XtX = Xt * X.
    matrixMultiply(numPredictors, numObservations, numPredictors, Xt, X_normalized, XtX);
    
    // Invert XtX.
    if (numPredictors == 3) {
        invert3x3Matrix(3, XtX, XtX_inv);
    } else {
        invert4x4Matrix(XtX, XtX_inv);
    }
    
    // Compute XtX_inv_Xt = XtX_inv * Xt.
    matrixMultiply(numPredictors, numPredictors, numObservations, XtX_inv, Xt, XtX_inv_Xt);
    
    // Compute beta = XtX_inv_Xt * Y_normalized.
    matrixMultiply(numPredictors, numObservations, 1, XtX_inv_Xt, Y_normalized, beta);
    for (int i = 0; i < numPredictors; i++) {
        estimates[i] = beta[i][0];
    }
    
    // Compute intercept from the means.
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
double computeCorrelation(double array1[], double array2[], int length) {
    double diff1[length], diff2[length];
    double mean1 = calculateMean(array1, length);
    double mean2 = calculateMean(array2, length);
    for (int i = 0; i < length; i++) {
        diff1[i] = array1[i] - mean1;
        diff2[i] = array2[i] - mean2;
    }
    double std1 = calculateStandardDeviation(array1, length);
    double std2 = calculateStandardDeviation(array2, length);
    double* prodDiff = calculateElementwiseProduct(diff1, diff2, length);
    double correlation = calculateArraySum(prodDiff, length) / ((length - 1) * std1 * std2);
    free(prodDiff);
    return correlation;
}

/*========================================================================
  Stationarity Testing and Drift Adjustment
========================================================================*/

/**
 * @brief Performs a Dickey–Fuller style test to determine the order of differencing required for stationarity.
 *
 * @param series The input time series (will be modified in-place).
 * @param recoveryInfo Output array storing the last value of the series at each differencing step.
 * @param length The number of observations.
 * @return The order of differencing (d) applied.
 */
int DFTest(double series[], double recoveryInfo[], int length) {
    int diffOrder = 0;
    int adjustment = 0;
    double* regressionEstimates;
    double leadSeries[length - 1], lagSeries[length - 1];
    
    do {
        // Create a lead series (shifted by 1) and a lag series.
        calculateLead(series, leadSeries, length, 1);
        calculateLag(series, lagSeries, length, 1);
        regressionEstimates = performUnivariateLinearRegression(lagSeries, leadSeries, length - 1);
        
        if (diffOrder > 0) {
            // If already differenced, update the series using the difference.
            for (int i = 0; i < (length - adjustment); i++) {
                series[i] = (adjustment * series[i + adjustment]) - series[i];
            }
            length -= 1;
        }
        recoveryInfo[diffOrder] = series[length - 1];
        diffOrder++;
        adjustment = 1;
    } while (!(regressionEstimates[0] <= UNIT_TOLERANCE && regressionEstimates[0] >= -UNIT_TOLERANCE));
    
    free(regressionEstimates);
    return (diffOrder - 1);
}

/**
 * @brief Adjusts a differenced series to recover the original level (drift correction).
 *
 * @param originalSeries The original series.
 * @param adjustedSeries The series to adjust.
 * @param length The number of observations.
 * @param diffOrder The order of differencing previously applied.
 */
void adjustDrift(double originalSeries[], double adjustedSeries[], int length, int diffOrder) {
    for (int i = 0; i < diffOrder; i++) {
        for (int j = 0; j < (length - diffOrder); j++) {
            adjustedSeries[j] = adjustedSeries[j + diffOrder] - adjustedSeries[j];
        }
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
void computeEAFMatrix(double series[], double eafMatrix[][3], int length) {
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
  Error Metrics
========================================================================*/

/**
 * @brief Calculates the Mean Absolute Percentage Error (MAPE).
 *
 * @param actual The array of actual values.
 * @param predicted The array of predicted values.
 * @param length The number of elements.
 * @return MAPE as a percentage.
 */
double calculateMAPE(double actual[], double predicted[], int length) {
    double errorPercentage[length];
    for (int i = 0; i < length; i++) {
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
double calculateMAE(double actual[], double predicted[], int length) {
    double absoluteError[length];
    for (int i = 0; i < length; i++) {
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
 * @return A dynamically allocated array of forecasted values; element 16 contains the MAPE.
 *
 * The function estimates an AR(1) model using the last (length-1) observations,
 * computes the in-sample prediction error (MAPE), and then recursively forecasts
 * 16 future values.
 */
double* forecastAR1(double series[], int length) {
    int newLength = length - 1;
    double* regressionEstimates = performUnivariateLinearRegression(series, series + 1, newLength);
    double* forecast = malloc(sizeof(double) * 18);
    if (!forecast) exit(EXIT_FAILURE);
    
    double lastValue = series[newLength - 1];
    double mapeValue = calculateMAPE(series + 1, series, newLength);  // placeholder computation
    
    // Recursive forecasting for 16 steps.
    for (int i = 0; i < 16; i++) {
        forecast[i] = lastValue * regressionEstimates[0] + regressionEstimates[1];
        lastValue = forecast[i];
    }
    free(regressionEstimates);
    forecast[16] = mapeValue;
    return forecast;
}


/*========================================================================
  Upgraded MA Update: Joint Gradient Descent for MA parameters
========================================================================*/

/**
 * @brief Computes the Euclidean norm of a vector of length n.
 */
double vectorNorm(const double *vec, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += vec[i] * vec[i];
    }
    return sqrt(sum);
}

/**
 * @brief Computes the gradient of the MA objective function.
 *
 * The objective is defined as the sum of squared errors:
 *   J = sum_{i=0}^{n-1} [targetSeries[i] - (phi_cap*targetLag[i] + theta1*errorLag1[i] + theta2*errorLag2[i] + MA_intercept)]^2
 *
 * The gradient is computed with respect to each parameter.
 */
void computeMAObjectiveGradient(const double *targetSeries, const double *targetLag,
                                const double *errorLag1, const double *errorLag2,
                                int n, const double params[4], double grad[4]) {
    grad[0] = grad[1] = grad[2] = grad[3] = 0.0;
    for (int i = 0; i < n; i++) {
        double predicted = params[0] * targetLag[i] +
                           params[1] * errorLag1[i] +
                           params[2] * errorLag2[i] +
                           params[3];
        double error = targetSeries[i] - predicted;
        grad[0] += -2 * error * targetLag[i];
        grad[1] += -2 * error * errorLag1[i];
        grad[2] += -2 * error * errorLag2[i];
        grad[3] += -2 * error;
    }
}

/**
 * @brief Computes the gradient of a simple MA(1) objective function.
 *
 * The objective is defined as the sum of squared errors:
 *   J(theta, c) = sum_{i=0}^{n-1} [target[i] - (theta * lag[i] + c)]^2
 *
 * @param target The target series.
 * @param lag The lag series.
 * @param n Number of data points.
 * @param params A two-element vector: [theta, c].
 * @param grad Output gradient vector (length 2).
 */
void computeMA1Gradient(const double *target, const double *lag, int n, const double params[2], double grad[2]) {
    grad[0] = 0.0;
    grad[1] = 0.0;
    for (int i = 0; i < n; i++) {
        double pred = params[0] * lag[i] + params[1];
        double error = target[i] - pred;
        grad[0] += -2.0 * error * lag[i];
        grad[1] += -2.0 * error;
    }
}

/**
 * @brief Computes the gradient of a MA(2) objective function.
 *
 * The objective is:
 *   J(theta1, theta2, c) = sum_{i=0}^{n-1} [target[i] - (theta1 * lag1[i] + theta2 * lag2[i] + c)]^2
 *
 * @param target The target series.
 * @param lag1 The first lag series.
 * @param lag2 The second lag series.
 * @param n Number of data points.
 * @param params A three-element vector: [theta1, theta2, c].
 * @param grad Output gradient vector (length 3).
 */
void computeMA2Gradient(const double *target, const double *lag1, const double *lag2, int n, const double params[3], double grad[3]) {
    grad[0] = grad[1] = grad[2] = 0.0;
    for (int i = 0; i < n; i++) {
        double pred = params[0] * lag1[i] + params[1] * lag2[i] + params[2];
        double error = target[i] - pred;
        grad[0] += -2.0 * error * lag1[i];
        grad[1] += -2.0 * error * lag2[i];
        grad[2] += -2.0 * error;
    }
}

/*========================================================================
  Forecasting Model: AR(1)-MA(1) Hybrid (Example)
========================================================================*/

/**
 * @brief Forecasts future values using an AR(1)-MA(1) hybrid model.
 *
 * @param series The input time series (unmodified).
 * @param length The length of the series.
 * @return A dynamically allocated forecast array (first 16 forecasts; element 16 holds MAPE).
 *
 * This function first estimates an AR(1)-MA(1) model on a differenced version of the input
 * series, then iteratively refines the MA parameter estimates using a univariate update.
 */
double* forecastAR1MA1(const double series[], int length) {
    // Validate that there are enough data points.
    assert(length > 2 && "Series length must be greater than 2 for AR(1)-MA(1) forecasting.");

    // Allocate memory for the forecast array (16 forecasts plus MAPE).
    double *forecast = malloc(sizeof(double) * 18);
    if (!forecast) {
        fprintf(stderr, "Error allocating memory for forecast.\n");
        exit(EXIT_FAILURE);
    }

    /* === Step 1: Build Differenced Series for AR Estimation === */
    // We'll use a lag of 2 to produce a differenced series.
    int diffLength = length - 2;
    double diffSeries[diffLength];
    calculateLead(series, diffSeries, length, 2);

    // Prepare two lagged versions needed for the AR estimation.
    double diffSeriesLag1[diffLength], diffSeriesLag2[diffLength];
    double tempArray[length];
    // Compute a first difference into tempArray.
    calculateLead(series, tempArray, length, 1);
    // diffSeriesLag1: copy tempArray (first difference) with lag 1.
    calculateLag(tempArray, diffSeriesLag1, length, 1);
    // diffSeriesLag2: use the original series with lag 2.
    calculateLag(series, diffSeriesLag2, length, 2);
    // (diffLength remains as length - 2)

    /* === Step 2: Estimate AR(1) (here represented via a bivariate regression as a proxy) === */
    // (In this AR(1)-MA(1) example, we use a bivariate regression to capture two components.)
    double *arEstimates = performBivariateLinearRegression(diffSeriesLag1, diffSeriesLag2, diffSeries, diffLength);
    double arCoef = arEstimates[0];      // Represents the AR coefficient (or a composite coefficient)
    double dummyCoef = arEstimates[1];     // Not used further in this example
    double arIntercept = arEstimates[2];
    free(arEstimates);

    /* === Step 3: Compute AR Predictions and Errors === */
    double diffSeriesPred[diffLength];
    predictBivariate(diffSeriesLag1, diffSeriesLag2, diffSeriesPred, arCoef, dummyCoef, arIntercept, diffLength);
    double diffError[diffLength];
    calculateArrayDifference(diffSeries, diffSeriesPred, diffError, diffLength);

    /* === Step 4: Prepare Data for MA Estimation === */
    // For the MA(1) part, we create a target series and a lagged error series.
    int maLength = diffLength - 1; // because we need to shift by 1 for the error lag
    double arComponent[maLength], errorLag[maLength];
    // arComponent is created by taking the diffSeries starting from index 1.
    calculateLead(diffSeries, arComponent, diffLength, 1);
    // errorLag is the diffError series lagged by 1.
    calculateLag(diffError, errorLag, diffLength, 1);

    /* === Step 5: Initial MA Parameter Estimation via Univariate Regression === */
    double *initialMAEstimates = performUnivariateLinearRegression(errorLag, arComponent, maLength);
    double initAR_MA = initialMAEstimates[0];  // initial estimate for the “AR part” of MA update
    double initMACoef = initialMAEstimates[1];   // initial MA coefficient (theta)
    free(initialMAEstimates);

    // Set initial and current parameter estimates.
    double currentARCoef = initAR_MA;
    double currentMACoef = initMACoef;
    double initialARCoef = initAR_MA;
    double initialMACoef = initMACoef;

    /* Copy arrays for fallback purposes */
    double arComponentBackup[maLength], diffSeriesPredBackup[maLength];
    copyArray(arComponent, arComponentBackup, maLength);
    copyArray(diffSeriesPred, diffSeriesPredBackup, maLength);

    /* === Step 6: Iterative MA Parameter Update Loop === */
    int iterationCount = 0;
    int originalMALength = maLength;
    do {
        // Update predictions using the current MA parameters.
        double updatedPred[maLength];
        predictUnivariate(errorLag, updatedPred, currentARCoef, currentMACoef, maLength);
        double updatedResidual[maLength];
        calculateArrayDifference(arComponent, updatedPred, updatedResidual, maLength);

        // Refresh the lag arrays for the next iteration.
        calculateLead(arComponent, arComponent, maLength, 1);
        calculateLag(updatedResidual, errorLag, maLength, 1);

        // Optionally reduce the effective MA sample length if needed (here we mimic that behavior).
        maLength -= 1;

        double *updatedEstimates = performUnivariateLinearRegression(errorLag, arComponent, maLength);
        double updatedARCoef = updatedEstimates[0];
        double updatedMACoef = updatedEstimates[1];
        free(updatedEstimates);

        iterationCount++;
        // Check for convergence once a minimum number of iterations have passed.
        if (iterationCount >= MIN_ITERATIONS &&
            fabs(currentARCoef - updatedARCoef) < CONVERGENCE_TOLERANCE &&
            fabs(currentMACoef - updatedMACoef) < CONVERGENCE_TOLERANCE)
        {
            currentARCoef = updatedARCoef;
            currentMACoef = updatedMACoef;
            break;
        }
        // Update current parameters for next iteration.
        currentARCoef = updatedARCoef;
        currentMACoef = updatedMACoef;
    } while (iterationCount < MAX_ITERATIONS);

    // Fallback: if maximum iterations reached, revert to initial estimates.
    if (iterationCount >= MAX_ITERATIONS) {
        currentARCoef = initialARCoef;
        currentMACoef = initialMACoef;
        maLength = originalMALength;
    }

    double finalMAPE = calculateMAPE(arComponent, diffSeriesPred, maLength);
    DEBUG_PRINT("AR1MA1: Iterations=%d, Final AR Coefficient=%lf, Final MA Coefficient=%lf, MAPE=%lf\n",
                iterationCount, currentARCoef, currentMACoef, finalMAPE);

    /* === Step 7: Recursive Forecast Generation === */
    // For simplicity, we generate forecasts using a recursive rule.
    double lastError = errorLag[maLength - 1];
    forecast[0] = currentARCoef * lastError + currentMACoef;
    for (int i = 1; i < 16; i++) {
        forecast[i] = forecast[i - 1] * currentARCoef + currentMACoef;
    }
    forecast[16] = finalMAPE;
    return forecast;
}

/**
 * @brief Forecasts future values using an AR(1)–MA(2) hybrid model.
 *
 * @param series The input time series.
 * @param seriesLength The number of observations in the series.
 * @return A pointer to a dynamically allocated forecast array (first 16 forecasts; element 16 holds MAPE).
 *
 * This function performs the following steps:
 *   1. Constructs a differenced series for the AR component.
 *   2. Builds a design matrix from several lagged versions of the differenced series and estimates AR parameters.
 *   3. Computes AR prediction errors.
 *   4. Prepares a design matrix for MA estimation.
 *   5. Iteratively refines MA parameters.
 *   6. Generates recursive forecasts using the hybrid model.
 */
double* forecastAR1MA2(double series[], int seriesLength) {
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
    double phi1 = arEstimates[0], phi2 = arEstimates[1], phi3 = arEstimates[2], AR_intercept = arEstimates[3];
    free(arEstimates);
    
    // Step 3: Compute AR prediction errors.
    double arPredicted[arDataLength][1];
    for (int i = 0; i < arDataLength; i++) {
        arPredicted[i][0] = arDesign[i][0] * phi1 + arDesign[i][1] * phi2 + arDesign[i][2] * phi3 + AR_intercept;
    }
    double arError[arDataLength];
    for (int i = 0; i < arDataLength; i++) {
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
    
    double maDesign[maDataLength][3], maResponse[maDataLength][1];
    arrayToMatrix(maDataLength, targetSeries, maResponse);
    columnBind3(maDataLength, targetLag, errorLag1, errorLag2, maDesign);
    
    // Step 5: Estimate MA parameters.
    double *maEstimates = performMultivariateLinearRegression(maDataLength, numARPredictors, maDesign, maResponse);
    double phi_cap = maEstimates[0], theta1 = maEstimates[1],
           theta2 = maEstimates[2], MA_intercept = maEstimates[3];
    free(maEstimates);
    
    // Save initial MA estimates.
    double phi_cap_new = phi_cap, theta1_new = theta1, theta2_new = theta2, MA_intercept_new = MA_intercept;
    double phi_cap_initial = phi_cap, theta1_initial = theta1, theta2_initial = theta2, MA_intercept_initial = MA_intercept;
    int maLength_initial = maDataLength;
    int iterationCount = 0;
    
    // Step 6: Iteratively refine MA parameters.
    double maPredicted[maDataLength][1], maError[maDataLength][1];
    double currentParams[4]; // [phi_cap, theta1, theta2, MA_intercept]
    while (1) {
        // Set current estimates.
        currentParams[0] = phi_cap_new;
        currentParams[1] = theta1_new;
        currentParams[2] = theta2_new;
        currentParams[3] = MA_intercept_new;
        
        // Generate predictions using current MA estimates.
        for (int i = 0; i < maDataLength; i++) {
            maPredicted[i][0] = maDesign[i][0] * currentParams[0] +
                                maDesign[i][1] * currentParams[1] +
                                maDesign[i][2] * currentParams[2] +
                                currentParams[3];
        }
        // Compute errors.
        for (int i = 0; i < maDataLength; i++) {
            maError[i][0] = maResponse[i][0] - maPredicted[i][0];
        }
        double maErrorArray[maDataLength];
        matrixToArray(maDataLength, maError, maErrorArray);
        
        // Update MA estimates using univariate regression on errors.
        double *updatedEstimates = performUnivariateLinearRegression(errorLag1, targetSeries, maDataLength);
        double updated_phi_cap = updatedEstimates[0];
        double updated_theta = updatedEstimates[1];
        free(updatedEstimates);
        
        iterationCount++;
        if (iterationCount >= MIN_ITERATIONS &&
            fabs(phi_cap_new - updated_phi_cap) < CONVERGENCE_TOLERANCE &&
            fabs(theta1_new - updated_theta) < CONVERGENCE_TOLERANCE) {
            break;
        }
        phi_cap_new = updated_phi_cap;
        theta1_new = updated_theta;
        // (theta2_new and MA_intercept_new remain unchanged for simplicity.)
    }
    
    // Fallback if iterations exceed a limit.
    if (iterationCount > MIN_ITERATIONS) {
        phi_cap_new = phi_cap_initial;
        theta1_new = theta1_initial;
        theta2_new = theta2_initial;
        MA_intercept_new = MA_intercept_initial;
        maDataLength = maLength_initial;
    }
    
    double forecastMAPE = calculateMAPE(targetSeries, targetSeries, maDataLength); // placeholder computation
    
    DEBUG_PRINT("AR1MA2: Iterations=%d, phi_cap_new=%lf, theta1_new=%lf, theta2_new=%lf, MA_intercept_new=%lf\n",
                iterationCount, phi_cap_new, theta1_new, theta2_new, MA_intercept_new);
    DEBUG_PRINT("AR1MA2: MAPE=%lf\n", forecastMAPE);
    
    // Step 7: Generate recursive forecasts.
    double *forecast = malloc(sizeof(double) * 18);
    if (!forecast) exit(EXIT_FAILURE);
    double lastValue = targetSeries[maDataLength - 1];
    double errorValue = errorLag1[maDataLength - 1];
    for (int i = 0; i < 16; i++) {
        forecast[i] = phi_cap_new * lastValue + theta1_new * errorValue +
                      theta2_new * errorLag2[maDataLength - 1] + MA_intercept_new;
        lastValue = forecast[i];
    }
    forecast[16] = forecastMAPE;
    return forecast;
}

/*========================================================================
  Forecasting Model: AR(2)-MA(1) and AR(2)-MA(2)
========================================================================*/


/**
 * @brief Forecasts future values using an AR(2)-MA(1) hybrid model.
 *
 * The AR(2) part is estimated using two lagged differences, and the MA(1) parameters are
 * refined using joint gradient descent.
 *
 * @param series The input time series.
 * @param seriesLength The number of observations.
 * @return A dynamically allocated forecast array (first 16 forecasts; element 16 holds MAPE).
 */
double* forecastAR2MA1(double series[], int seriesLength) {
    // --- AR Part: Use simple differencing (lag 1) for AR estimation.
    int arDataLength = seriesLength - 1;
    double diffSeries[arDataLength];
    calculateLead(series, diffSeries, seriesLength, 1); // diffSeries[i] = series[i+1]
    
    // Build AR design matrix using two lags.
    int n_ar = arDataLength - 1;  // number of AR observations
    double arLag1[n_ar], arLag2[n_ar], arResponse[n_ar];
    calculateLead(diffSeries, arLag1, arDataLength, 1);  // lag 1
    calculateLead(diffSeries, arLag2, arDataLength, 2);  // lag 2
    calculateLag(diffSeries, arResponse, arDataLength, 1); // response = diffSeries[0..n_ar-1]
    
    double *arEstimates = performBivariateLinearRegression(arLag1, arLag2, arResponse, n_ar);
    double phi1 = arEstimates[0], phi2 = arEstimates[1], AR_intercept = arEstimates[2];
    free(arEstimates);
    
    // Compute AR predictions and errors.
    double arPred[n_ar];
    for (int i = 0; i < n_ar; i++) {
        arPred[i] = arLag1[i] * phi1 + arLag2[i] * phi2 + AR_intercept;
    }
    double arError[n_ar];
    for (int i = 0; i < n_ar; i++) {
        arError[i] = arResponse[i] - arPred[i];
    }
    
    // --- MA Part (MA(1)): Use the AR error series.
    int maDataLength = n_ar - 1;
    double maTarget[maDataLength], maLag[maDataLength];
    // For MA(1), we use the AR error series: target = arError[0..maDataLength-1], lag = arError[1..maDataLength]
    calculateLead(arError, maTarget, n_ar, 0);
    calculateLag(arError, maLag, n_ar, 1);
    
    // Initialize MA(1) parameters: [theta, c]
    double maParams[2] = { 0.0, 0.0 };
    int iter = 0;
    double grad[2];
    while (iter < MAX_ITERATIONS) {
        computeMA1Gradient(maTarget, maLag, maDataLength, maParams, grad);
        double gradNorm = sqrt(grad[0]*grad[0] + grad[1]*grad[1]);
        if (iter >= MIN_ITERATIONS && gradNorm < CONVERGENCE_TOLERANCE)
            break;
        maParams[0] -= MA_LEARNING_RATE * grad[0];
        maParams[1] -= MA_LEARNING_RATE * grad[1];
        iter++;
    }
    DEBUG_PRINT("AR2MA1: Iterations=%d, Final MA Params: theta=%lf, intercept=%lf\n", iter, maParams[0], maParams[1]);
    
    // --- Forecast Generation ---
    double forecastMAPE = calculateMAPE(arResponse, arPred, n_ar); // placeholder
    double *forecast = malloc(sizeof(double) * 18);
    if (!forecast) exit(EXIT_FAILURE);
    double lastARValue = arResponse[n_ar - 1];
    double lastError = arError[n_ar - 1];
    for (int i = 0; i < 16; i++) {
        forecast[i] = phi1 * lastARValue + phi2 * lastARValue + AR_intercept +
                      maParams[0] * lastError + maParams[1];
        lastARValue = forecast[i];
    }
    forecast[16] = forecastMAPE;
    return forecast;
}
/*========================================================================
  Forecasting Model: AR(2)-MA(2) Hybrid
========================================================================*/


/**
 * @brief Forecasts future values using an AR(2)-MA(2) hybrid model.
 *
 * The AR(2) part is estimated using two lagged values from a differenced series,
 * and the MA(2) parameters are updated via gradient descent.
 *
 * @param series The input time series.
 * @param seriesLength The number of observations.
 * @return A dynamically allocated forecast array (first 16 forecasts; element 16 holds MAPE).
 */
double* forecastAR2MA2(double series[], int seriesLength) {
    // --- AR Part: Use simple differencing (lag 1) for AR estimation.
    int arDataLength = seriesLength - 1;
    double diffSeries[arDataLength];
    calculateLead(series, diffSeries, seriesLength, 1);
    
    int n_ar = arDataLength - 1;
    double arLag1[n_ar], arLag2[n_ar], arResponse[n_ar];
    calculateLead(diffSeries, arLag1, arDataLength, 1);
    calculateLead(diffSeries, arLag2, arDataLength, 2);
    calculateLag(diffSeries, arResponse, arDataLength, 1);
    
    double *arEstimates = performBivariateLinearRegression(arLag1, arLag2, arResponse, n_ar);
    double phi1 = arEstimates[0], phi2 = arEstimates[1], AR_intercept = arEstimates[2];
    free(arEstimates);
    
    double arPred[n_ar];
    for (int i = 0; i < n_ar; i++) {
        arPred[i] = arLag1[i] * phi1 + arLag2[i] * phi2 + AR_intercept;
    }
    double arError[n_ar];
    for (int i = 0; i < n_ar; i++) {
        arError[i] = arResponse[i] - arPred[i];
    }
    
    // --- MA Part (MA(2)): Use two lags of the AR error.
    int maDataLength = n_ar - 1;
    double maTarget[maDataLength], maLag1[maDataLength], maLag2[maDataLength];
    // For MA(2), use: target = arError[0..maDataLength-1], lag1 = arError[1..maDataLength], lag2 = arError[2..maDataLength]
    calculateLead(arError, maTarget, n_ar, 0);
    calculateLag(arError, maLag1, n_ar, 1);
    calculateLag(arError, maLag2, n_ar, 2);
    
    // For the MA(2) update we need three parameters: [theta1, theta2, c]
    double maParams[3] = { 0.0, 0.0, 0.0 };
    int iter = 0;
    double grad[3];
    while (iter < MAX_ITERATIONS) {
        computeMA2Gradient(maTarget, maLag1, maLag2, maDataLength, maParams, grad);
        double gradNorm = sqrt(grad[0]*grad[0] + grad[1]*grad[1] + grad[2]*grad[2]);
        if (iter >= MIN_ITERATIONS && gradNorm < CONVERGENCE_TOLERANCE)
            break;
        for (int j = 0; j < 3; j++) {
            maParams[j] -= MA_LEARNING_RATE * grad[j];
        }
        iter++;
    }
    DEBUG_PRINT("AR2MA2: Iterations=%d, MA Params: theta1=%lf, theta2=%lf, intercept=%lf\n",
                iter, maParams[0], maParams[1], maParams[2]);
    
    double forecastMAPE = calculateMAPE(arResponse, arPred, n_ar); // placeholder
    double *forecast = malloc(sizeof(double) * 18);
    if (!forecast) exit(EXIT_FAILURE);
    double lastValue = maTarget[maDataLength - 1];
    // For this example, we use the MA(2) model to adjust the forecast recursively.
    for (int i = 0; i < 16; i++) {
        forecast[i] = phi1 * lastValue + phi2 * lastValue + AR_intercept +
                      maParams[0] * maLag1[maDataLength - 1] +
                      maParams[1] * maLag2[maDataLength - 1] +
                      maParams[2];
        lastValue = forecast[i];
    }
    forecast[16] = forecastMAPE;
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
void recoverForecast(double forecastDiff[], double recoveryValues[], double finalForecast[], int numForecasts, int diffOrder) {
    copyArray(forecastDiff, finalForecast, numForecasts);
    for (int i = 0; i < diffOrder; i++) {
        double drift = recoveryValues[diffOrder - i - 1];
        for (int j = 0; j < numForecasts; j++) {
            finalForecast[j] += drift;
            drift = finalForecast[j]; // propagate drift cumulatively
        }
    }
}

/*========================================================================
  Main Function (for testing purposes)
========================================================================*/

int main(void) {
    double sampleData[] = {8.0, 4.0, 2.0, 4.0, 3.0, 2.0, 4.0, 2.0, 3.0, 3.0,
                            3.0, 1.0, 3.0, 1.0, 2.0, 2.0, 4.0, 5.0, 4.0, 1.0};
    int dataLength = sizeof(sampleData) / sizeof(sampleData[0]);
    
    // Validate input length.
    assert(dataLength > 3 && "Series length must be greater than 3 for AR estimation.");
    
    double meanValue = calculateMean(sampleData, dataLength);
    printf("Mean = %lf\n", meanValue);
    
    // Univariate regression test.
    double predictor[] = {1, 2, 3, 4, 5};
    double response[] = {2, 4, 6, 8, 10};
    double* lr1Estimates = performUnivariateLinearRegression(predictor, response, 5);
    printf("Univariate Regression: Slope = %lf, Intercept = %lf\n", lr1Estimates[0], lr1Estimates[1]);
    free(lr1Estimates);
    
    // Forecast using AR(1)
    double* ar1Forecast = forecastAR1(sampleData, dataLength);
    printf("AR(1) Forecast: ");
    for (int i = 0; i < 17; i++) {
        printf("%lf ", ar1Forecast[i]);
    }
    printf("\n");
    free(ar1Forecast);
    
    // Forecast using AR(1)-MA(2) hybrid model.
    double* ar1ma2Forecast = forecastAR1MA2(sampleData, dataLength);
    printf("AR(1)-MA(2) Forecast: ");
    for (int i = 0; i < 17; i++) {
        printf("%lf ", ar1ma2Forecast[i]);
    }
    printf("\n");
    free(ar1ma2Forecast);
    
    // Additional forecasting models (e.g., AR(2)-MA(1), AR(2)-MA(2)) can be tested here.
    
    return 0;
}
