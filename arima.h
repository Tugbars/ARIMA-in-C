/**
 * @file timeseries_forecast.c
 * @brief Implements time‐series forecasting routines including utility functions,
 * linear regression routines, stationarity testing, diagnostic matrix computations, and error metrics.
 *
 * This file contains helper functions for array and matrix operations, various linear regression
 * functions (univariate, bivariate, and multivariate), procedures for testing and transforming a 
 * non‐stationary time series (using a Dickey–Fuller style approach), computing an extended 
 * autocorrelation matrix for model diagnostics, and functions to compute forecast error metrics 
 * such as MAPE and MAE.
 */

/*==============================================================================
  Group: Utility Functions for Array and Matrix Operations
==============================================================================*/




/**
 * @defgroup ArrayMatrixOperations Utility Functions for Array and Matrix Operations
 * @{
 *
 * @brief Functions to perform basic statistical calculations, lag/lead shifts, and matrix operations.
 */

/**
 * @brief Computes the mean (average) of a 1D array.
 *
 * @param arr    The array of doubles.
 * @param length The number of elements in the array.
 * @return The mean value of the array.
 */
double calcMean(double arr[], int length);

/**
 * @brief Computes the sum of the elements in a 1D array.
 *
 * @param arr    The array of doubles.
 * @param length The number of elements in the array.
 * @return The sum of the array elements.
 */
double calcArrSum(double arr[], int length);

/**
 * @brief Computes the sample standard deviation of a 1D array.
 *
 * @param arr    The array of doubles.
 * @param length The number of elements in the array.
 * @return The sample standard deviation.
 */
double calcSD(double arr[], int length);

/**
 * @brief Computes element‐by‐element differences between two 1D arrays.
 *
 * @param arr1     The first array.
 * @param arr2     The second array.
 * @param arr_diff Output array that will store the differences (arr1[i] - arr2[i]).
 * @param size     The number of elements in each array.
 */
void calcArrDiff(double arr1[], double arr2[], double arr_diff[], int size);

/**
 * @brief Computes element‐by‐element differences between two 2D arrays (matrices).
 *
 * @param m        The number of rows.
 * @param n        The number of columns.
 * @param arr1     The first matrix.
 * @param arr2     The second matrix.
 * @param arr_diff Output matrix that will contain the differences.
 */
void calcMDArrDiff(int m, int n, double arr1[][n], double arr2[][n], double arr_diff[][n]);

/**
 * @brief Computes the element‐wise product of two arrays.
 *
 * @param arr1 The first array.
 * @param arr2 The second array.
 * @param size The number of elements in each array.
 * @return Pointer to an array containing the element‐wise products.
 *
 * @warning The returned pointer is to an array allocated on the stack. In production code,
 * consider using dynamic allocation to avoid undefined behavior.
 */
double *calcArrProd(double arr1[], double arr2[], int size);

/**
 * @brief Produces a new array in which each element is squared.
 *
 * @param arr  The input array.
 * @param size The number of elements in the array.
 * @return Pointer to an array containing the squares of the original array elements.
 *
 * @warning This function returns a pointer to a local array. In production code this may cause
 * memory issues and should be handled appropriately.
 */
double *squareArr(double arr[], int size);

/*--- Lag/Lead Operations ---*/

/**
 * @brief Shifts an array forward by a specified lag to produce a “lead” series.
 *
 * @param arr      The input array.
 * @param arr_lead The output array such that arr_lead[i] = arr[i+lag].
 * @param length   The number of elements in the input array.
 * @param lag      The lead shift amount.
 */
void calcArrLead(double arr[], double arr_lead[], int length, int lag);

/**
 * @brief Produces a lagged version of the input array.
 *
 * @param arr      The input array.
 * @param arr_lag  The output array containing lagged values.
 * @param length   The number of elements in the array.
 * @param lag      The lag amount.
 */
void calcArrLag(double arr[], double arr_lag[], int length, int lag);

/**
 * @brief A variant of the lag/lead operation that allows specifying both a lead and a lag.
 *
 * @param arr      The input array.
 * @param arr_lead The output array after shifting.
 * @param length   The number of elements in the array.
 * @param lead     The lead amount.
 * @param lag      The lag amount.
 */
void calcArrLeadLag(double arr[], double arr_lead[], int length, int lead, int lag);

/*--- Matrix Operations ---*/

/**
 * @brief Computes the mean of each column in a 2D array.
 *
 * @param m        The number of rows.
 * @param n        The number of columns.
 * @param arr      The input 2D array.
 * @param mean_arr Output 2D array to store the column means.
 */
void calc2DArrMean(int m, int n, double arr[][n], double mean_arr[][n]);

/**
 * @brief Normalizes a 2D array by subtracting the mean of each column.
 *
 * @param m        The number of rows.
 * @param n        The number of columns.
 * @param arr      The input 2D array.
 * @param norm_arr Output 2D array that will contain the normalized values.
 */
void modArrNorm(int m, int n, double arr[][n], double norm_arr[][n]);

/**
 * @brief Column binds three 1D arrays into a 2D array.
 *
 * @param m         The number of rows.
 * @param n         The number of columns (should be 3).
 * @param arr1      The first 1D array.
 * @param arr2      The second 1D array.
 * @param arr3      The third 1D array.
 * @param cbind_arr Output 2D array with each input array as one column.
 */
void Cbind(int m, int n, double arr1[], double arr2[], double arr3[], double cbind_arr[][n]);

/**
 * @brief Column binds four 1D arrays into a 2D array.
 *
 * @param m         The number of rows.
 * @param n         The number of columns (should be 4).
 * @param arr1      The first 1D array.
 * @param arr2      The second 1D array.
 * @param arr3      The third 1D array.
 * @param arr4      The fourth 1D array.
 * @param cbind_arr Output 2D array with four columns.
 */
void Cbind4(int m, int n, double arr1[], double arr2[], double arr3[], double arr4[], double cbind_arr[][n]);

/**
 * @brief Converts a 1D array into a 2D array with one column.
 *
 * @param m      The number of elements.
 * @param arr    The input 1D array.
 * @param arr_2d Output 2D array (with one column).
 */
void D2MD(int m, double arr[], double arr_2d[][1]);

/**
 * @brief Converts a 2D array with one column into a 1D array.
 *
 * @param m       The number of rows.
 * @param arr     The input 2D array.
 * @param arr_1d  Output 1D array.
 */
void MD2D(int m, double arr[][1], double arr_1d[]);

/**
 * @brief Computes the transpose of a matrix.
 *
 * @param m     Number of rows in the input matrix.
 * @param n     Number of columns in the input matrix.
 * @param arr   The input matrix.
 * @param t_arr Output matrix that will store the transpose.
 * @param size  A parameter (possibly the total number of elements) used for processing.
 */
void transpose(int m, int n, double arr[][n], double t_arr[][m], int size);

/**
 * @brief Performs matrix multiplication between two matrices.
 *
 * @param m1     Number of rows in the first matrix.
 * @param n1     Number of columns in the first matrix (and rows in the second).
 * @param m2     Number of rows in the second matrix.
 * @param n2     Number of columns in the second matrix.
 * @param arr1   The first matrix.
 * @param arr2   The second matrix.
 * @param p_arr  Output matrix that will contain the product (dimensions m1 x n2).
 * @param size   A parameter (possibly total number of elements) used for processing.
 */
void product3(int m1, int n1, int m2, int n2, double arr1[][n1], double arr2[][n2], double p_arr[][n2], int size);

/**
 * @brief Computes the inverse of a 3x3 matrix.
 *
 * @param n       The dimension of the matrix (should be 3).
 * @param a       The input matrix.
 * @param inv_arr Output matrix that will contain the inverse of a.
 * @param size    A parameter (possibly the size of the matrix) used for processing.
 */
void inverse(int n, double a[][n], double inv_arr[][n], int size);

/**
 * @brief Computes the inverse of a 4x4 matrix.
 *
 * @param m   The input 4x4 matrix.
 * @param inv Output 4x4 matrix that will contain the inverse.
 */
void inverse4(double m[4][4], double inv[4][4]);

/** @} */ // end of ArrayMatrixOperations group

/*==============================================================================
  Group: Linear Regression Functions
==============================================================================*/

/**
 * @defgroup LinearRegressionFunctions Linear Regression Functions
 * @{
 *
 * @brief Functions to estimate parameters using linear regression in univariate,
 * bivariate, and multivariate settings.
 */

/**
 * @brief Computes the slope (beta) and intercept by regressing y on x.
 *
 * @param arr_x  The predictor array.
 * @param arr_y  The response array.
 * @param length The number of observations.
 * @return Pointer to an array containing two elements: beta (slope) and intercept.
 *
 * @details The function centers the data, computes standard deviations and Pearson correlation,
 * and uses the formulas:
 *          beta = Corr * (S_y / S_x)
 *          intercept = mean(y) - beta * mean(x)
 */
double *LR1(double arr_x[], double arr_y[], int length);

/**
 * @brief Generates predictions from a univariate linear regression model.
 *
 * @param arr       The predictor array.
 * @param arr_pred  Output array that will contain the predicted values.
 * @param beta      The estimated slope.
 * @param intercept The estimated intercept.
 * @param length    The number of observations.
 */
void LR1_Pred(double arr[], double arr_pred[], double beta, double intercept, int length);

/**
 * @brief Computes regression estimates for a model with two predictors.
 *
 * @param arr_x1 First predictor array.
 * @param arr_x2 Second predictor array.
 * @param arr_y  Response array.
 * @param length Number of observations.
 * @return Pointer to an array containing beta1, beta2, and the intercept.
 */
double *LR2(double arr_x1[], double arr_x2[], double arr_y[], int length);

/**
 * @brief Computes predictions for a bivariate regression model.
 *
 * @param arr_x1   First predictor array.
 * @param arr_x2   Second predictor array.
 * @param arr_pred Output array that will contain the predicted values.
 * @param beta1    Estimated coefficient for the first predictor.
 * @param beta2    Estimated coefficient for the second predictor.
 * @param intercept Estimated intercept.
 * @param length   Number of observations.
 */
void LR2_Pred(double arr_x1[], double arr_x2[], double arr_pred[], double beta1, double beta2, double intercept, int length);

/**
 * @brief Performs multivariate linear regression with n predictors and m observations.
 *
 * @param m Number of observations.
 * @param n Number of predictors.
 * @param X Design matrix of predictors.
 * @param Y Response variable (2D array with one column).
 * @return Pointer to an array containing the n estimated coefficients followed by the intercept.
 *
 * @details The design matrix is first normalized by subtracting column means. The function
 * then forms the normal equations (XᵀX) and solves for the coefficients using matrix inversion.
 * The intercept is computed separately from the means of X and Y.
 */
double *LRM(int m, int n, double X[][n], double Y[][1]);

/**
 * @brief Computes predictions from a multivariate regression model.
 *
 * @param m         Number of observations.
 * @param n         Number of predictors.
 * @param X         Design matrix of predictors.
 * @param Y_pred    Output 2D array (one column) that will contain the predicted values.
 * @param estimates Array containing the estimated coefficients and intercept (last element).
 */
void LRM_Pred(int m, int n, double X[][n], double Y_pred[][1], double estimates[n]);

/**
 * @brief Computes the Pearson correlation coefficient between two arrays.
 *
 * @param arr_x  First array.
 * @param arr_y  Second array.
 * @param length Number of observations.
 * @return The Pearson correlation coefficient.
 */
double Corr(double arr_x[], double arr_y[], int length);

/** @} */ // end of LinearRegressionFunctions group

/*==============================================================================
  Group: Stationarity Testing and Data Preprocessing
==============================================================================*/

/**
 * @defgroup StationarityTesting Stationarity Testing and Data Preprocessing
 * @{
 *
 * @brief Functions to determine the order of differencing required for stationarity
 * and to adjust the time series accordingly.
 */

/**
 * @brief Iteratively applies a Dickey–Fuller style regression test to determine the required differencing order.
 *
 * @param arr      The input time series array.
 * @param arr_recov Output array that stores recovery information (used for later drift adjustment).
 * @param length   The number of observations in the time series.
 * @return The order of differencing (d) applied to achieve stationarity.
 *
 * @details The function creates lagged and lead versions of the series and regresses the lead on the lagged series using LR1.
 * If the estimated coefficient is not within ±1.001 of unity, the series is differenced and the test repeats.
 */
int DFTest(double arr[], double arr_recov[], int length);

/**
 * @brief Adjusts a differenced series to “recover” the original level by removing the drift.
 *
 * @param arr      The original time series array.
 * @param arr_stry The series that will be adjusted.
 * @param length   The number of observations.
 * @param d        The differencing order applied.
 *
 * @details The function performs cumulative adjustments to add back the drift lost during differencing.
 */
void Drift(double arr[], double arr_stry[], int length, int d);

/** @} */ // end of StationarityTesting group

/*==============================================================================
  Group: Autocorrelation/Diagnostic Matrix
==============================================================================*/

/**
 * @defgroup DiagnosticMatrix Autocorrelation/Diagnostic Matrix (EAFMatrix)
 * @{
 *
 * @brief Computes an extended autocorrelation matrix for model diagnostic purposes.
 *
 * @details This function calculates correlations between the series and its leads/lags (lags 1, 2, and 3).
 * It then fits an AR(1) model using LR1 to compute prediction errors and determines the autocorrelation of these errors.
 * Finally, the procedure is repeated for an AR(2) model, resulting in a 3x3 diagnostic matrix.
 */

/**
 * @brief Computes the extended autocorrelation function (EAF) matrix.
 *
 * @param arr     The input time series array.
 * @param arr_eaf Output 3x3 matrix where each element is the correlation between the series (or its errors) and its various leads/lags.
 * @param length  The number of observations in the time series.
 */
void EAFMatrix(double arr[], double arr_eaf[][3], int length);

/** @} */ // end of DiagnosticMatrix group

/*==============================================================================
  Group: Error Metrics
==============================================================================*/

/**
 * @defgroup ErrorMetrics Error Metrics
 * @{
 *
 * @brief Functions to compute forecast error metrics.
 */

/**
 * @brief Computes the Mean Absolute Percentage Error (MAPE) between forecasted and actual values.
 *
 * @param arr_y     Array of actual values.
 * @param arr_y_cap Array of forecasted (predicted) values.
 * @param length    Number of observations.
 * @return The MAPE expressed as a percentage.
 */
double MAPE(double arr_y[], double arr_y_cap[], int length);

/**
 * @brief Computes the Mean Absolute Error (MAE) between forecasted and actual values.
 *
 * @param arr_y     Array of actual values.
 * @param arr_y_cap Array of forecasted values.
 * @param length    Number of observations.
 * @return The MAE.
 */
double MAE(double arr_y[], double arr_y_cap[], int length);

/** @} */ // end of ErrorMetrics group
