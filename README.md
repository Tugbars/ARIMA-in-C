# ARIMA Implementation – Design Decisions and Rationale

This implementation of ARIMA forecasting in C takes a modular approach that splits the problem into several mathematically and algorithmically distinct steps. Below we explain each major component, the decisions made during implementation, and the benefits and drawbacks of these choices.

---

## 1. Linear Regression Functions

### What It Does

- **Univariate Regression (LR1):**  
  The function first centers the data by subtracting the means of the predictor and response. Then it computes the standard deviations and uses the Pearson correlation coefficient to determine the slope (β) via the relationship:

  ![Equation](https://latex.codecogs.com/svg.latex?\beta%20%3D%20\text{Corr}%20\times%20\frac{S_y}{S_x})

  and the intercept as:

  ![Equation](https://latex.codecogs.com/svg.latex?\text{intercept}%20%3D%20\bar{y}%20-%20\beta\bar{x})

- **Bivariate and Multivariate Regression (LR2, LRM):**  
  For two or more predictors, the code computes the necessary sums and products (or uses normalized design matrices) to derive closed-form expressions based on covariances. In the multivariate case, the normal equations:

  ![Equation](https://latex.codecogs.com/svg.latex?X^\top%20X%20\beta%20%3D%20X^\top%20Y)

  are solved via matrix inversion (using LU decomposition for small matrices, e.g., 3×3 or 4×4).

- **Correlation Function (Corr):**  
  The Pearson correlation between two arrays is computed by centering the data, calculating the sum of the product of deviations, and then normalizing by the product of the standard deviations.

### Why It’s Done This Way

- **Simplicity and Speed:**  
  Closed-form OLS solutions are computationally fast and stable when the underlying assumptions hold. Centering and standardizing the data also help mitigate numerical issues.

- **Modularity:**  
  By isolating these routines, the AR and MA parameter estimation steps can re-use standard linear regression methods. In particular, the AR part of the ARIMA model is estimated via these OLS methods.

### Upsides and Downsides

- **Upside:**  
  - Fast, easy-to-understand, and relatively robust for well-behaved data.  
  - Provides a natural way to estimate the AR part of an ARIMA model.

- **Downside:**  
  - OLS is not the most efficient method if errors are non-Gaussian or if there is serial correlation.  
  - For large matrices or more complex models, closed-form solutions may become less stable (hence the reliance on LU decomposition for small systems).

---

## 2. Stationarity Testing and Data Preprocessing

### What It Does

- **ADF Test (DFTest):**  
  The implementation uses an augmented Dickey–Fuller (ADF) test to determine the necessary order of differencing \(d\) to achieve stationarity. The test iteratively:
  1. Creates a lagged and a lead version of the series.
  2. Regresses the lead on the lagged series using LR1.
  3. Checks whether the estimated coefficient is “close enough” to unity (±1.001 tolerance).  

  If not, the series is differenced (each iteration reduces the series length by one), and the test is repeated.

- **Drift Adjustment:**  
  Once the series is differenced, the recovery (or drift) information is saved. Forecasts produced on the differenced scale can then be "integrated" back to the original scale by cumulatively summing the drift information.

### Why It’s Done This Way

- **Ensuring Stationarity:**  
  ARIMA modeling assumes that the time series is stationary. By automatically applying the ADF test and differencing until stationarity is reached, the implementation avoids manual selection of \(d\) and adapts to the input data.

- **Simple and Iterative:**  
  The iterative differencing and testing approach is straightforward to implement and understand, even though more sophisticated methods (like KPSS or information criterion-based selection) exist.

---

## 3. Forecasting Models

### What They Do

- **AR(1) Forecasting:**  
  The AR(1) model is estimated by regressing \( y_t \) on \( y_{t-1} \). Forecasts are then generated recursively using:

  ![Equation](https://latex.codecogs.com/svg.latex?\hat{y}_{t%2Bh}%20%3D%20\phi^h%20y_t%20%2B%20\left(1%20%2B%20\phi%20%2B%20\phi^2%20%2B%20\cdots%20%2B%20\phi^{h-1}\right)%20\text{intercept})

  Additionally, an in-sample forecast error variance is computed as a measure of forecast uncertainty.

- **Hybrid AR(1)-MA(1) and AR(1)-MA(2) Models:**  
  For these models, the AR part is estimated via the above OLS method. The moving average (MA) parameters are then estimated using an adaptive gradient descent algorithm.

  - **Gradient Descent for MA Estimation:**  
    The objective function is the sum of squared forecast errors on the differenced series:

    ![Equation](https://latex.codecogs.com/svg.latex?J(\theta%2C%20c)%20%3D%20\sum_{i}%20\left(y_i%20-%20(\theta%20\times%20\text{lag}[i]%20%2B%20c)\right)^2.)

    The gradients with respect to \(\theta\) and \(c\) are computed, and an adaptive learning rate is used:
    - If the update reduces \(J\), the learning rate is increased.
    - Otherwise, it is decreased.

- **Forecast Recovery:**  
  When differencing is applied to the series (to achieve stationarity), forecasts are initially generated on the differenced scale. The `recoverForecast()` function integrates these forecasts (via cumulative summation) and adds back drift information so that the final predictions are on the original scale.
