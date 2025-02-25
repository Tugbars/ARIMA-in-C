# ARIMA Implementation – Design Decisions and Rationale

This implementation of ARIMA forecasting in C takes a modular approach that splits the problem into several mathematically and algorithmically distinct steps. Below we explain each major component, the decisions made during implementation, and the benefits and drawbacks of these choices.

---

## 1. Linear Regression Functions

### What It Does

- **Univariate Regression (LR1):**  
  The function first centers the data by subtracting the means of the predictor and response. Then it computes the standard deviations and uses the Pearson correlation coefficient to determine the slope ($\beta$) via the relationship  

  $$
  \beta = \text{Corr} \times \frac{S_y}{S_x}
  $$

  and the intercept as  

  $$
  \text{intercept} = \bar{y} - \beta\,\bar{x}
  $$

- **Bivariate and Multivariate Regression (LR2, LRM):**  
  For two or more predictors, the code computes the necessary sums and products (or uses normalized design matrices) to derive closed-form expressions based on covariances. In the multivariate case, the normal equations $X^\top X \beta = X^\top Y$ are solved via matrix inversion (using LU decomposition for small matrices, e.g., $3 \times 3$ or $4 \times 4$).

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
  The implementation uses an augmented Dickey–Fuller (ADF) test to determine the necessary order of differencing ($d$) to achieve stationarity. The test iteratively:
  1. Creates a lagged and a lead version of the series.
  2. Regresses the lead on the lagged series using LR1.
  3. Checks whether the estimated coefficient is “close enough” to unity ($\pm 1.001$ tolerance).  
  If not, the series is differenced (each iteration reduces the series length by one), and the test is repeated.

- **Drift Adjustment:**  
  Once the series is differenced, the recovery (or drift) information is saved. Forecasts produced on the differenced scale can then be "integrated" back to the original scale by cumulatively summing the drift information.

### Why It’s Done This Way

- **Ensuring Stationarity:**  
  ARIMA modeling assumes that the time series is stationary. By automatically applying the ADF test and differencing until stationarity is reached, the implementation avoids manual selection of $d$ and adapts to the input data.

- **Simple and Iterative:**  
  The iterative differencing and testing approach is straightforward to implement and understand, even though more sophisticated methods (like KPSS or information criterion-based selection) exist.

### Upsides and Downsides

- **Upside:**  
  - Automatic, rule-of-thumb method for determining differencing order.  
  - Maintains simplicity and transparency.

- **Downside:**  
  - The use of a fixed tolerance ($\pm 1.001$) and a simple piecewise linear approximation for p-values means that the test might be less robust than state-of-the-art methods.  
  - Over-differencing is a risk if the rule-of-thumb does not capture the true dynamics of the series.

---

## 3. Diagnostic Matrix (EAFMatrix)

### What It Does

- **Extended Autocorrelation Function Matrix:**  
  This component computes a $3 \times 3$ matrix of extended autocorrelations that serves as a diagnostic tool. The steps include:
  1. Calculating correlations between the original series and various shifted versions (leads/lags).
  2. Fitting an AR(1) model (using LR1) and then computing the autocorrelations of the model’s errors.
  3. Repeating the process for an AR(2) model.

  The resulting matrix encapsulates information about the serial dependence structure of the data.

### Why It’s Done This Way

- **Model Selection Aid:**  
  The extended autocorrelation function (EAF) matrix provides diagnostic information that can be used to select the most appropriate forecasting model by comparing error autocorrelations from different AR orders.

### Upsides and Downsides

- **Upside:**  
  - Provides a clear, numerical diagnostic that can guide model selection.

- **Downside:**  
  - The process is computationally intensive (since it involves repeated regressions) and relies on the stability of OLS estimates.

---

## 4. Forecasting Models

### What They Do

- **AR(1) Forecasting:**  
  The AR(1) model is estimated by regressing $y_t$ on $y_{t-1}$. Forecasts are then generated recursively using:

  $$
  \hat{y}_{t+h} = \phi^h y_t + \left(1 + \phi + \phi^2 + \cdots + \phi^{h-1}\right) \text{intercept}
  $$

  Additionally, an in-sample forecast error variance is computed as a measure of forecast uncertainty.

- **Hybrid AR(1)-MA(1) and AR(1)-MA(2) Models:**  
  For these models, the AR part is estimated via the above OLS method. The moving average (MA) parameters are then estimated using an adaptive gradient descent algorithm.  

  - **Gradient Descent for MA Estimation:**  
    The objective function is the sum of squared forecast errors on the differenced series:

    $$
    J(\theta, c) = \sum_{i} \left(y_i - (\theta \times \text{lag}[i] + c)\right)^2
    $$

    The gradients with respect to $\theta$ and $c$ are computed, and an adaptive learning rate is used:
    - If the update reduces $J$, the learning rate is increased.
    - Otherwise, it is decreased.

### Why It’s Done This Way

- **Separation of AR and MA Components:**  
  AR parameters are estimated using well-understood OLS methods, while MA parameters—which lack closed-form solutions—are estimated via gradient descent.

### Upsides and Downsides

- **Upside:**  
  - The AR part is estimated robustly via OLS.  
  - The use of adaptive gradient descent for the MA part is straightforward and adaptable to small datasets.

- **Downside:**  
  - The separate estimation of AR and MA components may introduce bias compared to a full maximum likelihood joint estimation.  
  - The gradient descent algorithm can be sensitive to initial conditions, learning rate adjustments, and may converge slowly in some cases.

---
