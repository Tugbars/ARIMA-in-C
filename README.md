# ARIMA Implementation – Design Decisions and Rationale

This ARIMA (AutoRegressive Integrated Moving Average) implementation in C is a robust, modular forecasting tool designed to handle the complexities of real-world time series data, such as the noisy `sampleData` with its dramatic jumps from 10 to 40. Our goal was to craft a system that’s mathematically sound yet practical, blending robust statistical methods, efficient numerical techniques, and data-driven automation—all within a self-contained C codebase. Below, I’ll guide you through how it works, why we chose this path, and the trade-offs we navigated, incorporating every decision and enhancement we made together.

Our journey began with a clear objective: build an ARIMA model that thrives on noisy data without relying on external libraries. We opted for Huber loss to tame outliers, QR decomposition for precise root checking, and automated order selection via ADF tests and ACF/PACF diagnostics. Let’s explore how each piece fits into the puzzle.

---

We kick off with **data preprocessing**, a vital step to stabilize our input series before modeling. The `adjustOutliers` function uses the Median Absolute Deviation (MAD) to identify and cap extreme values. For a series \( y_t \), we compute the median \( m \), then the MAD as:

![Equation](https://latex.codecogs.com/svg.latex?\color{white}\text{MAD} = \text{median}(|y_t - m|))

We scale it by 0.6745 (the normal consistency constant) and a factor of 2.5 to set a threshold:

![Equation](https://latex.codecogs.com/svg.latex?\color{white}\text{threshold} = 2.5 \times \frac{\text{MAD}}{0.6745})

Values outside \( m \pm \text{threshold} \) are clipped to these bounds. Why MAD? It’s less sensitive to outliers like 40.229511 than standard deviation, ensuring robustness—a key choice given our data’s volatility. Early on, we stumbled: `qsort` used `strcmp` (for strings) instead of a double comparison. We fixed this with:

```c
int double_cmp(const void *a, const void *b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}
```

This correction ensures accurate sorting, making the adjustment reliable. The upside is a smoother series for ARIMA; the downside is a fixed \( k = 2.5 \), which might not suit every dataset’s outlier profile.

---

Next, we ensure **stationarity**, ARIMA’s foundation. The `differenceSeries` function applies \( d \)-th order differencing to remove trends:

![Equation](https://latex.codecogs.com/svg.latex?\color{white}\Delta y_t = y_t - y_{t-1}, \quad \Delta^d y_t = \Delta(\Delta^{d-1} y_t))

Each pass shortens the series by 1, so for \( d = 1 \) in `main`, we compute \( y_t' = y_t - y_{t-1} \). We paired this with `ensureStationary`, which automates \( d \) using the Augmented Dickey-Fuller (ADF) test:

- **ADF Test (`ADFTestExtendedAutoLag`)**: Fits the regression:

  ![Equation](https://latex.codecogs.com/svg.latex?\color{white}\Delta y_t = \alpha + \beta y_{t-1} + \sum_{j=1}^p \gamma_j \Delta y_{t-j} + \epsilon_t)

  It selects \( p \) by minimizing AIC up to \( \lfloor (n-1)^{1/3} \rfloor \). If the p-value \( \geq 0.05 \), it differences again, up to a maximum \( d \) (default 2, or user-specified).
- **P-Value**: Linearly interpolates between critical values (-3.43, -2.86, -2.57) for 1%, 5%, and 10% significance.

Why automate \( d \)? ARIMA demands stationarity—a stable mean and variance—and ADF removes the guesswork, adapting to the data. The iterative approach is straightforward, aligning with ARIMA’s “I” component. However, ADF’s limited power against near-unit roots and our basic p-value interpolation are trade-offs; a full distribution table or KPSS test could refine it.

To undo differencing, `integrateSeries` reconstructs the original scale:

![Equation](https://latex.codecogs.com/svg.latex?\color{white}y_{t+h} = y_t + \Delta y_{t+1}, \quad y_{t+h} = y_{t+h-1} + \Delta y_{t+h} \quad \text{(for } h > 1\text{)})

Using the last observed value \( y_t \), it sums \( \Delta y_{t+h} \) cumulatively. This is perfect for \( d = 1 \), our focus, but for \( d > 1 \), it lacks prior values (e.g., \( y_{t-1} \)), a simplification we accepted for simplicity.

---

### Parameter Estimation

Now stationary, we estimate AR and MA parameters. For AR, `estimateARWithCMLE` models:

![Equation](https://latex.codecogs.com/svg.latex?\color{white}y_t = \mu + \phi_1 y_{t-1} + \cdots + \phi_p y_{t-p} + \epsilon_t)

We chose Conditional Maximum Likelihood Estimation (CMLE) with Newton-Raphson:

- **Initialization**: Yule-Walker via `yuleWalker`, solving:

  ![Equation](https://latex.codecogs.com/svg.latex?\color{white}R \phi = r)

  where \( R_{ij} = r_{|i-j|} \) (Toeplitz ACF matrix) and \( r = [r_1, ..., r_p] \). It’s quick and stable, a solid launchpad.
- **Loss**: Huber loss for robustness:

  ![Equation](https://latex.codecogs.com/svg.latex?\color{white}L(\epsilon) = \begin{cases} \frac{1}{2} \epsilon^2 & \text{if } |\epsilon| \leq \delta \\ \delta (|\epsilon| - \frac{\delta}{2}) & \text{otherwise} \end{cases})

  with \( \delta = 1.345 \) (95% efficiency under normality).
- **Optimization**: Updates \( \theta = [\phi_1, ..., \phi_p, \mu] \) via:

  ![Equation](https://latex.codecogs.com/svg.latex?\color{white}\theta = \theta - \text{step} \cdot H^{-1} g)

  where:
  - Gradient: \( g_i = -\sum \epsilon_t y_{t-i-1} \), \( g_\mu = -\sum \epsilon_t \).
  - Hessian: \( H_{ij} = \sum y_{t-i-1} y_{t-j-1} \), \( H_{\mu\mu} = 1 \).
  - Line search halves `step` if loss rises.

Why Huber? Gaussian MLE falters with outliers like 40.229511; Huber’s linear tail mitigates this, a deliberate robustness boost. The gradient uses raw errors instead of Huber’s \( \psi \)-function—a simplification that may slightly bias \( \phi \), but Newton-Raphson’s line search ensures convergence. The Hessian approximation mimics Gaussian curvature, a practical choice over exact computation.

For MA, `estimateMAWithMLE` models:

![Equation](https://latex.codecogs.com/svg.latex?\color{white}y_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \cdots + \theta_q \epsilon_{t-q})

- **Initialization**: \( \theta_i = 0.5 \cdot r_{i+1} \) from ACF, damped for stability.
- **Errors**: Recursively:

  ![Equation](https://latex.codecogs.com/svg.latex?\color{white}\epsilon_t = y_t - (\mu + \sum_{j=1}^q \theta_j \epsilon_{t-j}))

  assuming \( \epsilon_{t<q} = 0 \).
- **Loss and Optimization**: Huber-based Newton-Raphson, with:

  ![Equation](https://latex.codecogs.com/svg.latex?\color{white}g_i = -\sum \epsilon_t \epsilon_{t-i-1}, \quad H_{ij} = \sum \epsilon_{t-i-1} \epsilon_{t-j-1})

This aligns with AR for consistency, leveraging Huber’s robustness. The initial error assumption is typical for conditional MLE but adds early bias—exact likelihood (e.g., Kalman filtering) was bypassed for simplicity.

Both use `checkRoots` to enforce stationarity (AR) and invertibility (MA), computing eigenvalues of:

![Equation](https://latex.codecogs.com/svg.latex?\color{white}\text{AR: } 1 - \phi_1 z - \cdots - \phi_p z^p, \quad \text{MA: } 1 + \theta_1 z + \cdots + \theta_q z^q)

via QR iteration. Roots \( |\text{root}| \leq 1.001 \) are scaled by 0.95. QR replaced our initial power iteration for accuracy—a critical upgrade.

---

### Forecasting

`forecastARIMA` orchestrates the process:

- **Preprocessing**: Applies `adjustOutliers`.
- **Stationarity**: Sets \( d \) via `ensureStationary`, or auto-selects \( p, d, q \) with `selectOrdersWithFeedback` (ACF/PACF) and ADF if \( -1 \) is passed.
- **Estimation**: Runs `estimateARWithCMLE` and `estimateMAWithMLE`.
- **Forecasting**: Recursively:

  ![Equation](https://latex.codecogs.com/svg.latex?\color{white}y_{t+h} = \mu + \sum_{j=1}^p \phi_j y_{t+h-j} + \sum_{k=1}^q \theta_k \epsilon_{t+h-k})

  where \( y_{t+h-j} = f_{t+h-j} \) if \( h-j > 0 \), \( \epsilon_{t+h-k} = 0 \) if \( h-k > 0 \).
- **Variance**:

  ![Equation](https://latex.codecogs.com/svg.latex?\color{white}\sigma_h^2 = \sigma^2 \left(1 + \sum_{j=0}^{h-1} \psi_j^2\right))

  with \( \psi_j = \phi_j \) or \( \theta_j \), \( \sigma^2 = \frac{\sum \epsilon_t^2}{n} \).
- **Integration**: `integrateSeries` for \( d > 0 \).

This mirrors standard ARIMA forecasting, enhanced by Huber and QR robustness. Auto-selection uses ACF (\( q \)) and PACF (\( p \)) with a \( \frac{2}{\sqrt{n}} \) threshold—simple but effective, though lacking BIC/AIC refinement. Variance is a basic psi-weight sum, underestimating long-term uncertainty without full MA(\( \infty \)) expansion.

---

### Diagnostics

We check residuals with:
- **Residual ACF (`computeResidualACF`)**:

  ![Equation](https://latex.codecogs.com/svg.latex?\color{white}r_k = \frac{\sum (e_t - \bar{e})(e_{t+k} - \bar{e})}{\sum (e_t - \bar{e})^2})

  reporting max \( |r_k| \).
- **Ljung-Box (`computeLjungBox`)**:

  ![Equation](https://latex.codecogs.com/svg.latex?\color{white}Q = n(n+2) \sum_{k=1}^m \frac{r_k^2}{n-k})

These are correct, offering residual whiteness insights. Raw stats without thresholds give flexibility—users can compare to \( \frac{2}{\sqrt{n}} \) or chi-squared values.

---

### Design Decisions and Trade-Offs

Our approach prioritizes robustness (Huber, QR), automation (ADF, order selection), and simplicity (no external libraries). Key enhancements:
- Fixed `adjustOutliers` with `double_cmp`.
- Upgraded to QR from power iteration in `checkRoots`.
- Chose Huber over Gaussian for outlier resilience.

**Upsides**:
- Excels with noisy data (e.g., `sampleData`).
- Self-contained, efficient for small \( p, q \).
- Adaptive and robust.

**Downsides**:
- Gradient approximations may bias estimates.
- Limited \( d > 1 \) integration support.
- Simplified variance underestimates long-term uncertainty.

This ARIMA(2,1,4) shines for noisy series, striking a practical balance between theory and application.

---
