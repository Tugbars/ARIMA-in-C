# ARIMA Implementation – Design Decisions and Rationale

This ARIMA (AutoRegressive Integrated Moving Average) forecasting implementation in C is crafted to be modular, robust, and adaptable, breaking the complex ARIMA process into distinct, mathematically grounded steps. Each component reflects deliberate design choices aimed at balancing accuracy, computational efficiency, and resilience to real-world data challenges (e.g., outliers in `sampleData`). Below, I’ll walk through how the system works, why we made these choices, and the trade-offs involved, weaving in the improvements we iterated on together.

Our journey started with a goal: build an ARIMA model that’s not just theoretically sound but practical for noisy, non-ideal time series like the one with jumps from 10 to 40 in `sampleData`. We opted for a blend of robust statistical methods (Huber loss), proven numerical techniques (QR decomposition), and data-driven automation (ADF tests, ACF/PACF order selection), all while keeping the code self-contained in C.

---

We begin with **data preprocessing**, a critical step to tame the wild swings in our input series. The `adjustOutliers` function uses the Median Absolute Deviation (MAD) to detect and cap extreme values. For a series \( y_t \), we compute the median \( m \), then calculate the MAD as:

![equation](https://latex.codecogs.com/svg.latex?\color{white}\text{MAD}%20%3D%20\text{median}(|y_t%20-%20m|))

This is scaled by 0.6745 (the normal consistency constant) and a factor of 2.5 to set a threshold:

![equation](https://latex.codecogs.com/svg.latex?\color{white}\text{threshold}%20%3D%202.5%20\times%20\frac{\text{MAD}}{0.6745})

Values beyond \( m \pm \text{threshold} \) are clipped to these bounds. Why MAD over standard deviation? It’s robust—less influenced by outliers like the 40.229511 spike we’re targeting. Initially, we encountered an issue: `qsort` used `strcmp` (for strings) instead of a double comparison. We corrected this with:

```c
int double_cmp(const void *a, const void *b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}
```

This fix ensures accurate sorting, making the outlier adjustment mathematically valid. The upside is a cleaner series for ARIMA estimation; the downside is a fixed \( k = 2.5 \), which might be too strict or lenient depending on the data’s tail behavior.

---

Next, we tackle **stationarity**, the backbone of ARIMA, ensuring a stable mean and variance crucial for reliable modeling. The `differenceSeries` function applies \( d \)-th order differencing to remove trends:

![equation](https://latex.codecogs.com/svg.latex?\color{white}\Delta%20y_t%20%3D%20y_t%20-%20y_{t-1},%20\quad%20\Delta^d%20y_t%20%3D%20\Delta(\Delta^{d-1}%20y_t))

Each iteration reduces the series length by 1. For our specification in `main` with \( d = 1 \), the result is:

![equation](https://latex.codecogs.com/svg.latex?\color{white}y_t'%20%3D%20y_t%20-%20y_{t-1})

To automate the choice of \( d \), we use `ensureStationary`, which relies on the Augmented Dickey-Fuller (ADF) test:

- **ADF Test (`ADFTestExtendedAutoLag`)**: Fits the regression:

  ![equation](https://latex.codecogs.com/svg.latex?\color{white}\Delta%20y_t%20%3D%20\alpha%20+%20\beta%20y_{t-1}%20+%20\sum_{j=1}^p%20\gamma_j%20\Delta%20y_{t-j}%20+%20\epsilon_t)

  It selects the number of lags \( p \) by minimizing the Akaike Information Criterion (AIC), up to a maximum of:

  ![equation](https://latex.codecogs.com/svg.latex?\color{white}\lfloor%20(n-1)^{1/3}%20\rfloor)

  where \( n \) is the series length. If the p-value satisfies:

  ![equation](https://latex.codecogs.com/svg.latex?\color{white}\text{p-value}%20\geq%200.05)

  the series is differenced again, up to a maximum \( d \) (default 2, or user-specified).
- **P-Value**: Interpolates linearly between critical values (-3.43, -2.86, -2.57) for 1%, 5%, and 10% significance levels.

Why this method? ARIMA assumes stationarity, and ADF automation adapts \( d \) to the data, removing guesswork. The iterative differencing aligns with ARIMA’s “I” component—simple and effective. However, ADF’s limited power against near-unit roots and our basic p-value interpolation are trade-offs; more detailed statistical tables or a KPSS test could improve accuracy.

To reverse differencing, `integrateSeries` reconstructs the original scale:

![equation](https://latex.codecogs.com/svg.latex?\color{white}y_{t+h}%20%3D%20y_t%20+%20\Delta%20y_{t+1})

For subsequent steps:

![equation](https://latex.codecogs.com/svg.latex?\color{white}y_{t+h}%20%3D%20y_{t+h-1}%20+%20\Delta%20y_{t+h}%20\quad%20\text{(for%20}h%20>%201\text{)})

Starting with the last observed value \( y_t \), it sums the differenced forecasts cumulatively. This works perfectly for \( d = 1 \), our focus, but for \( d > 1 \), it’d need prior values (e.g., \( y_{t-1} \))—a simplification we chose for practicality.

---

### Parameter Estimation

With a stationary series, we estimate AR and MA parameters. For AR, `estimateARWithCMLE` models:

![equation](https://latex.codecogs.com/svg.latex?\color{white}y_t%20%3D%20\mu%20+%20\phi_1%20y_{t-1}%20+%20\cdots%20+%20\phi_p%20y_{t-p}%20+%20\epsilon_t)

We chose Conditional Maximum Likelihood Estimation (CMLE) with Newton-Raphson:

- **Initialization**: Yule-Walker via `yuleWalker`, solving:

  ![equation](https://latex.codecogs.com/svg.latex?\color{white}R%20\phi%20%3D%20r)

  where \( R_{ij} = r_{|i-j|} \) (Toeplitz ACF matrix) and \( r = [r_1, ..., r_p] \). It’s fast and stable, a great starting point.
- **Loss**: Huber loss for robustness:

  ![equation](https://latex.codecogs.com/svg.latex?\color{white}L(\epsilon)%20%3D%20\begin{cases}%20\frac{1}{2}%20\epsilon^2%20&%20\text{if%20}|\epsilon|%20\leq%20\delta%20\\%20\delta%20(|\epsilon|%20-%20\frac{\delta}{2})%20&%20\text{otherwise}%20\end{cases})

  with:

  ![equation](https://latex.codecogs.com/svg.latex?\color{white}\delta%20%3D%201.345)

  (95% efficiency under normality).
- **Optimization**: Updates the parameter vector:

  ![equation](https://latex.codecogs.com/svg.latex?\color{white}\theta%20%3D%20[\phi_1,%20...,%20\phi_p,%20\mu])

  using:

  ![equation](https://latex.codecogs.com/svg.latex?\color{white}\theta%20%3D%20\theta%20-%20\text{step}%20\cdot%20H^{-1}%20g)

  where:
  - Gradient components are:

    ![equation](https://latex.codecogs.com/svg.latex?\color{white}g_i%20%3D%20-\sum%20\epsilon_t%20y_{t-i-1})

    for the AR coefficients, and:

    ![equation](https://latex.codecogs.com/svg.latex?\color{white}g_\mu%20%3D%20-\sum%20\epsilon_t)

    for the intercept.
  - Hessian elements are:

    ![equation](https://latex.codecogs.com/svg.latex?\color{white}H_{ij}%20%3D%20\sum%20y_{t-i-1}%20y_{t-j-1})

    for the AR terms, and:

    ![equation](https://latex.codecogs.com/svg.latex?\color{white}H_{\mu\mu}%20%3D%201)

    for the intercept.
  - Line search halves `step` if the loss increases.

Why Huber? Gaussian MLE struggles with outliers like 40.229511; Huber’s linear tail reduces their impact, enhancing robustness—a key improvement we prioritized. The gradient uses raw errors rather than Huber’s \( \psi \)-function—a simplification that might bias \( \phi \) slightly, but Newton-Raphson’s line search ensures convergence. The Hessian approximation assumes Gaussian curvature, a practical choice over exact computation.

For MA, `estimateMAWithMLE` models:

![equation](https://latex.codecogs.com/svg.latex?\color{white}y_t%20%3D%20\mu%20+%20\epsilon_t%20+%20\theta_1%20\epsilon_{t-1}%20+%20\cdots%20+%20\theta_q%20\epsilon_{t-q})

- **Initialization**: 

  ![equation](https://latex.codecogs.com/svg.latex?\color{white}\theta_i%20%3D%200.5%20\cdot%20r_{i+1})

  from ACF, damped for stability.
- **Errors**: Recursively:

  ![equation](https://latex.codecogs.com/svg.latex?\color{white}\epsilon_t%20%3D%20y_t%20-%20(\mu%20+%20\sum_{j=1}^q%20\theta_j%20\epsilon_{t-j}))

  assuming \( \epsilon_{t<q} = 0 \).
- **Loss and Optimization**: Same Huber-based Newton-Raphson as AR, with:

  ![equation](https://latex.codecogs.com/svg.latex?\color{white}g_i%20%3D%20-\sum%20\epsilon_t%20\epsilon_{t-i-1},%20\quad%20H_{ij}%20%3D%20\sum%20\epsilon_{t-i-1}%20\epsilon_{t-j-1})

This aligns with AR for consistency, leveraging Huber’s robustness. The initial error assumption is typical for conditional MLE but adds early bias—exact likelihood (e.g., Kalman filtering) was bypassed for simplicity.

Both use `checkRoots` to enforce stationarity (AR) and invertibility (MA), computing eigenvalues of:

![equation](https://latex.codecogs.com/svg.latex?\color{white}\text{AR:}%201%20-%20\phi_1%20z%20-%20\cdots%20-%20\phi_p%20z^p,%20\quad%20\text{MA:}%201%20+%20\theta_1%20z%20+%20\cdots%20+%20\theta_q%20z^q)

via QR iteration. Roots \( |\text{root}| \leq 1.001 \) are scaled by 0.95. QR replaced our initial power iteration for accuracy—a critical upgrade.

---

### Forecasting

`forecastARIMA` orchestrates the process:

- **Preprocessing**: Applies `adjustOutliers` (fixed with `double_cmp`).
- **Stationarity**: Sets \( d \) via `ensureStationary`, or auto-selects \( p, d, q \) with `selectOrdersWithFeedback` (ACF/PACF) and ADF if \( -1 \) is passed.
- **Estimation**: Runs `estimateARWithCMLE` and `estimateMAWithMLE`.
- **Forecasting**: Recursively:

  ![equation](https://latex.codecogs.com/svg.latex?\color{white}y_{t+h}%20%3D%20\mu%20+%20\sum_{j=1}^p%20\phi_j%20y_{t+h-j}%20+%20\sum_{k=1}^q%20\theta_k%20\epsilon_{t+h-k})

  where \( y_{t+h-j} = f_{t+h-j} \) if \( h-j > 0 \), \( \epsilon_{t+h-k} = 0 \) if \( h-k > 0 \).
- **Variance**: 

  ![equation](https://latex.codecogs.com/svg.latex?\color{white}\sigma_h^2%20%3D%20\sigma^2%20(1%20+%20\sum_{j=0}^{h-1}%20\psi_j^2))

  with \( \psi_j = \phi_j \) or \( \theta_j \), and:

  ![equation](https://latex.codecogs.com/svg.latex?\color{white}\sigma^2%20%3D%20\frac{\sum%20\epsilon_t^2}{n})

- **Integration**: `integrateSeries` for \( d > 0 \).

This mirrors standard ARIMA forecasting, enhanced by Huber and QR robustness. Auto-selection uses ACF (\( q \)) and PACF (\( p \)) with a \( \frac{2}{\sqrt{n}} \) threshold—simple but effective, though lacking BIC/AIC refinement. Variance is a basic psi-weight sum, underestimating long-term uncertainty without full MA(\( \infty \)) expansion.

---

### Diagnostics

We check residuals with:
- **Residual ACF (`computeResidualACF`)**:

  ![equation](https://latex.codecogs.com/svg.latex?\color{white}r_k%20%3D%20\frac{\sum%20(e_t%20-%20\bar{e})(e_{t+k}%20-%20\bar{e})}{\sum%20(e_t%20-%20\bar{e})^2})

  reporting max \( |r_k| \).
- **Ljung-Box (`computeLjungBox`)**:

  ![equation](https://latex.codecogs.com/svg.latex?\color{white}Q%20%3D%20n(n+2)%20\sum_{k=1}^m%20\frac{r_k^2}{n-k})

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
