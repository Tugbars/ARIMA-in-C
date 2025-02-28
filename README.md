Below is an updated and comprehensive documentation for your ARIMA implementation in C, designed for a GitHub README. It follows the formatting style of your old documentation (e.g., LaTeX equations framed with `![Equation](...)`, white text color for visibility in dark themes), while adopting a more organic explanatory flow with titles used to emphasize key sections rather than as rigid separators. Since we developed this code together, I’ve incorporated all methods, design decisions, improvements, and rationales we discussed, ensuring an elaborate and detailed explanation tailored to your implementation.

---

# ARIMA Implementation – Design Decisions and Rationale

This ARIMA (AutoRegressive Integrated Moving Average) forecasting implementation in C is crafted to be modular, robust, and adaptable, breaking the complex ARIMA process into distinct, mathematically grounded steps. Each component reflects deliberate design choices aimed at balancing accuracy, computational efficiency, and resilience to real-world data challenges (e.g., outliers in `sampleData`). Below, I’ll walk through how the system works, why we made these choices, and the trade-offs involved, weaving in the improvements we iterated on together.

Our journey started with a goal: build an ARIMA model that’s not just theoretically sound but practical for noisy, non-ideal time series like the one with jumps from 10 to 40 in `sampleData`. We opted for a blend of robust statistical methods (Huber loss), proven numerical techniques (QR decomposition), and data-driven automation (ADF tests, ACF/PACF order selection), all while keeping the code self-contained in C.

---

We begin with **data preprocessing**, a critical step to tame the wild swings in our input series. The `adjustOutliers` function uses the Median Absolute Deviation (MAD) to detect and cap extreme values. For a series \( y_t \), we compute the median \( m \), then the MAD as \( \text{median}(|y_t - m|) \), scaling it by 0.6745 (the normal consistency constant) and a factor of 2.5 to set a threshold:

![Equation](https://latex.codecogs.com/svg.latex?\color{white}\text{threshold}%20%3D%202.5%20\times%20\frac{\text{MAD}}{0.6745})

Values beyond \( m \pm \text{threshold} \) are clipped to these bounds. Why MAD over standard deviation? It’s robust—less swayed by the very outliers we’re targeting, like that 40.229511 spike. Initially, we hit a snag: `qsort` used `strcmp` (for strings) instead of a double comparison. We fixed this with:

```c
int double_cmp(const void *a, const void *b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}
```

This ensures proper sorting, making outlier adjustment mathematically valid. The upside is a cleaner series for ARIMA estimation; the downside is a fixed \( k = 2.5 \), which might be too strict or lenient depending on the data’s tail behavior.

---

Next, we tackle **stationarity**, the backbone of ARIMA. The `differenceSeries` function applies \( d \)-th order differencing (\( \Delta^d y_t \)) to remove trends:

![Equation](https://latex.codecogs.com/svg.latex?\color{white}\Delta%20y_t%20%3D%20y_t%20-%20y_{t-1},\quad%20\Delta^d%20y_t%20%3D%20\Delta(\Delta^{d-1}%20y_t))

Each iteration reduces the series length by 1, so for \( d = 1 \) in our `main` call, we get \( y_t' = y_t - y_{t-1} \). We paired this with `ensureStationary`, which uses an Augmented Dickey-Fuller (ADF) test to determine \( d \):

- **ADF Test (`ADFTestExtendedAutoLag`)**: Fits \( \Delta y_t = \alpha + \beta y_{t-1} + \sum_{j=1}^p \gamma_j \Delta y_{t-j} + \epsilon_t \), selecting \( p \) via AIC minimization up to \( \lfloor (n-1)^{1/3} \rfloor \). If \( p \)-value \( \geq 0.05 \), it differences again, up to a max \( d \) (default 2, or user-specified).
- **P-Value**: Interpolates between critical values (-3.43, -2.86, -2.57).

Why this way? Automating \( d \) with ADF ensures stationarity without guesswork, critical since ARIMA assumes a stable mean and variance. The iterative differencing is simple and aligns with ARIMA’s “I” component. However, ADF’s low power against near-unit roots and our crude p-value interpolation (linear between 1% and 10%) are limitations—more precise tables or KPSS tests could complement it.

To reverse differencing, `integrateSeries` integrates forecasts back to the original scale:

![Equation](https://latex.codecogs.com/svg.latex?\color{white}y_{t+h}%20%3D%20y_t%20+%20\Delta%20y_{t+1},\quad%20y_{t+h}%20%3D%20y_{t+h-1}%20+%20\Delta%20y_{t+h}%20\text{(for%20}h%20>%201\text{)})

Starting with \( y_{t} \) (last observed value), it cumulatively sums \( \Delta y_{t+h} \). This is spot-on for \( d = 1 \), but for \( d > 1 \), we’d need prior values (e.g., \( y_{t-1} \)), a simplification we accepted since \( d = 1 \) suits our use case.

---

### Parameter Estimation

With a stationary series, we estimate AR and MA parameters. For the AR part, `estimateARWithCMLE` models:

![Equation](https://latex.codecogs.com/svg.latex?\color{white}y_t%20%3D%20\mu%20+%20\phi_1%20y_{t-1}%20+%20\cdots%20+%20\phi_p%20y_{t-p}%20+%20\epsilon_t)

We chose **Conditional Maximum Likelihood Estimation (CMLE)** with Newton-Raphson optimization:

- **Initialization**: Yule-Walker estimates via `yuleWalker`, solving \( R \phi = r \) where \( R_{ij} = r_{|i-j|} \) (Toeplitz ACF matrix) and \( r = [r_1, ..., r_p] \). Fast and stable, it’s a great starting point.
- **Loss**: Huber loss for robustness:
  ![Equation](https://latex.codecogs.com/svg.latex?\color{white}L(\epsilon)%20%3D%20\begin{cases}%20\frac{1}{2}%20\epsilon^2%20&%20\text{if%20}|\epsilon|%20\leq%20\delta%20\\%20\delta%20(|\epsilon|%20-%20\frac{\delta}{2})%20&%20\text{otherwise}\end{cases})
  with \( \delta = 1.345 \) (95% efficiency under normality).
- **Optimization**: Updates \( \theta = [\phi_1, ..., \phi_p, \mu] \) via \( \theta = \theta - \text{step} \cdot H^{-1} g \), where:
  - Gradient: \( g_i = -\sum \epsilon_t y_{t-i-1} \), \( g_\mu = -\sum \epsilon_t \).
  - Hessian: \( H_{ij} = \sum y_{t-i-1} y_{t-j-1} \), \( H_{\mu\mu} = 1 \).
  - Line search halves `step` if loss increases.

Why Huber over Gaussian? Outliers like 40.229511 could skew Gaussian MLE; Huber’s linear tail dampens their effect, improving robustness—a key improvement we prioritized. The gradient approximation (raw errors vs. Huber’s \( \psi \)) is a trade-off: it’s simpler but may bias \( \phi \) slightly. Newton-Raphson with line search ensures convergence, though the approximate Hessian assumes Gaussian curvature, a practical compromise.

For the MA part, `estimateMAWithMLE` models:

![Equation](https://latex.codecogs.com/svg.latex?\color{white}y_t%20%3D%20\mu%20+%20\epsilon_t%20+%20\theta_1%20\epsilon_{t-1}%20+%20\cdots%20+%20\theta_q%20\epsilon_{t-q})

- **Initialization**: \( \theta_i = 0.5 \cdot r_{i+1} \) from ACF, a heuristic damped for stability.
- **Errors**: Recursively \( \epsilon_t = y_t - (\mu + \sum \theta_j \epsilon_{t-j}) \), assuming \( \epsilon_{t<q} = 0 \).
- **Loss and Optimization**: Same Huber-based Newton-Raphson as AR, with \( g_i = -\sum \epsilon_t \epsilon_{t-i-1} \), \( H_{ij} = \sum \epsilon_{t-i-1} \epsilon_{t-j-1} \).

This mirrors AR estimation for consistency, leveraging Huber’s robustness. The initial error assumption is standard in conditional MLE but introduces early bias—unavoidable without exact likelihood (e.g., Kalman filtering), which we avoided for simplicity.

Both AR and MA use `checkRoots` to ensure stationarity/invertibility, computing companion matrix eigenvalues via QR iteration:

![Equation](https://latex.codecogs.com/svg.latex?\color{white}\text{AR:%20}1%20-%20\phi_1%20z%20-%20\cdots%20-%20\phi_p%20z^p,%20\text{MA:%20}1%20+%20\theta_1%20z%20+%20\cdots%20+%20\theta_q%20z^q)

Roots \( |\text{root}| \leq 1.001 \) trigger a 0.95 scaling. QR’s accuracy (over our initial power iteration attempt) ensures reliable stability checks, a critical enhancement.

---

### Forecasting

The heart of the system, `forecastARIMA`, ties everything together:

- **Preprocessing**: Applies `adjustOutliers` (fixed with `double_cmp`).
- **Stationarity**: Uses `ensureStationary` to set \( d \), or auto-selects \( p, d, q \) via `selectOrdersWithFeedback` (ACF/PACF) and ADF if \( -1 \) is passed.
- **Estimation**: Calls `estimateARWithCMLE` and `estimateMAWithMLE`.
- **Forecasting**: Recursively:
  ![Equation](https://latex.codecogs.com/svg.latex?\color{white}y_{t+h}%20%3D%20\mu%20+%20\sum_{j=1}^p%20\phi_j%20y_{t+h-j}%20+%20\sum_{k=1}^q%20\theta_k%20\epsilon_{t+h-k})
  where \( y_{t+h-j} = f_{t+h-j} \) if \( h-j > 0 \), \( \epsilon_{t+h-k} = 0 \) if \( h-k > 0 \).
- **Variance**: \( \sigma_h^2 = \sigma^2 (1 + \sum_{j=0}^{h-1} \psi_j^2) \), with \( \psi_j = \phi_j \) or \( \theta_j \), \( \sigma^2 = \sum \epsilon_t^2 / n \).
- **Integration**: `integrateSeries` for \( d > 0 \).

This is textbook ARIMA forecasting, with robustness from Huber loss and QR. Auto-selection via ACF (\( q \)) and PACF (\( p \)) with a 2/√n threshold is standard, though simplified (no BIC/AIC refinement). Variance uses a basic psi-weight approach—effective but underestimates long-term uncertainty without full MA(\( \infty \)) expansion.

---

### Diagnostics

Post-forecast, we assess fit with:
- **Residual ACF (`computeResidualACF`)**: \( r_k = \frac{\sum (e_t - \bar{e})(e_{t+k} - \bar{e})}{\sum (e_t - \bar{e})^2} \), reporting max \( |r_k| \).
- **Ljung-Box (`computeLjungBox`)**: \( Q = n(n+2) \sum_{k=1}^m \frac{r_k^2}{n-k} \).

Both are mathematically correct, providing insight into residual whiteness. We opted for raw stats over thresholds for flexibility—users can interpret against \( 2/\sqrt{n} \) or chi-squared tables.

---

### Design Decisions and Trade-Offs

Why this approach? We aimed for robustness (Huber, QR), automation (ADF, order selection), and simplicity (no external libraries). Key improvements:
- Fixed `adjustOutliers` sorting.
- Swapped power iteration for QR in `checkRoots`.
- Chose Huber over Gaussian likelihood for outlier resilience.

**Upsides**:
- Handles noisy data well (e.g., `sampleData`).
- Self-contained, fast for small \( p, q \).
- Robust and adaptive.

**Downsides**:
- Gradient approximations may bias estimates slightly.
- Limited \( d > 1 \) support in integration.
- Simplified variance underestimates long-term uncertainty.

This implementation shines for ARIMA(2,1,4) on noisy series, balancing theory and practice effectively.

---
