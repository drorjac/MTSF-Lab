# Mathematical and implementation notes

## Temporal and statistical models

Direct models map L observations to H target steps. MLP uses last-value centering
and a linear skip; LSTM/GRU encode history. TCN uses left-padded dilated convolutions.
Transformer uses positional embeddings and history attention; DLinear decomposes
a smoothed trend. Native NHITS-style prediction pools history at several scales,
fits backcasts, and interpolates forecast components. Native PatchTST encodes
channel-independent patches with a cross-channel target head. Canonical Nixtla
models are separate fixed-step univariate references trained with MAE.

AR/VAR use ridge-stabilized recursive least squares. ARIMA fits maximum-likelihood
training parameters then filters observed local histories. Nonconvergence is a
failed run; on noise-free deterministic signals the innovation variance can be
singular, so ARIMA is best compared on stochastic data such as the AR system.
Ridge/forest/XGBoost predict windows directly. Kalman fits an observed-coordinate
transition and covariance matrices, with identity observation mapping and
Joseph-form covariance updates.

## Sparse, neural, and hybrid dynamics

SINDy builds polynomial monomials and estimates derivatives from observations,
separately within each trajectory. Finite differences are default; Savitzky–Golay
smoothing is explicit. STLSQ repeatedly thresholds and refits active terms. Native
and PySINDy implementations are checked for agreement on a controlled oscillator.

Neural fields use differentiable RK4 flow matching over declared measured transition
steps. Simulator derivatives are never training labels. The physical oscillator
prior is `dx=v; dv=-k*x-c*v`. For standardized z, evaluate `x=s*z+m` and divide the
physical derivative by s. Trainable mechanistic models estimate positive k,c;
residual fields add a neural correction to fixed physical or training-only sparse priors.

Forecast residual hybrids correct physical predictions in output space. Adaptive
hybrids learn constrained mixture weights. Average hybrids train their temporal
component independently before averaging with physics. Physics-guided LSTM combines
short-horizon, full-horizon, and declared dynamics penalties; checkpoint selection
uses measured validation forecast MSE.

Latent ODE encodes observed history with a GRU, integrates inferred latent dynamics,
and decodes measured targets. The latent state is not the true physical state.
Autonomous integrators require regular, unforced observations. Temporal networks
accept irregular sequences but treat them as plain sequences.

## Evaluation limits

Metrics use original units; MASE training differences stay within trajectories.
Test-time corruption leaves targets clean. Robust training additionally alters
training measurements. Polynomial recovery is scored only for eligible fully
observed synthetic systems, with truth transformed to standardized coordinates for
evaluation only. Partial, forced, and parameter-drift cases get no recovery score.

Wasserstein distributions, rollout autocorrelations, normalized Welch spectra, and
maximum state norms are computed over separate rollout segments. Overlapping
origins are never joined into fictional continuous orbits or used as independent
interval replicates. They are not Lyapunov-exponent estimates, and on short chaotic
trajectories they say little about the invariant measure.

Reports preserve data/source/target/protocol facets. Seed intervals use t statistics,
and matched seed differences are paired. Tagged cross-dataset MASE gives each dataset
equal weight with common model coverage; raw RMSE is not pooled across physical units.

CPU RSS is a whole-process high-water mark, not incremental model memory. GPU memory
requires compatible hardware. Parameter counts are descriptive, not matched capacities.
Optional FLOPs cover supported PyTorch operations, excluding control and solver costs.
Timings are machine-specific. Optuna sees validation scores only; test evaluation
follows frozen parameter selection.
