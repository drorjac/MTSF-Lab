# Research questions

The platform exists to answer one question under controlled conditions:

> When, why, and under what conditions do data-driven, model-based, and hybrid
> approaches outperform one another in learning and forecasting dynamical time series?

## Dimensions

| Dimension | Question | Config |
|---|---|---|
| Univariate vs. multivariate | When does observing additional coupled variables help? | `experiments/observability.yaml` |
| Modeling paradigm | When do mechanistic, data-driven, or hybrid models win? | `research_demo.py` |
| Temporal vs. dynamical learning | Does learning the evolution law beat learning an input-output map? | `experiments/unseen_trajectories.yaml` |
| Data efficiency | How much training data does each method need? | `experiments/data_efficiency.yaml` |
| Forecast horizon | How does error accumulate with lead time? | `experiments/horizons.yaml` |
| Robustness | Which methods withstand noise, gaps, outliers, and drift? | `experiments/robustness.yaml`, `parameter_shift.yaml` |
| Compute | What is the accuracy / complexity / training-cost trade-off? | cost figures in every report |

## Main hypothesis

Hybrid models should beat purely data-driven networks in low-data settings and at
long horizons *when their structural assumptions roughly match the process*. The
benchmark deliberately includes systems where that assumption fails (chaotic
Lorenz-63, forced Duffing, partial observations), so the hypothesis can be refuted.

## Hybrid designs under test

1. **SINDy + neural residual** (`sindy_hybrid`): identify sparse equations, then
   learn the residual dynamics. Compared with SINDy and a neural ODE alone.
2. **Multi-horizon physics-guided network** (`physics_guided`): an LSTM trained on
   short-horizon, long-horizon, and dynamics-consistency losses,
   `L = λs·L_short + λl·L_long + λd·L_dyn`. Does the constraint slow error growth?
3. **Adaptive mixture** (`adaptive_hybrid`): a learned, state-dependent gate between
   a mechanistic and a neural forecast. Does it beat plain averaging
   (`average_hybrid`)?

## Open directions

- **Data-efficiency crossover.** Is there a training-set size at which a purely
  data-driven model starts beating a model with a strong but imperfect prior?
- **Structural generalization.** Do learned evolution laws extrapolate better to
  unseen initial conditions, forcing regimes, and long horizons than sequence
  predictors, and how much does misspecified physics hurt?
- **Value of extra sensors.** In which systems is additional observability useful,
  and when does it only make learning harder? This connects to RF sensing and
  environmental nowcasting work.
