# MTSF-Lab

A complete Python/PyTorch experiment platform for comparing time-series forecasts,
learned evolution laws, and hybrid models on identical observed targets. Built from
scratch from the [project specification](docs/project-plan.md), independently of
pysteps and LSTM_ED.

**Open the [interactive research atlas](results/end_to_end/dashboard/index.html)**
for the finished demonstration: observed histories, forecast-origin sliders,
calendar-time plots, phase portraits, learning curves, horizon heatmaps, sample
efficiency, compute comparisons, and learned equations. The page embeds its data
and Plotly, so it works offline in a normal browser. Publication figures and CSV
comparisons are in the [scientific report](results/end_to_end/report/summary.html).
The short study demonstrates functionality; it does not establish universal model
superiority. See the [validation record](docs/validation.md).

![Interactive time-series research atlas](results/end_to_end/dashboard/preview.png)

## Install and run

Use Python 3.11 and [uv](https://docs.astral.sh/uv/). Python 3.11.16 and dependencies
are pinned in `.python-version` and `uv.lock`.

```bash
bash scripts/install.sh   # also registers the "MTSF-Lab (.venv)" Jupyter kernel
# Platform-independent equivalent, including Windows:
uv sync --frozen --extra dev --extra references --extra research

uv run --frozen --all-extras dynforecast run model=lstm model.epochs=20
uv run --frozen --all-extras dynforecast benchmark --config configs/experiments/smoke.yaml
uv run --frozen --all-extras dynforecast report --results results/smoke

# All model families and compact matched research sweeps:
uv run --frozen --all-extras python scripts/research_demo.py --budget-seconds 600 --epochs 15
```

Open `results/research_demo/dashboard/index.html`. No server is necessary. The CLI
also supports `simulate`, `download`, `dashboard`, `tune`, and `backtest`. YAML
composes via relative `includes`, numeric OmegaConf interpolation, and dotted
`key=value` overrides. The scripts directory supplies Python wrapper entry points.

The lock uses official CPU PyTorch wheels. CUDA training and device-aware scheduling
are implemented, but actual validation here used CPU hardware. Supply a compatible
PyTorch build in a separate environment for GPU work and retain its dependency
manifest. Core-only installation (`uv sync --frozen`) omits reference adapters,
tuning, trackers, and interactive graphics; the full installation supports every
shipped workflow and test.

## Models and sources

| Family | Registered models |
|---|---|
| Statistics / state-space | `persistence`, `seasonal_naive`, `ar`, `var`, `arima`, `kalman` |
| Classical ML | `ridge`, `random_forest`, `xgboost` |
| Temporal neural | `mlp`, `lstm`, `gru`, `tcn`, `dlinear`, `transformer`, `nhits`, `patchtst` |
| Dynamics | `sindy`, `pysindy`, `neural_ode`, `latent_ode` |
| Mechanistic / hybrid | `mechanistic`, `trainable_mechanistic`, `hybrid`, `sindy_hybrid`, `forecast_residual`, `adaptive_hybrid`, `average_hybrid`, `physics_guided` |
| Canonical references | `nf_nhits`, `nf_patchtst` through Nixtla NeuralForecast |

All 31 models implement `fit(train, validation, targets, history, horizon, dt)` and
`predict(histories) -> [origins, horizon, targets]`. Native NHITS-style and PatchTST
models are transparent research implementations; canonical equivalence is not
claimed. Separate `nf_` adapters use the actual library models on one univariate
continuous trajectory, with a fixed step budget and MAE objective.

Synthetic sources: AR/ARX, damped oscillator, Van der Pol, Lotka–Volterra, Lorenz-63,
forced Duffing, and periodic Allen–Cahn reaction–diffusion. States, measurements,
true derivatives, and forcing are stored separately. Independent initial conditions,
parameter sets, sampling jitter, and parameter switches are supported.

Real sources: verified HTTPS ETT and daily temperature, plus offline public CO₂,
sunspot, and macroeconomic measurements distributed with statsmodels. Numeric CSV
and independent-series Monash TSF files/ZIP acquisition are supported. Downloads
carry SHA-256 provenance and validate before installation. Jena is supported, but
this cloud returned HTTP 403; working temperature and CO₂ alternatives are used
in the completed study.

```bash
uv run --frozen --all-extras dynforecast download --dataset ett
uv run --frozen --all-extras dynforecast run --config configs/experiments/real_data.yaml
uv run --frozen --all-extras dynforecast run --config configs/datasets/weather.yaml
uv run --frozen --all-extras dynforecast run dataset=csv dataset.path=data/my_sensors.csv 'targets=[0]'
uv run --frozen --all-extras dynforecast run dataset=monash dataset.path=data/collection.tsf 'targets=[0]'
```

## Research workflows

| Question | Configuration or command |
|---|---|
| Same target, different observations | `configs/experiments/observability.yaml` |
| Forecast horizons | `configs/experiments/horizons.yaml` |
| Limited training data | `configs/experiments/data_efficiency.yaml` |
| Sensor corruption | `configs/experiments/robustness.yaml` |
| Unseen initial conditions | `configs/experiments/unseen_trajectories.yaml` |
| Parameter shift | `configs/experiments/parameter_shift.yaml` |
| Partial latent dynamics | `configs/experiments/latent_state.yaml` |
| Spatial fields | `configs/experiments/spatial_dynamics.yaml` |
| Hyperparameter selection | `tune --config configs/experiments/tuning.yaml` |
| Expanding chronological backtests | `backtest` with `backtest.folds=[0.3,0.5,0.65]` |
| Hard execution limits | `scheduler.isolated=true` or `configs/compute/hard_budget.yaml` |

```bash
uv run --frozen --all-extras dynforecast tune --config configs/experiments/tuning.yaml
uv run --frozen --all-extras dynforecast backtest model=ridge 'backtest.folds=[0.3,0.5,0.65]'
uv run --frozen --all-extras dynforecast run --config configs/experiments/latent_state.yaml
uv run --frozen --all-extras dynforecast benchmark --config configs/experiments/horizons.yaml scheduler.isolated=true scheduler.cpu_workers=2 budget_seconds=600
```

Optuna uses validation-only trials stored in SQLite, then evaluates the selected
settings on test data. Optional local MLflow/TensorBoard use `tracking.mlflow=true`
and `tracking.tensorboard=true`. `profile_flops=true` records supported PyTorch
operation FLOPs, with its limited scope explicitly labeled.

## Scientific contracts

- Chronological splits default to 60/20/20. Scaling uses only the selected training
  prefix. Missing inputs use causal forward-fill, initially using training means.
  Missing ground-truth targets are never silently fabricated.
- Targets stay within their evaluation split. Rolling origins use past measurements,
  including earlier test measurements. Independent trajectories are split at
  trajectory level; windows, derivatives, transitions, and MASE never cross joins.
- Models receive observations only. SINDy uses finite differences or explicitly
  selected Savitzky–Golay smoothing. Neural ODEs learn integrated measured flow;
  `model.flow_steps` controls the objective. Latent ODE forecasting does not establish
  identification of true physical hidden states.
- Autonomous integrators require regular, unforced trajectories. Partial/real
  observation-space dynamics require `dynamics_assumption=true`; latent forecasting
  supports partial measurements. Temporal models accept irregular sequences without
  claiming a continuous-time irregular-observation solver.
- Oscillator priors require ordered position/velocity inputs. Declared prior
  parameters are independent of simulator truth and correctly scaled from physical units.
- Test-time corruption preserves clean targets. Robust training also corrupts training
  inputs. Noise, random missingness, outages, outliers, bias, and parameter shifts
  remain separate experimental conditions.
- MAE/RMSE/MASE/sMAPE use physical units; undefined MASE is null. Equation recovery
  transforms eligible polynomial truth to the model's standardized coordinates,
  using truth only for scoring. Unsupported recovery cases receive no invented score.
- Intervals and paired comparisons use independent seeds, not overlapping windows
  as replicates. Raw errors across physical datasets are separated. Explicitly tagged
  cross-dataset summaries use equally weighted MASE and common model coverage.

See [methods](docs/methods.md), [architecture](docs/architecture.md), and the
[specification map](docs/specification-map.md).

## Artifacts and resume

Content-derived run directories contain configuration, provenance, package versions,
preprocessing, metrics, diagnostics, predictions, models, and neural checkpoints.
IDs include data/source hashes. Completed runs are skipped; changed inputs create
new IDs. Epoch resume restores optimizer, best validation state, early stopping,
and RNG state. Only load pickle/checkpoint files from your own trusted runs.

Default deadlines are cooperative. With `scheduler.isolated=true`, each job runs
in an owned subprocess with a termination signal followed by a kill if needed.
Configure `job_timeout_seconds`, `cpu_workers`, and CPU/CUDA `devices`. Report
creation follows the training budget. Failures, interruptions, and unlaunched jobs
remain explicit; failed/interrupted experiments produce nonzero CLI exit status.

Reports include seed intervals, paired comparisons, matched corruption degradation,
equation recovery, failure rates, Pareto lists, and PNG/PDF/CSV outputs. The offline
atlas samples at most 150 displayed origins per run; metrics use all evaluation origins.

## PyCharm and development

Open this folder and select its `.venv` interpreter. Shared `.run/` configurations
cover tests, smoke benchmarks, and the research demonstration. Notebooks use the
"MTSF-Lab (.venv)" kernel that `scripts/install.sh` registers:

| Notebook | What it shows |
|---|---|
| [01_quickstart](notebooks/01_quickstart.ipynb) | Simulated oscillator; temporal models vs learned evolution laws |
| [02_research_workflows](notebooks/02_research_workflows.ipynb) | Observation sets, validation-only tuning, latent forecasts, atlas |
| [03_real_data_quickstart](notebooks/03_real_data_quickstart.ipynb) | ETTh1 oil temperature, 24 h ahead, four model families |
| [04_your_own_csv](notebooks/04_your_own_csv.ipynb) | Template for forecasting columns of any CSV |

```bash
uv run --frozen --all-extras pytest -q
uv run --frozen --all-extras ruff check src tests scripts
uv run --frozen --all-extras ruff format --check src tests scripts
uv run --frozen --all-extras jupyter lab notebooks
```

For restricted shells, keep `UV_CACHE_DIR`, `MPLCONFIGDIR`, and `XDG_CACHE_HOME`
under the project or `/workspace`. No credentials or persistent services are required.
Reusable cloud setup/startup instructions are saved in a draft; publishing the
prepared environment snapshot is a separate product action.
