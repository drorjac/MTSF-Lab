# Architecture

```
data/          Series, loaders (CSV, TSF, downloads), splits, windows, scaling
simulations/   synthetic systems: oscillator, Van der Pol, Lorenz-63, Duffing, ...
models/        Forecaster interface + registry of 31 models
training/      shared PyTorch loop: early stopping, checkpoints, TensorBoard
evaluation/    forecast, equation-recovery, and rollout metrics
experiments/   runner (one run), collection (many trajectories), sweeps, tuning, scheduler
visualization/ static report, research tables, offline dashboard
cli.py         `dynforecast run | benchmark | report | dashboard | tune | backtest | ...`
```

## Data

`data.Series` keeps what a model may see (`observations`, timestamps) apart from
what only evaluation may use (simulator `latent` states and `derivatives`).
`Standardizer` is fitted on the training prefix only and forward-fills missing
inputs causally. `windows`, `training_windows`, and `transition_pairs` never build
a window or transition across two independent trajectories.

## Models

Every model implements `Forecaster.fit(train, validation, targets, history, horizon, dt)`
and `predict(histories) -> [origins, horizon, targets]`. `models/__init__.py` holds
the registry: each name maps to a builder and two flags, `dynamical` (needs regular,
unforced data) and `uses_prior` (needs the two-state oscillator). To add a model,
implement the interface and add one `ModelSpec` line.

## Running experiments

`experiments.runner.run_experiment` runs one config:
`prepare_data -> fit_model -> forecast -> score -> save_artifacts`. The run folder
name is a hash of config, data, code, and package versions, so finished runs are
reused and any change produces a new run. `collection` does the same for datasets of
independent trajectories, splitting whole trajectories. `tuning` selects
hyperparameters with Optuna on validation scores only, then evaluates once on test.
`scheduler` runs jobs in subprocesses with hard timeouts and CPU/GPU slots.

## Reporting

`visualization.report` compares runs only within an identical condition and uses
seeds as replicates. `tables` adds corruption-degradation and cross-dataset MASE
tables. `dashboard` writes one self-contained HTML page with the data and Plotly
embedded.
