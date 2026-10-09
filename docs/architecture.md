# Architecture

`data.Series` separates observations from simulator states, derivatives, forcing,
timestamps, and provenance. Models receive observations only. CSV/TSF loaders
preserve actual timestamps and independent series. Downloads validate before
installation; SHA-256 hashes identify the inputs.

`Standardizer` owns training statistics and causal missing-input handling.
`windows`, `training_windows`, and `transition_pairs` support independent
trajectories without artificial transitions. Single-series experiments use
chronological splits; collections split whole trajectories.

The explicit `models.Forecaster` registry selects 31 native models and references.
Direct predictors emit complete horizons; statistical/state-space and continuous
fields recurse. Diagnostics identify their objective, prior, and rollout type.

`training.train_network` owns seeding, Adam, gradient clipping, early stopping,
best validation state, optional TensorBoard, and atomic optimizer/RNG checkpoints.
Native latent dynamics encode history then integrate an inferred latent state.

`experiments.runner` and `collection` own identity, locking, preprocessing,
evaluation, artifacts, and failures. `tuning` selects on validation using persistent
Optuna studies before final test evaluation. Backtests fit successive training
prefixes. `scheduler` isolates jobs and allocates CPU/CUDA slots under hard timeouts.

`evaluation` measures physical error, eligible polynomial equation recovery, and
bounded rollout statistics. `visualization.report` facets exact data/source/target
conditions and seed pairing. `tables` computes matched corruption degradation and
tagged MASE aggregates. `dashboard` embeds actual arrays and Plotly in one offline page.

To add a model, implement fit/predict, register it, declare its data assumptions,
and verify learning and rollout behavior. Methods requiring true states, future
inputs, or simulator derivatives belong in explicitly labeled oracle protocols.
