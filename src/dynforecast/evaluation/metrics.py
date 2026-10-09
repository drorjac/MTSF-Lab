import numpy as np


def forecast_metrics(truth, predictions, training, seasonality=1):
    if truth.shape != predictions.shape or not np.isfinite(predictions).all():
        raise ValueError("Metrics require finite, identically shaped truth and predictions")
    error = predictions - truth
    from dynforecast.data.pipeline import trajectories

    if seasonality < 1:
        raise ValueError("MASE seasonality must be positive")
    differences = [
        np.abs(t[seasonality:] - t[:-seasonality])
        for t in trajectories(training)
        if len(t) > seasonality
    ]
    denominator = (
        np.mean(np.concatenate(differences), axis=0) if differences else np.zeros(truth.shape[-1])
    )
    mase = None if np.any(denominator < 1e-12) else float(np.mean(np.abs(error) / denominator))
    smape_den = np.abs(truth) + np.abs(predictions)
    smape = np.divide(
        2 * np.abs(error), smape_den, out=np.zeros_like(error), where=smape_den > 1e-12
    )
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mase": mase,
        "smape": float(100 * np.mean(smape)),
        "mae_by_horizon": np.mean(np.abs(error), axis=(0, 2)).tolist(),
        "rmse_by_horizon": np.sqrt(np.mean(error**2, axis=(0, 2))).tolist(),
        "rmse_by_target": np.sqrt(np.mean(error**2, axis=(0, 1))).tolist(),
        "forecast_mean": predictions.mean(axis=(0, 1)).tolist(),
        "forecast_std": predictions.std(axis=(0, 1)).tolist(),
        "truth_mean": truth.mean(axis=(0, 1)).tolist(),
        "truth_std": truth.std(axis=(0, 1)).tolist(),
    }


def equation_metrics(model, series, scaler, selected, test_start):
    """Synthetic-only evaluation in the same standardized coordinates as identification.

    Independent test states determine derivative prediction error. Polynomial truth is
    derived algebraically by fitting the known field on a seeded off-trajectory grid;
    it is never used in identification. Nonpolynomial/nonautonomous truth is excluded.
    """
    from dynforecast.simulations import vector_field

    system = series.metadata.get("system")
    if series.derivatives is None or len(selected) != series.latent.shape[1]:
        return {"equation_recovery": "unavailable: full latent-state observability required"}
    physical = series.latent[test_start:, selected]
    z = (physical - scaler.mean) / scaler.scale
    true_dz = series.derivatives[test_start:, selected] / scaler.scale
    predicted = model.library.transform(z) @ model.coefficients
    result = {"test_derivative_rmse": float(np.sqrt(np.mean((predicted - true_dz) ** 2)))}
    if series.metadata.get("drift"):
        result["equation_recovery"] = (
            "unavailable: parameter-switching truth is not one autonomous field"
        )
        return result
    if system not in ("oscillator", "vanderpol", "lotka_volterra", "lorenz"):
        result["equation_recovery"] = "unavailable: autonomous polynomial truth required"
        return result
    degree_needed = (
        3 if system == "vanderpol" else 2 if system in ("lorenz", "lotka_volterra") else 1
    )
    if model.degree < degree_needed:
        result["equation_recovery"] = "unavailable: candidate library omits true terms"
        return result
    rng = np.random.default_rng(1729)
    grid = rng.normal(size=(max(200, len(model.library.terms) * 10), len(selected)))
    field = vector_field(system, series.metadata["params"])
    unstandardized = grid * scaler.scale + scaler.mean
    full = np.empty_like(unstandardized)
    full[:, selected] = unstandardized
    derivatives = np.stack([field(0.0, point) for point in full])[:, selected] / scaler.scale
    reference = np.linalg.lstsq(model.library.transform(grid), derivatives, rcond=None)[0]
    active, estimated = np.abs(reference) > 1e-7, np.abs(model.coefficients) > 1e-7
    tp = (active & estimated).sum()
    result.update(
        {
            "coefficient_relative_error": float(
                np.linalg.norm(model.coefficients - reference)
                / max(np.linalg.norm(reference), 1e-12)
            ),
            "support_precision": float(tp / max(estimated.sum(), 1)),
            "support_recall": float(tp / max(active.sum(), 1)),
            "reference_coefficients": reference.tolist(),
        }
    )
    return result
