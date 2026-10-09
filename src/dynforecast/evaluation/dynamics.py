import numpy as np
from scipy.signal import welch
from scipy.stats import wasserstein_distance


def dynamical_metrics(truth, predictions, dt):
    """Compare rollout distributions, autocorrelations and spectral densities.

    Windows are kept as independent rollout segments; overlapping windows are not
    concatenated into a fictional continuous trajectory or used as independent CIs.
    """
    result = {"distribution_wasserstein": [], "autocorrelation_error": [], "spectral_distance": []}
    for target in range(truth.shape[-1]):
        actual, predicted = truth[..., target], predictions[..., target]
        result["distribution_wasserstein"].append(
            float(wasserstein_distance(actual.ravel(), predicted.ravel()))
        )
        errors = []
        for lag in range(1, min(10, truth.shape[1] // 2)):

            def autocorrelation(values):
                centered = values - values.mean(axis=1, keepdims=True)
                denominator = np.mean(centered**2, axis=1)
                numerator = np.mean(centered[:, lag:] * centered[:, :-lag], axis=1)
                return np.divide(
                    numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12
                )

            errors.append(np.mean(np.abs(autocorrelation(actual) - autocorrelation(predicted))))
        result["autocorrelation_error"].append(float(np.mean(errors)) if errors else None)
        if truth.shape[1] >= 8:
            _, one = welch(actual, fs=1 / dt, axis=1, nperseg=truth.shape[1])
            _, two = welch(predicted, fs=1 / dt, axis=1, nperseg=truth.shape[1])
            one, two = one.mean(axis=0), two.mean(axis=0)
            one /= max(one.sum(), 1e-12)
            two /= max(two.sum(), 1e-12)
            result["spectral_distance"].append(float(np.linalg.norm(one - two)))
        else:
            result["spectral_distance"].append(None)
    result["maximum_norm"] = float(np.max(np.linalg.norm(predictions, axis=-1)))
    result["scope"] = (
        "short rollout segment diagnostics; not a Lyapunov or invariant-measure estimate"
    )
    return result
