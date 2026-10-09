import numpy as np
import pytest
from dynforecast.simulations import simulate, vector_field


@pytest.mark.parametrize("name", ["oscillator", "vanderpol", "lotka_volterra", "lorenz", "duffing"])
def test_simulation_derivatives_and_observations(name):
    # Small sampling step keeps finite-difference truncation error controlled for Lorenz.
    clean = simulate(name, n=100, dt=0.001)
    noisy = simulate(name, n=100, dt=0.001, noise=0.1, observed=[0])
    np.testing.assert_array_equal(clean.latent, noisy.latent)
    np.testing.assert_allclose(
        np.gradient(clean.latent, 0.001, axis=0)[2:-2],
        clean.derivatives[2:-2],
        rtol=0.01,
        atol=0.01,
    )
    assert noisy.observations.shape == (100, 1)
    assert not np.array_equal(noisy.observations[:, 0], noisy.latent[:, 0])
    f = vector_field(name, clean.metadata["params"])
    np.testing.assert_allclose(f(clean.time[20], clean.latent[20]), clean.derivatives[20])


def test_oscillator_energy_decays():
    series = simulate("oscillator", n=500)
    energy = np.sum(series.latent**2, axis=1) / 2
    assert np.max(np.diff(energy)) < 1e-8
