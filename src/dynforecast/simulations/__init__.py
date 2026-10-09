"""Simulated truth is stored separately and is never passed to forecasting models."""

from scipy.integrate import solve_ivp
import numpy as np
from dynforecast.data.pipeline import Series


SYSTEMS = {
    "oscillator": {
        "initial": [1.0, 0.0],
        "params": {"stiffness": 1.0, "damping": 0.15},
        "equation": "dx=v; dv=-stiffness*x-damping*v",
    },
    "vanderpol": {
        "initial": [1.0, 0.0],
        "params": {"mu": 1.0},
        "equation": "dx=v; dv=mu*(1-x^2)*v-x",
    },
    "lotka_volterra": {
        "initial": [1.2, 1.0],
        "params": {"alpha": 1.5, "beta": 1.0, "delta": 1.0, "gamma": 3.0},
        "equation": "dx=alpha*x-beta*x*y; dy=delta*x*y-gamma*y",
    },
    "lorenz": {
        "initial": [1.0, 1.0, 1.0],
        "params": {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3},
        "equation": "dx=sigma*(y-x); dy=x*(rho-z)-y; dz=x*y-beta*z",
    },
    "duffing": {
        "initial": [1.0, 0.0],
        "params": {"damping": 0.2, "alpha": -1.0, "beta": 1.0, "amplitude": 0.3, "omega": 1.2},
        "equation": "dx=v; dv=-damping*v-alpha*x-beta*x^3+u(t)",
    },
}


def vector_field(system, params):
    p = params
    if system == "oscillator":
        return lambda t, x: np.array([x[1], -p["stiffness"] * x[0] - p["damping"] * x[1]])
    if system == "vanderpol":
        return lambda t, x: np.array([x[1], p["mu"] * (1 - x[0] ** 2) * x[1] - x[0]])
    if system == "lotka_volterra":
        return lambda t, x: np.array(
            [
                p["alpha"] * x[0] - p["beta"] * x[0] * x[1],
                p["delta"] * x[0] * x[1] - p["gamma"] * x[1],
            ]
        )
    if system == "lorenz":
        return lambda t, x: np.array(
            [
                p["sigma"] * (x[1] - x[0]),
                x[0] * (p["rho"] - x[2]) - x[1],
                x[0] * x[1] - p["beta"] * x[2],
            ]
        )
    if system == "duffing":
        return lambda t, x: np.array(
            [
                x[1],
                -p["damping"] * x[1]
                - p["alpha"] * x[0]
                - p["beta"] * x[0] ** 3
                + p["amplitude"] * np.cos(p["omega"] * t),
            ]
        )
    raise ValueError(f"Unknown continuous system: {system}")


def simulate(
    system="oscillator",
    n=600,
    dt=0.05,
    seed=0,
    noise=0.0,
    observed=None,
    initial=None,
    params=None,
    irregular=0.0,
    drift=None,
    grid=16,
):
    if n < 3 or dt <= 0 or noise < 0:
        raise ValueError("Require n>=3, dt>0 and noise>=0")
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    if not 0 <= irregular < 1:
        raise ValueError("Irregular sampling jitter must be in [0,1)")
    if irregular:
        t = np.concatenate(
            [[0.0], np.cumsum(dt * np.clip(1 + rng.normal(0, irregular, n - 1), 0.05, None))]
        )
    if system == "reaction_diffusion":
        if grid < 4:
            raise ValueError("Reaction-diffusion requires at least four periodic grid points")
        p = {"diffusion": 0.05, "reaction": 1.0}
        p.update(params or {})
        spacing = 2 * np.pi / grid
        x0 = (
            0.2 * np.sin(np.arange(grid) * spacing) + 0.1 * np.cos(2 * np.arange(grid) * spacing)
            if initial is None
            else np.asarray(initial)
        )

        def field(time, state):
            laplacian = (np.roll(state, 1) - 2 * state + np.roll(state, -1)) / spacing**2
            return p["diffusion"] * laplacian + p["reaction"] * (state - state**3)

        solution = solve_ivp(field, (t[0], t[-1]), x0, t_eval=t, rtol=1e-8, atol=1e-10)
        if not solution.success:
            raise RuntimeError(solution.message)
        x = solution.y.T
        derivative = np.stack([field(ti, xi) for ti, xi in zip(t, x)])
        observed = list(range(grid)) if observed is None else observed
        y = x[:, observed] + rng.normal(0, noise, (n, len(observed)))
        return Series(
            t,
            y,
            [f"x{i}" for i in observed],
            {
                "system": system,
                "params": p,
                "dt": dt,
                "seed": seed,
                "grid": grid,
                "observed": list(observed),
                "irregular": irregular,
                "equation": "du/dt=D*periodic_laplacian(u)+r*(u-u^3)",
                "provenance": "method of lines; periodic central differences",
            },
            x,
            derivative,
        )
    if system in ("ar", "arx"):
        p = {"coefficient": 0.8, "innovation_std": 0.1, "driver_weight": 0.3}
        p.update(params or {})
        if abs(p["coefficient"]) >= 1:
            raise ValueError("AR simulator requires a stationary coefficient")
        u = np.sin(t)[:, None] if system == "arx" else None
        x = np.zeros((n, 1))
        for i in range(1, n):
            x[i] = p["coefficient"] * x[i - 1] + rng.normal(0, p["innovation_std"])
            if u is not None:
                x[i] += p["driver_weight"] * u[i - 1]
        deriv = None
        equation = "x[k]=coefficient*x[k-1]+innovation (+driver_weight*u[k-1])"
    else:
        if system not in SYSTEMS:
            raise ValueError(f"Unknown system: {system}")
        spec = SYSTEMS[system]
        p = dict(spec["params"])
        p.update(params or {})
        base = vector_field(system, p)
        if drift:
            if set(drift) - {"at", "params"}:
                raise ValueError("Drift must specify at (time) and params")
            shifted = vector_field(system, {**p, **drift["params"]})

            def f(time, state):
                return (base if time < drift["at"] else shifted)(time, state)
        else:
            f = base
        sol = solve_ivp(
            f,
            (t[0], t[-1]),
            initial if initial is not None else spec["initial"],
            t_eval=t,
            rtol=1e-9,
            atol=1e-11,
        )
        if not sol.success or sol.y.shape[1] != n:
            raise RuntimeError(f"Simulation failed: {sol.message}")
        x = sol.y.T
        deriv = np.stack([f(ti, xi) for ti, xi in zip(t, x)])
        equation = spec["equation"]
        u = (p["amplitude"] * np.cos(p["omega"] * t))[:, None] if system == "duffing" else None
    observed = list(range(x.shape[1])) if observed is None else list(observed)
    y = x[:, observed] + rng.normal(0, noise, (n, len(observed)))
    return Series(
        t,
        y,
        [f"x{i}" for i in observed],
        {
            "system": system,
            "params": p,
            "equation": equation,
            "dt": dt,
            "seed": seed,
            "noise": noise,
            "irregular": irregular,
            "drift": drift,
            "observed": observed,
            "provenance": "solve_ivp rtol=1e-9 atol=1e-11",
        },
        x,
        deriv,
        u,
    )
