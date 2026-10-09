"""Offline public measurement datasets distributed with statsmodels."""

import hashlib
from .pipeline import Series


def load_builtin(name, max_rows=None, columns=None):
    import statsmodels.api as sm

    if name == "co2":
        frame = sm.datasets.co2.load_pandas().data
        # Restrict to observed measurement timestamps; gaps remain explicit in time.
        frame = frame.dropna()
        time = (frame.index - frame.index[0]).total_seconds().to_numpy()
    elif name == "sunspots":
        frame = sm.datasets.sunspots.load_pandas().data
        time = frame.pop("YEAR").to_numpy()
    elif name == "macrodata":
        frame = sm.datasets.macrodata.load_pandas().data
        time = (frame.pop("year") + (frame.pop("quarter") - 1) / 4).to_numpy()
    else:
        raise ValueError(name)
    if columns:
        frame = frame[columns]
    if max_rows:
        frame, time = frame.iloc[:max_rows], time[:max_rows]
    values = frame.to_numpy(float)
    return Series(
        time,
        values,
        list(frame.columns),
        {
            "dataset": name,
            "provenance": "statsmodels public dataset",
            "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            "missing_policy": "unobserved timestamps excluded, no fabricated measurement labels",
            "time_origin": frame.index[0].isoformat() if name == "co2" else None,
            "time_units": "seconds" if name == "co2" else "year",
        },
    )
