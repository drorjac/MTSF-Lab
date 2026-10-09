"""Public acquisition is opt-in. Downloads use verified HTTPS and atomic replacement."""

from pathlib import Path
import hashlib
import json
import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from .pipeline import Series

URLS = {
    "ett": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "jena": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
    "weather": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
}


def _validate_table(path, dataset):
    frame = pd.read_csv(path)
    if len(frame) < 100 or frame.shape[1] < 2:
        raise ValueError("Downloaded dataset does not match expected tabular format")
    expected = {"ett": "OT", "jena": "T (degC)", "weather": "Temp"}[dataset]
    if expected not in frame.columns:
        raise ValueError(f"Missing expected target {expected}")
    return len(frame)


def download_dataset(dataset, directory="data/raw"):
    if dataset not in URLS:
        raise ValueError(
            "Automatic acquisition supports ett and jena; supply a local CSV or TSF otherwise"
        )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = (
        directory
        / {
            "ett": "ETTh1.csv",
            "jena": "jena_climate_2009_2016.csv",
            "weather": "daily-min-temperatures.csv",
        }[dataset]
    )
    if not path.exists():
        download = path.with_suffix(".download")
        candidate = path.with_suffix(".tmp")
        try:
            with (
                urllib.request.urlopen(URLS[dataset], timeout=60) as response,
                download.open("wb") as f,
            ):
                while chunk := response.read(1024 * 1024):
                    f.write(chunk)
            if dataset == "jena":
                with zipfile.ZipFile(download) as archive:
                    member = "jena_climate_2009_2016.csv"
                    with archive.open(member) as source, candidate.open("wb") as f:
                        import shutil

                        shutil.copyfileobj(source, f)
                _validate_table(candidate, dataset)
                os.replace(candidate, path)
                download.unlink()
            else:
                _validate_table(download, dataset)
                os.replace(download, path)
        finally:
            download.unlink(missing_ok=True)
            candidate.unlink(missing_ok=True)
    rows = _validate_table(path, dataset)
    provenance = {
        "dataset": dataset,
        "url": URLS[dataset],
        "rows": rows,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2))
    return path


def load_csv(path, timestamp=None, columns=None, max_rows=None, missing_values=None):
    path = Path(path)
    frame = pd.read_csv(path, nrows=max_rows)
    timestamp = (
        timestamp
        or ("Date" if "Date" in frame else None)
        or ("date" if "date" in frame else "Date Time" if "Date Time" in frame else None)
    )
    if timestamp:
        dates = pd.to_datetime(frame[timestamp], dayfirst=timestamp == "Date Time", errors="raise")
        if dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError("CSV timestamps must be ordered and unique")
        t = (dates - dates.iloc[0]).dt.total_seconds().to_numpy()
    else:
        t = np.arange(len(frame), dtype=float)
    columns = columns or list(frame.select_dtypes(include="number").columns)
    if not columns or timestamp in columns:
        raise ValueError("Choose numeric observation columns separately from timestamps")
    numeric = frame[columns].apply(pd.to_numeric, errors="raise")
    if missing_values:
        for name, sentinel in missing_values.items():
            if name in numeric:
                numeric[name] = numeric[name].replace(sentinel, np.nan)
    values = numeric.to_numpy(float)
    return Series(
        t,
        values,
        columns,
        {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "kind": "csv",
            "missing_values": missing_values,
            "time_origin": dates.iloc[0].isoformat() if timestamp else None,
            "time_units": "seconds" if timestamp else "samples",
        },
    )


def load_tsf(path):
    """Read Monash .tsf collections as independent univariate series, never concatenate."""
    attributes, result, in_data = [], [], False
    for line in Path(path).read_text(encoding="cp1252").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("@attribute"):
            attributes.append(line.split()[1])
        elif line.lower() == "@data":
            in_data = True
        elif in_data:
            fields = line.split(":")
            if len(fields) != len(attributes) + 1:
                raise ValueError("Malformed TSF row")
            values = np.array([np.nan if v == "?" else float(v) for v in fields[-1].split(",")])
            metadata = dict(zip(attributes, fields[:-1]))
            result.append(Series(np.arange(len(values)), values[:, None], ["value"], metadata))
    if not result:
        raise ValueError("No TSF series found")
    return result
