import hashlib
import io
import json
import zipfile
import pytest
from dynforecast.data.loaders import download_dataset


@pytest.mark.parametrize("name,target", [("ett", "OT"), ("jena", "T (degC)")])
def test_public_downloader_validates_and_records_hash(tmp_path, monkeypatch, name, target):
    content = (
        f"date,{target}\n" + "".join(f"2020-01-{i % 28 + 1:02d},{i}\n" for i in range(120))
    ).encode()
    payload = content
    if name == "jena":
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("jena_climate_2009_2016.csv", content)
        payload = archive.getvalue()
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(payload))
    path = download_dataset(name, tmp_path)
    metadata = json.loads(path.with_suffix(".provenance.json").read_text())
    assert metadata["sha256"] == hashlib.sha256(content).hexdigest()
    assert metadata["rows"] == 120 and metadata["url"].startswith("https://")
    assert path.read_bytes() == content


def test_downloader_rejects_wrong_schema(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(b"wrong\n1\n"))
    with pytest.raises(ValueError, match="expected tabular"):
        download_dataset("ett", tmp_path)
    assert not (tmp_path / "ETTh1.csv").exists()
    assert not list(tmp_path.glob("*.download"))
