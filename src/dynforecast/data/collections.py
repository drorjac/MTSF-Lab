import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile
from .loaders import load_tsf


def download_tsf(url, directory="data/monash", member=None):
    """Acquire a public HTTPS TSF or ZIP, validate it and preserve source provenance."""
    if not url.startswith("https://"):
        raise ValueError("Public dataset acquisition requires verified HTTPS")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = hashlib.sha256(json.dumps([url, member]).encode()).hexdigest()[:16]
    target = directory / f"{stem}.tsf"
    if not target.exists():
        temporary = directory / f"{stem}.download"
        candidate = directory / f"{stem}.candidate"
        try:
            with (
                urllib.request.urlopen(url, timeout=60) as response,
                temporary.open("wb") as output,
            ):
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if zipfile.is_zipfile(temporary):
                with zipfile.ZipFile(temporary) as archive:
                    members = [name for name in archive.namelist() if name.lower().endswith(".tsf")]
                    selected = member or (members[0] if len(members) == 1 else None)
                    if selected is None:
                        raise ValueError(
                            "Select a TSF member when the archive contains multiple collections"
                        )
                    candidate.write_bytes(archive.read(selected))
            else:
                candidate.write_bytes(temporary.read_bytes())
            load_tsf(candidate)
            candidate.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
            candidate.unlink(missing_ok=True)
    load_tsf(target)
    target.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "url": url,
                "member": member,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "format": "Monash TSF",
            },
            indent=2,
        )
    )
    return target
