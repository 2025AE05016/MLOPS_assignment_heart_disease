from pathlib import Path
import hashlib
import json
import urllib.request

import pandas as pd

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]
URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"


def acquire(root=Path("data")):
    import io
    import zipfile
    root = Path(root)
    raw = root / "raw/processed.cleveland.data"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        raw.write_bytes(archive.read("processed.cleveland.data"))
    frame = pd.read_csv(raw, names=FEATURES + ["target"], na_values="?")
    if len(frame) != 303 or not frame.target.isin(range(5)).all():
        raise ValueError("Unexpected Cleveland dataset shape or targets")
    frame["target"] = (frame.target > 0).astype(int)
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    frame.to_csv(processed / "heart.csv", index=False)
    metadata = {"source": URL, "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "rows": len(frame), "missing": frame.isna().sum().to_dict()}
    (processed / "provenance.json").write_text(json.dumps(metadata, indent=2))
    return frame
