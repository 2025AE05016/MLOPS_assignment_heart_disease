"""Inference CLI: score patients with the saved pipeline or a running API.

Examples:
  python scripts/predict.py examples/patient.json
  python scripts/predict.py data/processed/heart.csv --limit 5
  python scripts/predict.py examples/patient.json --url http://localhost:8000
"""
import argparse
import json
from pathlib import Path
import urllib.request

import joblib
import pandas as pd

from heartops.data import FEATURES


def load_patients(path):
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)[FEATURES]
    records = json.loads(path.read_text())
    return pd.DataFrame(records if isinstance(records, list) else [records], columns=FEATURES)


def predict_local(frame, model_path="artifacts/model.joblib", threshold=0.5):
    probability = joblib.load(model_path).predict_proba(frame)[:, 1]
    label = (probability >= threshold).astype(int)
    return [{"prediction": int(y), "disease_probability": float(p),
             "confidence": float(p if y else 1 - p)} for y, p in zip(label, probability)]


def predict_remote(frame, url):
    results = []
    for record in frame.astype(object).where(frame.notna(), None).to_dict("records"):
        request = urllib.request.Request(f"{url.rstrip('/')}/predict",
                                         data=json.dumps(record).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            results.append(json.loads(response.read()))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="JSON object/list or CSV with the 13 features")
    parser.add_argument("--url", help="API base URL; omit to use the local model")
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    frame = load_patients(args.input)
    if args.limit:
        frame = frame.head(args.limit)
    results = (predict_remote(frame, args.url) if args.url
               else predict_local(frame, args.model))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
