# Assignment task map

| Task (marks) | Implementation | Evidence |
|---|---|---|
| 1. Acquisition / EDA (5) | `src/heartops/data.py`, `eda.py`, `scripts/download_data.py`, `notebooks/01_eda.ipynb` | raw + cleaned CSV, provenance, 7 EDA figures |
| 2. Features / models (8) | `model.py` pipeline, `train.py` (LogReg + RandomForest grids, 5-fold CV), `notebooks/02_training.ipynb` | search CSVs, `metrics.json`, comparison/ROC/confusion plots |
| 3. Tracking (5) | MLflow parent + nested trial runs, plots/artifacts per run, model registry | screenshots 02–03d |
| 4. Packaging (7) | joblib + MLflow model, pinned `requirements.txt`, `requirements-lock.txt`, seeds, data hash | serialization test, `artifacts/model.joblib` |
| 5. CI/CD (8) | `.github/workflows/ci.yml` (lint → train/test → container → kind deploy), `tests/` (16) | screenshot 01; push for Actions screenshot |
| 6. Container (5) | `Dockerfile`, `compose.yaml`, FastAPI `/predict` | screenshots 04, 06 |
| 7. Deployment (7) | `deploy/api.yaml`, `deploy/ingress.yaml`, `scripts/deploy_kind.sh` | `artifacts/k8s/`, screenshot 05 |
| 8. Monitoring (3) | JSON request logs, Prometheus metrics, provisioned Grafana dashboard | screenshots 07–09 |
| 9. Report (2) | `scripts/build_report.py` → 10-page PDF/DOCX/MD, architecture diagram | `reports/` |
