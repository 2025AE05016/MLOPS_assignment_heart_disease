# HeartOps — UCI Heart Disease MLOps assignment

End-to-end heart-disease classifier: reproducible data acquisition, EDA, tuned scikit-learn
pipelines, MLflow tracking and registry, pytest, a four-stage GitHub Actions pipeline,
Docker, Kubernetes (kind / Minikube / Docker Desktop) with Ingress, and Prometheus + Grafana.

Repository: https://github.com/2025AE05016/MLOPS_assignment_heart_disease
Student: Ankit Bukshete, BITS ID: 2025AE05016.
Report: [reports/assignment_report.pdf](reports/assignment_report.pdf) (10 pages, also DOCX/MD).

![architecture](reports/figures/architecture.png)

## Repository layout

| Path | Contents |
|---|---|
| `src/heartops/` | `data.py` (download/clean), `eda.py`, `model.py` (pipeline), `train.py` (CV + MLflow), `api.py` (FastAPI) |
| `notebooks/` | `01_eda`, `02_training`, `03_inference` (executed, outputs saved) |
| `scripts/` | `download_data.py`, `predict.py` (inference CLI), `deploy_kind.sh`, report/notebook/diagram builders |
| `data/` | `raw/processed.cleveland.data`, cleaned `processed/heart.csv`, `provenance.json` |
| `tests/` | 16 pytest unit tests (data, EDA, pipeline, API, inference, quality gate) |
| `.github/workflows/ci.yml` | lint → train + test → Docker smoke test → kind deploy |
| `deploy/` | `api.yaml` (Deployment + LoadBalancer Service), `ingress.yaml`, `kind-config.yaml`, `monitoring/` |
| `artifacts/` | `model.joblib`, `metrics.json`, grid-search CSVs, plots, logs, `k8s/` deployment evidence |
| `screenshots/` | Evidence screenshots (see `screenshots/README.md`) |

## 1. Clean setup (Python 3.12)

```bash
python3 -m venv .venv && source .venv/bin/activate
make install              # pip install -r requirements.txt && pip install -e . --no-deps
make data                 # download UCI archive -> data/processed/heart.csv
make train                # EDA figures, 2 model grids, MLflow runs, artifacts/model.joblib
make lint test            # ruff + pytest
make notebooks report     # execute notebooks, build 10-page PDF/DOCX
```

`requirements.txt` pins direct dependencies; `requirements-lock.txt` is the full frozen
environment used for the recorded run.

## 2. Data and modelling

Source: [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease),
DOI 10.24432/C52P4X, CC BY 4.0 — Cleveland subset, 303 patients, 13 predictors + target.
`?` markers become NaN (ca: 4, thal: 2) and are imputed **inside CV folds** to avoid
leakage; diagnosis 1–4 → 1. Original codes are kept (cp 1–4, slope 1–3, thal 3/6/7).

Pipeline: median impute + `StandardScaler` (numeric) and mode impute + `OneHotEncoder`
(categorical) → classifier. Stratified 80/20 split (seed 42); 5-fold stratified CV on the
training patients tunes Logistic Regression (C × class_weight, 8 configs) and Random Forest
(max_depth × min_samples_leaf, 6 configs). Highest mean CV ROC-AUC wins; the held-out set
is scored once. Current result: **logistic (C=0.1)**, CV ROC-AUC 0.911; held-out accuracy
0.885, precision 0.839, recall 0.929, ROC-AUC 0.961 (`artifacts/metrics.json`).

## 3. Experiment tracking (MLflow)

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001   # port 5000 is AirPlay on macOS
```

Experiment `heartops-cleveland`: a parent run per model family (params, CV metrics, grid CSV,
out-of-fold ROC/confusion plots, EDA figures, model), a nested child run for each of the 14
grid configurations, and `selected-heldout-evaluation` with held-out metrics. The winner is
registered as **`heartops-classifier`** in the Model Registry.

## 4. Inference

```bash
python scripts/predict.py examples/patient.json                     # local model
python scripts/predict.py data/processed/heart.csv --limit 5         # batch
python scripts/predict.py examples/patient.json --url http://localhost:8081   # deployed API
```

## 5. API and container

```bash
make serve                       # or: docker build -t heartops:local . && docker run -p 8000:8000 heartops:local
curl -fsS http://localhost:8000/ready
curl -fsS -H 'Content-Type: application/json' --data @examples/patient.json http://localhost:8000/predict
# {"prediction":0,"disease_probability":0.38,"confidence":0.62,"threshold":0.5}
```

Endpoints: `POST /predict`, `GET /health`, `GET /ready`, `GET /metrics`, `GET /docs`.
Invalid input → HTTP 422. Confidence is the probability of the predicted class (uncalibrated).
The image runs as non-root UID 10001. Only load trusted joblib models.

## 6. Kubernetes deployment

**kind (used for the recorded deployment, also run in CI):**

```bash
docker build -t heartops:local .
./scripts/deploy_kind.sh        # creates cluster, ingress-nginx, deploys, verifies /predict
curl -H 'Content-Type: application/json' --data @examples/patient.json http://localhost:8081/predict
kind delete cluster --name heartops   # cleanup
```

Access is through the Ingress on `http://localhost:8081`. On kind the LoadBalancer
`EXTERNAL-IP` stays `<pending>` (no cloud controller); it is assigned on Docker Desktop
Kubernetes, with `minikube tunnel`, or on GKE/EKS/AKS.

**Minikube:** `eval "$(minikube docker-env)" && docker build -t heartops:local . &&
kubectl apply -f deploy/api.yaml && minikube service heartops --url`.
**Cloud:** push the image to a registry and replace `image:` with its digest.

## 7. Monitoring and logging

```bash
docker compose up -d --build    # API :8000, Prometheus :9090, Grafana :3000
```

Each request is logged as JSON (request id, route, status, seconds; no patient data).
Prometheus scrapes `/metrics`: `heartops_requests_total`, `heartops_request_seconds`,
`heartops_predictions_total{prediction}`, `heartops_disease_probability`. Grafana
(anonymous read-only for this local demo; admin/admin to edit) auto-provisions the
**HeartOps API** dashboard: request rate, 4xx/5xx rate, p50/p95 latency, predictions per class,
mean predicted probability and target health. Pods carry `prometheus.io/*` scrape annotations.

## 8. CI/CD

`.github/workflows/ci.yml` runs on push/PR as four dependent jobs, each failing fast with logs:

1. **Lint** — `ruff check` (GitHub annotations)
2. **Train + test** — download data, train with MLflow, `pytest -v` (JUnit XML), job summary
   with test counts and held-out metrics, build report; uploads model, MLflow store, reports
3. **Container** — `docker build`, `/ready` + `/predict` smoke test, 422 check; uploads logs
4. **Deploy (kind)** — ephemeral cluster, `scripts/deploy_kind.sh`, uploads k8s evidence

Not a clinical tool. Limitations: 303 historical patients, no external validation,
calibration, drift monitoring or API authentication.
