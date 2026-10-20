# Verification record

Executed on 20 October 2026, macOS (x86_64), Python 3.12.6, Docker Desktop 24.0.7, kind.

| Check | Result | Evidence |
|---|---|---|
| Data download + validation | 303 rows, 13 predictors + target, SHA-256 recorded | `data/processed/provenance.json` |
| EDA | 7 figures (histograms, correlations, class balance, densities, category rates, boxplots, missing) | `reports/figures/`, `notebooks/01_eda.ipynb` |
| Training | 14 grid configurations, 5-fold CV; logistic C=0.1 selected (CV ROC-AUC 0.9113) | `artifacts/metrics.json`, `artifacts/training.log` |
| Held-out (61 patients) | accuracy 0.8852, precision 0.8387, recall 0.9286, ROC-AUC 0.9610 | `artifacts/metrics.json` |
| MLflow | 17 runs (2 parent, 14 nested trials, 1 final); model registered `heartops-classifier` v1 | `mlflow.db`, `mlruns/`, screenshots 02–03d |
| Lint | ruff: all checks passed | screenshot 01 |
| Unit tests | 16 passed | `artifacts/tests.xml`, `artifacts/tests.log`, screenshot 01 |
| Notebooks | 01_eda, 02_training, 03_inference executed without errors | `notebooks/` |
| Docker | image built; `/ready` 200, `/predict` 200, invalid input 422 | `artifacts/docker-build.log`, `artifacts/container-prediction.json`, screenshot 04 |
| Kubernetes (kind v1.37) | Deployment 2/2 ready, Service, Ingress; `/predict` 200 via `http://localhost:8081` | `artifacts/k8s/`, screenshot 05 |
| Monitoring | Prometheus target UP; Grafana dashboard populated by ~900 requests | screenshots 07–09 |
| Report | PDF verified at exactly 10 pages; DOCX and Markdown generated | `reports/assignment_report.*` |

| GitHub Actions | Run #1 on push to main: lint, train + test, Docker smoke test, kind deploy all succeeded (7m 8s); 5 artifacts | https://github.com/Ankit-Bukshete/Heart_Disease_Mlops/actions/runs/39438018947, screenshot 10 |

Still to do: video recording of the pipeline.

On kind the LoadBalancer EXTERNAL-IP remains `<pending>`; the service is exposed via Ingress.
