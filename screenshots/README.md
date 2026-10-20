# Screenshots

All images come from real runs on 20 Oct 2026. Browser views (MLflow, Swagger, Prometheus,
Grafana) were captured from the live services with headless Chrome. Terminal views are the
actual command output, rendered to PNG by `scripts/render_terminal.py`.

| File | Shows |
|---|---|
| `01_lint_and_unit_tests.png` | `ruff check` passes; `pytest -v` 16 passed |
| `02_mlflow_runs.png` | MLflow experiment: parent runs + 14 nested grid-trial runs |
| `02b_mlflow_compare_charts.png` | MLflow chart view comparing CV metrics across all runs |
| `03_mlflow_run_logistic.png` | Logistic run: params, CV metrics, tags (data SHA-256) |
| `03b_mlflow_artifact_plot.png` | Logged artifacts: out-of-fold ROC plot, confusion matrix, grid CSV |
| `03c_mlflow_final_run.png` | Final held-out evaluation run |
| `03d_mlflow_model_registry.png` | `heartops-classifier` v1 in the Model Registry |
| `04_docker_build_and_predict.png` | Docker build, compose services, `/predict` 200 and 422 |
| `05_kubernetes_deployment.png` | kind cluster: 2/2 pods, Service, Ingress, `/predict` via ingress |
| `06_api_swagger_docs.png` | FastAPI `/docs` |
| `07_api_request_logs_and_metrics.png` | JSON request logs and Prometheus metrics |
| `08_prometheus_targets.png` | Prometheus target `heartops` UP |
| `09_grafana_dashboard.png` | Grafana dashboard after ~900 requests |
| `10_github_actions_run.png` | GitHub Actions run #1: all 4 jobs green (7m 8s), 5 uploaded artifacts |
| `10b_github_actions_summary.png` | Cropped run summary + job graph (used in the report) |

## Still to capture yourself

1. **Video** — short recording: data download, training, MLflow comparison, `pytest`, CI run,
   `docker build` + `/predict`, `./scripts/deploy_kind.sh`, Grafana dashboard.
