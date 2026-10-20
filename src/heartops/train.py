import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, roc_auc_score,
                             ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     train_test_split)
from .data import acquire, FEATURES
from .eda import make_figures
from .model import build_pipeline

SEED = 42
METRICS = ["accuracy", "precision", "recall", "roc_auc"]
REGISTERED_NAME = "heartops-classifier"


def candidates():
    return {
        "logistic": (LogisticRegression(max_iter=2000, random_state=SEED),
                     {"classifier__C": [0.01, 0.1, 1.0, 10.0],
                      "classifier__class_weight": [None, "balanced"]}),
        "forest": (RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1),
                   {"classifier__max_depth": [3, 5, None],
                    "classifier__min_samples_leaf": [1, 4]})}


def evaluation_plots(y_true, probability, prefix, folder, title):
    """Confusion matrix and ROC curve; returns the saved paths."""
    prediction = (probability >= 0.5).astype(int)
    paths = [folder / f"{prefix}_confusion_matrix.png", folder / f"{prefix}_roc_curve.png"]
    ConfusionMatrixDisplay.from_predictions(y_true, prediction, cmap="Blues")
    plt.title(f"{title}: confusion matrix")
    plt.savefig(paths[0], dpi=110, bbox_inches="tight")
    plt.close("all")
    RocCurveDisplay.from_predictions(y_true, probability)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.title(f"{title}: ROC curve")
    plt.savefig(paths[1], dpi=110, bbox_inches="tight")
    plt.close("all")
    return paths


def comparison_plot(results, path):
    frame = pd.DataFrame(results).set_index("name")[[f"cv_{m}" for m in METRICS]]
    axis = frame.T.plot.bar(rot=0, figsize=(9, 5), ylim=(0.5, 1))
    axis.set(title="Model comparison (5-fold CV on training patients)", ylabel="Score")
    for container in axis.containers:
        axis.bar_label(container, fmt="%.3f", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close("all")


def main():
    data = acquire()
    output = Path("artifacts")
    output.mkdir(exist_ok=True)
    figures = Path("reports/figures")
    make_figures(data, figures)
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data.target, test_size=0.2, random_state=SEED, stratify=data.target)
    # Keep the held-out patients out of preprocessing and hyperparameter selection.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("heartops-cleveland")
    provenance = json.loads(Path("data/processed/provenance.json").read_text())
    results = []
    fitted = {}
    for name, (classifier, grid) in candidates().items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({"stage": "model-selection", "family": name,
                             "data_sha256": provenance["sha256"]})
            search = GridSearchCV(build_pipeline(classifier), grid, cv=cv,
                                  scoring={m: m for m in METRICS},
                                  refit="roc_auc", n_jobs=1, return_train_score=True)
            search.fit(X_train, y_train)
            cv_results = search.cv_results_
            # One nested child run per grid configuration so every trial is tracked.
            for index, params in enumerate(cv_results["params"]):
                with mlflow.start_run(run_name=f"{name}-trial-{index}", nested=True):
                    mlflow.log_params({k.removeprefix("classifier__"): v
                                       for k, v in params.items()})
                    mlflow.log_metrics({f"cv_{m}": float(cv_results[f"mean_test_{m}"][index])
                                        for m in METRICS})
                    mlflow.log_metric("train_roc_auc",
                                      float(cv_results["mean_train_roc_auc"][index]))
            index = search.best_index_
            scores = {f"cv_{m}": float(cv_results[f"mean_test_{m}"][index]) for m in METRICS}
            scores["cv_roc_auc_std"] = float(cv_results["std_test_roc_auc"][index])
            pd.DataFrame(cv_results).to_csv(output / f"{name}_search.csv", index=False)
            # Out-of-fold predictions give per-model plots without touching held-out data.
            oof = cross_val_predict(search.best_estimator_, X_train, y_train, cv=cv,
                                    method="predict_proba")[:, 1]
            plots = evaluation_plots(y_train, oof, f"{name}_cv", output,
                                     f"{name} (out-of-fold)")
            mlflow.log_params({**search.best_params_, "seed": SEED, "folds": 5,
                               "grid_size": len(cv_results["params"])})
            mlflow.log_metrics(scores)
            mlflow.log_artifact(str(output / f"{name}_search.csv"))
            for plot in plots:
                mlflow.log_artifact(str(plot), artifact_path="plots")
            mlflow.log_artifacts(str(figures), artifact_path="eda")
            mlflow.sklearn.log_model(search.best_estimator_, name="pipeline",
                                     input_example=X_train.head(2))
            results.append({"name": name, **scores, "params": search.best_params_})
            fitted[name] = search.best_estimator_
    comparison_plot(results, figures / "model_comparison.png")
    winner = max(results, key=lambda row: row["cv_roc_auc"])
    model = fitted[winner["name"]]
    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    metrics = {"accuracy": accuracy_score(y_test, prediction),
               "precision": precision_score(y_test, prediction, zero_division=0),
               "recall": recall_score(y_test, prediction, zero_division=0),
               "roc_auc": roc_auc_score(y_test, probability)}
    for plot in evaluation_plots(y_test, probability, "heldout", figures, "Selected model"):
        plot.replace(figures / plot.name.removeprefix("heldout_"))
    joblib.dump(model, output / "model.joblib")
    summary = {"selected": winner["name"], "train_rows": len(X_train),
               "test_rows": len(X_test), "seed": SEED, "candidates": results,
               "heldout_metrics": metrics, "threshold": 0.5,
               "features": FEATURES, "class_counts": data.target.value_counts().to_dict(),
               "data_sha256": provenance["sha256"]}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2))
    with mlflow.start_run(run_name="selected-heldout-evaluation"):
        mlflow.set_tags({"stage": "final-evaluation", "data_sha256": provenance["sha256"]})
        mlflow.log_params({"selected_model": winner["name"],
                           **{k.removeprefix("classifier__"): v
                              for k, v in winner["params"].items()}})
        mlflow.log_metrics({f"heldout_{k}": v for k, v in metrics.items()})
        mlflow.log_artifact(str(output / "metrics.json"))
        mlflow.log_artifacts(str(figures), artifact_path="figures")
        mlflow.sklearn.log_model(model, name="pipeline", input_example=X_train.head(2),
                                 registered_model_name=REGISTERED_NAME)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
