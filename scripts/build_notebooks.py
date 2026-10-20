"""Create and execute the EDA, training and inference notebooks (run from repo root)."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell as code, new_markdown_cell as md, new_notebook

SETUP = code("import os, pathlib\n"
             "if pathlib.Path.cwd().name == 'notebooks':\n    os.chdir('..')\n"
             "import warnings; warnings.filterwarnings('ignore')\n"
             "from IPython.display import Image, display")

NOTEBOOKS = {
    "01_eda.ipynb": [
        md("# 01 · Data acquisition & EDA\nDownloads the original UCI Cleveland file, "
           "validates it, and produces descriptive figures. EDA never drives model selection."),
        SETUP,
        code("from heartops.data import acquire, FEATURES\n"
             "from heartops.eda import make_figures, summary_table\n"
             "data = acquire()\ndata.head()"),
        md("## Shape, types and missing values\n`ca` and `thal` carry `?` markers in the raw "
           "file. They stay as NaN in the cleaned CSV and are imputed *inside* CV folds."),
        code("print(data.shape)\nsummary_table(data)"),
        code("data.describe().T.round(2)"),
        md("## Class balance"),
        code("data.target.value_counts(normalize=True).rename({0: 'absent', 1: 'present'})"),
        code("names = make_figures(data)\nfor name in ['class_balance', 'missing_values', "
             "'histograms', 'numeric_by_target', 'boxplots', 'categorical_rates', 'correlations']:\n"
             "    display(Image(f'reports/figures/{name}.png'))"),
        md("## Observations\n- Classes are roughly balanced (54% / 46%), so accuracy is "
           "meaningful but ROC-AUC is used for selection.\n- `thalach` is lower and `oldpeak` "
           "higher for diseased patients; `cp=4` (asymptomatic) and `thal=7` show high disease "
           "rates.\n- `chol` and `trestbps` separate the classes weakly.\n- Only 6 missing "
           "values in total (ca=4, thal=2) → simple median/mode imputation is sufficient."),
        code("data.corr()['target'].drop('target').sort_values(key=abs, ascending=False).round(3)"),
    ],
    "02_training.ipynb": [
        md("# 02 · Feature engineering, training & experiment review\n"
           "Training is run by `python -m heartops.train` (also executed in CI). "
           "This notebook inspects the pipeline and the MLflow-tracked results."),
        SETUP,
        code("from sklearn.linear_model import LogisticRegression\n"
             "from heartops.model import build_pipeline\nbuild_pipeline(LogisticRegression())"),
        code("from heartops.train import candidates\n"
             "{name: grid for name, (_, grid) in candidates().items()}"),
        md("## Grid-search results (5-fold stratified CV, training patients only)"),
        code("import pandas as pd\ncols = ['params', 'mean_test_roc_auc', 'std_test_roc_auc', "
             "'mean_test_accuracy', 'mean_test_recall', 'rank_test_roc_auc']\n"
             "for name in ['logistic', 'forest']:\n"
             "    print(name)\n"
             "    display(pd.read_csv(f'artifacts/{name}_search.csv')[cols]"
             ".sort_values('rank_test_roc_auc'))"),
        code("display(Image('reports/figures/model_comparison.png'))\n"
             "for name in ['logistic', 'forest']:\n"
             "    display(Image(f'artifacts/{name}_cv_roc_curve.png', width=380))"),
        md("## MLflow runs"),
        code("import mlflow\nmlflow.set_tracking_uri('sqlite:///mlflow.db')\n"
             "runs = mlflow.search_runs(experiment_names=['heartops-cleveland'])\n"
             "runs[[c for c in runs.columns if c in ('tags.mlflow.runName', 'metrics.cv_roc_auc', "
             "'metrics.cv_accuracy', 'metrics.heldout_roc_auc', 'metrics.heldout_accuracy')]]"
             ".sort_values('tags.mlflow.runName')"),
        md("## Selected model on the held-out 20%"),
        code("import json\nm = json.load(open('artifacts/metrics.json'))\n"
             "print('selected:', m['selected'], m['candidates'][0]['params'] if "
             "m['selected']=='logistic' else m['candidates'][1]['params'])\n"
             "pd.Series(m['heldout_metrics']).round(4)"),
        code("display(Image('reports/figures/confusion_matrix.png', width=380))\n"
             "display(Image('reports/figures/roc_curve.png', width=380))"),
    ],
    "03_inference.ipynb": [
        md("# 03 · Inference\nLoad the packaged pipeline (preprocessing + model) and score "
           "new patients. Same artifact that the Docker image serves."),
        SETUP,
        code("import json, joblib, pandas as pd\nfrom heartops.data import FEATURES\n"
             "model = joblib.load('artifacts/model.joblib')\n"
             "patient = json.load(open('examples/patient.json'))\npatient"),
        code("p = model.predict_proba(pd.DataFrame([patient], columns=FEATURES))[0, 1]\n"
             "{'prediction': int(p >= .5), 'disease_probability': round(p, 4), "
             "'confidence': round(max(p, 1 - p), 4)}"),
        md("Missing `ca`/`thal` are imputed by the saved pipeline:"),
        code("model.predict_proba(pd.DataFrame([{**patient, 'ca': None, 'thal': None}], "
             "columns=FEATURES, dtype=float))[0, 1]"),
        md("## Batch scoring with the CLI"),
        code("import subprocess, sys\nprint(subprocess.run([sys.executable, 'scripts/predict.py', "
             "'data/processed/heart.csv', '--limit', '3'], capture_output=True, text=True).stdout)"),
        md("## Calling the deployed API\nWith the container (`docker compose up`) or the "
           "Kubernetes ingress running:\n```bash\ncurl -H 'Content-Type: application/json' "
           "--data @examples/patient.json http://localhost:8000/predict\n"
           "python scripts/predict.py examples/patient.json --url http://localhost:8081\n```"),
    ],
}

for name, cells in NOTEBOOKS.items():
    notebook = new_notebook(cells=cells, metadata={"kernelspec": {
        "name": "python3", "display_name": "Python 3", "language": "python"}})
    NotebookClient(notebook, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": "."}}).execute()
    nbformat.write(notebook, Path("notebooks") / name)
    print("executed", name)
