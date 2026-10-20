import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from heartops.api import app
from heartops.data import CATEGORICAL, FEATURES, NUMERIC, acquire
from heartops.eda import make_figures, summary_table
from heartops.model import build_pipeline
from heartops.train import candidates

PATIENT = json.loads(Path('examples/patient.json').read_text())


def test_feature_groups_partition_features():
    assert set(NUMERIC).isdisjoint(CATEGORICAL)
    assert sorted(NUMERIC + CATEGORICAL) == sorted(FEATURES)


def test_cleaned_csv_and_provenance_written():
    acquire()
    provenance = json.loads(Path('data/processed/provenance.json').read_text())
    assert provenance['rows'] == 303 and len(provenance['sha256']) == 64
    assert Path('data/processed/heart.csv').read_text().splitlines()[0].split(',')[-1] == 'target'


def test_summary_table_reports_missing():
    table = summary_table(acquire())
    assert table.loc['ca', 'missing'] == 4 and table.loc['age', 'missing'] == 0


def test_eda_figures(tmp_path):
    names = make_figures(acquire(), tmp_path)
    for expected in ['histograms.png', 'correlations.png', 'class_balance.png',
                     'numeric_by_target.png', 'missing_values.png']:
        assert expected in names


def test_pipeline_scales_and_encodes():
    data = acquire()
    model = build_pipeline(LogisticRegression(max_iter=2000)).fit(data[FEATURES], data.target)
    transformed = model.named_steps['features'].transform(data[FEATURES])
    categories = sum(data[c].nunique() for c in CATEGORICAL)
    assert transformed.shape == (303, len(NUMERIC) + categories)
    assert not np.isnan(transformed).any()
    np.testing.assert_allclose(transformed[:, :len(NUMERIC)].mean(axis=0), 0, atol=1e-6)


@pytest.mark.parametrize('name', ['logistic', 'forest'])
def test_candidate_grids_are_valid(name):
    classifier, grid = candidates()[name]
    pipeline = build_pipeline(classifier)
    for key, values in grid.items():
        pipeline.set_params(**{key: values[0]})


def test_api_accepts_missing_optional_and_rejects_extra():
    with TestClient(app) as client:
        assert client.get('/health').json() == {'status': 'ok'}
        assert client.post('/predict', json={**PATIENT, 'ca': None, 'thal': None}).status_code == 200
        assert client.post('/predict', json={**PATIENT, 'name': 'x'}).status_code == 422
        missing = {k: v for k, v in PATIENT.items() if k != 'age'}
        assert client.post('/predict', json=missing).status_code == 422


def test_inference_script():
    if not Path('artifacts/model.joblib').exists():
        pytest.skip('train first')
    output = subprocess.run([sys.executable, 'scripts/predict.py', 'data/processed/heart.csv',
                             '--limit', '3'], capture_output=True, text=True, check=True)
    results = json.loads(output.stdout)
    assert len(results) == 3 and all(r['prediction'] in (0, 1) for r in results)


def test_trained_model_quality_gate():
    path = Path('artifacts/metrics.json')
    if not path.exists():
        pytest.skip('train first')
    metrics = json.loads(path.read_text())
    assert metrics['heldout_metrics']['roc_auc'] >= 0.80
    assert metrics['selected'] in candidates()
