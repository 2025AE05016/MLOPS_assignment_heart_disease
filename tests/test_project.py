import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from heartops.api import app
from heartops.data import FEATURES, acquire
from heartops.model import build_pipeline


def test_dataset():
    data = acquire()
    assert data.shape == (303, 14)
    assert set(data.target.unique()) == {0, 1}
    assert data.ca.isna().sum() == 4
    assert data.thal.isna().sum() == 2


def test_preprocessing_and_serialization(tmp_path):
    import joblib
    data = acquire()
    model = build_pipeline(LogisticRegression(max_iter=2000))
    model.fit(data[FEATURES], data.target)
    sample = data[FEATURES].iloc[:2].copy()
    sample.loc[sample.index[0], 'ca'] = np.nan
    sample.loc[sample.index[0], 'cp'] = 99
    expected = model.predict_proba(sample)
    assert np.isfinite(expected).all()
    joblib.dump(model, tmp_path / 'model.joblib')
    np.testing.assert_allclose(joblib.load(tmp_path / 'model.joblib').predict_proba(sample), expected)
    assert model.named_steps['features'].named_transformers_['numeric'].named_steps['impute'].statistics_[0] == data.age.median()


def test_api():
    patient = json.loads(Path('examples/patient.json').read_text())
    with TestClient(app) as client:
        assert client.get('/ready').status_code == 200
        response = client.post('/predict', json=patient)
        assert response.status_code == 200
        result = response.json()
        assert result['prediction'] in (0, 1)
        assert 0 <= result['disease_probability'] <= 1
        assert 0.5 <= result['confidence'] <= 1
        patient['age'] = -1
        assert client.post('/predict', json=patient).status_code == 422
        body = client.get('/metrics').text
        assert 'heartops_requests_total' in body and 'heartops_predictions_total' in body


@pytest.mark.parametrize('field,value', [('thal', 2), ('cp', 0), ('chol', float('inf'))])
def test_invalid_codes(field, value):
    patient = json.loads(Path('examples/patient.json').read_text())
    patient[field] = value
    # Infinity is not JSON, so use a valid JSON representation for numeric overflow.
    body = json.dumps(patient).replace('Infinity', '1e999')
    with TestClient(app) as client:
        assert client.post('/predict', content=body, headers={'Content-Type': 'application/json'}).status_code == 422
