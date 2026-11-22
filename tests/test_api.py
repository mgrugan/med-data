"""API tests against the committed models (skipped if the pipeline has not been run)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "manifest.json").exists(), reason="models not trained")

PATIENT = dict(age=72, gender="Male", admission_type="Emergency", admission_source="Emergency Room",
               medical_condition="Circulatory", medication="Insulin only", a1c_result=">8%", glucose_serum="Not tested",
               prior_inpatient=1, prior_emergency=0, prior_outpatient=0, days_hospitalized=2)


@pytest.fixture(scope="module")
def client():
    from backend.app.main import app
    return TestClient(app)


def test_health_and_options(client):
    assert client.get("/api/health").json()["status"] == "ok"
    o = client.get("/api/options").json()
    assert o["display_model"] in o["models"] and 0.9 in o["levels"] and o["los_max"] == 14


def test_predict_remaining_is_clamped_and_interval_ordered(client):
    for d in [0, 2, 30]:
        r = client.post("/api/predict", json={**PATIENT, "days_hospitalized": d}).json()
        p = r["prediction"]
        assert p["remaining"] >= 0 and p["remaining_interval"][0] >= 0
        assert p["interval"][0] <= p["total_los"] <= p["interval"][1]
        assert "may occur" in r["language"]
        assert "discharged" not in r["language"].split("not a discharge date")[0].lower()


def test_predict_rejects_invalid_category(client):
    assert client.post("/api/predict", json={**PATIENT, "gender": "Other"}).status_code == 422
    assert client.post("/api/predict", json={**PATIENT, "a1c_result": "9.1%"}).status_code == 422


def test_served_predictions_match_training_models():
    """Inference re-implementations (JSON GLM, native boosters, NumPy transformer) must reproduce the
    exact predictions of the fitted training objects."""
    from backend.app.ml.predictor import Predictor
    from training.data import clean, load_raw, make_splits
    pred = Predictor(ROOT / "models")
    test = make_splits(clean(load_raw())[0], persist=False).test.sample(25, random_state=0)
    for key in ["gamma_glm", "xgboost", "xgboost_aft", "catboost", "condition_median"]:
        m = joblib.load(ROOT / "models" / f"{key}.joblib")
        served = [pred.predict_one(key, test.iloc[[i]][list(test.columns)]) for i in range(len(test))]
        np.testing.assert_allclose(served, m.predict(test), rtol=1e-4, atol=1e-3, err_msg=key)


def test_validation_report_served(client):
    r = client.get("/api/validation").json()
    assert r["selection"]["selection_uses_test_set"] is False
    assert {c["id"] for c in r["quality_gate"]["checks"]} == {
        "beats_baseline", "models_distinguishable", "shuffled_target", "interval_width", "calibration"}


def test_recovery_endpoints(client):
    ex = client.post("/api/recovery/extract", json={"text": "Take ibuprofen 400 mg every 6 hours for 3 days.", "use_llm": False}).json()
    assert ex["plan"]["medications"][0]["dose"] == "400 mg"
    s = client.post("/api/recovery/schedule", json={"plan": ex["plan"], "start_date": "2026-01-01", "view_days": 5}).json()
    assert len(s["days"][0]["timed"]) == 4 and len(s["days"][3]["timed"]) == 0
    up = client.post("/api/recovery/upload", files={"file": ("orders.txt", b"Walk 10 minutes twice daily.", "text/plain")})
    assert up.status_code == 200 and up.json()["plan"]["activity"]
