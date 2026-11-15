from fastapi.testclient import TestClient

from app.main import app

SAMPLE = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0,
    "ca": 0, "thal": 1,
}


def test_endpoints():
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"

        info = client.get("/info").json()
        assert info["n_features"] == 13

        r = client.post("/predict", json=SAMPLE)
        assert r.status_code == 200
        assert isinstance(r.json()["heart_disease"], bool)

        bad = {**SAMPLE, "age": -5}
        assert client.post("/predict", json=bad).status_code == 422
