from fastapi.testclient import TestClient
from main_api import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]
    
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "online"}

def test_predict_success():
    response = client.post("/predict", json={"feature_1": 10.0, "feature_2": 20.0})
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["prediction_score"] > 0

def test_predict_fail():
    response = client.post("/predict", json={"feature_1": -5.0, "feature_2": 20.0})
    assert response.status_code == 400
    assert "positive" in response.json()["detail"]
