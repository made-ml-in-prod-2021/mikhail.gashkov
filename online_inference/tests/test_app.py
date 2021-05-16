from fastapi.testclient import TestClient

from ..maincode import app, GREETING_MESSAGE


client = TestClient(app)


def test_base_endpoint():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == GREETING_MESSAGE


def test_predict_endpoint():
    response = client.get('/predict')
    assert response.status_code == 200
