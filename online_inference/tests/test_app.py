import pytest
from fastapi.testclient import TestClient

from online_inference.maincode import app, GREETING_MESSAGE
from online_inference.maincode.data_model import INPUT_FEATURES_LIST

OK_DATA_X = [[32,1,3,157,553,1,0,116,1,1.9264,1,4,2],
             [76,0,1,99,531,1,2,176,1,1.70557507,0,4,3],
             [26,0,3,145,257,1,2,162,1,3.61,2,0,1]]
OK_DATA_Y = [0, 0, 0]

client = TestClient(app)


def test_base_endpoint():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {'text': GREETING_MESSAGE}


def test_predict_bad_request():
    response = client.post('/predict',
                           json={
                               'blablabla': []
                           })
    assert response.status_code == 400


def test_predict_good_data():
    response = client.get('/predict',
                          json={'features': INPUT_FEATURES_LIST,
                                'data': OK_DATA_X
                                })
    assert response.status_code == 200
    response_data = response.json()
    assert 'target' in response_data
    assert response_data['target'] == OK_DATA_Y

