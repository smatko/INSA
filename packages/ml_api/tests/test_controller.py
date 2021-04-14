from houseprices_model import config as model_config
from houseprices_model.data_management import load_dataset
from houseprices_model import __version__ as _version

import json
import math

def test_health_endpoint_returns_200(flask_test_client):
    # When
    response = flask_test_client.get('/health')

    # Then
    assert response.status_code == 200


def test_prediction_endpoint_returns_prediction(flask_test_client):
    test_data = load_dataset(file_name=model_config.TESTING_DATA_FILE)
    post_json = test_data[0:5].to_json(orient='records')

    # When
    response = flask_test_client.post('/v1/predict',
                                      json=post_json)

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    prediction = response_json['predictions']
    assert len(prediction) == 5