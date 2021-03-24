import numpy as np
import pandas as pd

from houseprices_model.data_management import load_pipeline
from houseprices_model import config
from houseprices_model.validation import validate_inputs
from houseprices_model import __version__ as _version

import logging

_logger = logging.getLogger(__name__)

_price_pipe = load_pipeline()

def make_prediction(input_data):

    data = pd.read_json(input_data)
    validated_data = validate_inputs(data)
    prediction = _price_pipe.predict(validated_data[config.FEATURES])
    output = np.exp(prediction)
    response = {"predictions": output}

    _logger.info(
        f"Making predictions with model version: {_version} "
        f"Inputs: {validated_data} "
        f"Predictions: {response}"
    )

    return response