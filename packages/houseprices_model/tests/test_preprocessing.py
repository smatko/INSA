import math
import numpy as np

from houseprices_model.data_management import load_dataset
from houseprices_model.pipeline import price_pipe
from houseprices_model import config
from houseprices_model import preprocessors as pp 

from sklearn.model_selection import train_test_split

def test_pipeline_drops_unnecessary_features():
    # Given
    test_data = load_dataset('train.csv')

    X_train, X_test, y_train, y_test = train_test_split(
        test_data, test_data[config.TARGET], test_size=0.1, random_state=0
    )
    assert len(config.FEATURES) != len(X_train.columns)
    X_transformed, _ = price_pipe._fit(X_train, y_train)

    assert len(X_transformed[0]) == len(config.FEATURES)

def test_pipeline_transform_min_max_features():
    # Given
    test_data = load_dataset('train.csv')

    X_train, X_test, y_train, y_test = train_test_split(
        test_data, test_data[config.TARGET], test_size=0.1, random_state=0
    )

    X_transformed, _ = price_pipe._fit(X_train, y_train)

    for x in X_transformed:
        for v in x:
            assert 0.0 <= v <= 1.0

def test_transformer_drops_unnecessary_features():
    test_data = load_dataset('train.csv')

    X_train, X_test, y_train, y_test = train_test_split(
        test_data, test_data[config.TARGET], test_size=0.1, random_state=0
    )

    transformer = pp.KeepColumnsTransformer(
            variables=config.FEATURES,
        )

    assert len(config.FEATURES) != len(X_train.columns)
    X_transformed = transformer.transform(X_train)

    assert len(X_transformed.columns) == len(config.FEATURES)
