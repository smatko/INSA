import math

from houseprices_model.predict import make_prediction
from houseprices_model.data_management import load_dataset


def test_make_single_prediction():
    # Given
    test_data = load_dataset('test.csv')
    single_test_json = test_data[0:1].to_json(orient='records')

    # When
    subject = make_prediction(input_data=single_test_json)

    # Then
    assert subject is not None
    assert isinstance(subject.get('predictions')[0], float)
    
    print(math.ceil(subject.get('predictions')[0]))
    assert math.ceil(subject.get('predictions')[0]) == 112964

def test_make_multiple_predictions():
    # Given
    test_data = load_dataset('test.csv')
    original_data_length = len(test_data)
    multiple_test_json = test_data.to_json(orient='records')

    # When
    subject = make_prediction(input_data=multiple_test_json)

    # Then
    assert subject is not None
    assert len(subject.get('predictions')) != original_data_length