from houseprices_model.validation import validate_inputs
from houseprices_model.data_management import load_dataset

def test_validaton_na():
    test_data = load_dataset('train.csv')
    test_data = test_data[:2]

    assert len(test_data) == 2

    test_data['MSSubClass'] = None

    validated_data = validate_inputs(test_data)

    assert len(validated_data) < 2



    