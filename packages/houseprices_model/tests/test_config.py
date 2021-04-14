from houseprices_model import config

def test_config_allowed_features():
    assert "YrSold" not in config.FEATURES


