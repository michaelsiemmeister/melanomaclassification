
from melanoma_classification.utils.load_config import load_config


def test_load_config():
    conf_dict = load_config('example.config.yaml')
    assert isinstance(conf_dict, dict)
