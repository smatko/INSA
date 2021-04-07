import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

from houseprices_model import config
from houseprices_model import __version__ as _version

import logging

_logger = logging.getLogger(__name__)

def load_dataset(file_name):
    _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    return _data

def save_pipeline(pipeline_to_persist):
    save_file_name = config.MODEL_NAME + _version + ".pkl"
    save_path = config.TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(save_file_name)
    
    joblib.dump(pipeline_to_persist, save_path)
    _logger.info("saved pipeline")

def load_pipeline():
    load_file_name = config.MODEL_NAME + _version + ".pkl"
    file_path = config.TRAINED_MODEL_DIR / load_file_name
    saved_pipeline = joblib.load(filename=file_path)
    return saved_pipeline

def remove_old_pipelines(files_to_keep):
    for model_file in config.TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in [files_to_keep, "__init__.py"]:
            model_file.unlink()