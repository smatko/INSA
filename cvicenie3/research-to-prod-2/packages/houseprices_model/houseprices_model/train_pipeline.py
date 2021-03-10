import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

from houseprices_model import pipeline
from houseprices_model import config


def save_pipeline(pipeline_to_persist):
    save_file_name = config.MODEL_NAME
    save_path = config.TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_persist, save_path)
    print("saved pipeline")


def run_training():
    print("training model")
    data = pd.read_csv(config.DATASET_DIR / config.TRAINING_DATA_FILE)

    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES], data[config.TARGET], test_size=0.1, random_state=0
    )

    y_train = np.log(y_train)

    pipeline.price_pipe.fit(X_train[config.FEATURES], y_train)

    save_pipeline(pipeline_to_persist=pipeline.price_pipe)
    print("training finished")

if __name__ == "__main__":
    run_training()