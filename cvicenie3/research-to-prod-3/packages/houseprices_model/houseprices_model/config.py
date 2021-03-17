import pathlib

import houseprices_model


PACKAGE_ROOT = pathlib.Path(houseprices_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINING_DATA_FILE = "train.csv"

MODEL_NAME = "prices_model"

# data
TESTING_DATA_FILE = "test.csv"
TRAINING_DATA_FILE = "train.csv"
TARGET = "SalePrice"


# variables
FEATURES = [
    "MSSubClass",
    "OverallQual",
    "OverallCond",
    "1stFlrSF",
    "GrLivArea",
    "BsmtFullBath",
    "Fireplaces",
    "GarageCars",
    "LotFrontage",
]

# numerical variables with NA in train set
NUMERICAL_VARS_WITH_NA = ["LotFrontage"]

# variables to log transform
NUMERICALS_LOG_VARS = ["LotFrontage", "1stFlrSF", "GrLivArea"]

NUMERICAL_NA_NOT_ALLOWED = [feature for feature in FEATURES if feature not in NUMERICAL_VARS_WITH_NA]