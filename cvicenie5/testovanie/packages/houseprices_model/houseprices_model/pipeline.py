from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from houseprices_model import preprocessors as pp
from houseprices_model import config


price_pipe = Pipeline(
    [
        ("keep_columns",
        pp.KeepColumnsTransformer(variables=config.FEATURES)
        ),
        (
            "numerical_inputer",
            pp.NumericalImputer(variables=config.NUMERICAL_VARS_WITH_NA),
        ),
        ("log_transformer", pp.LogTransformer(variables=config.NUMERICALS_LOG_VARS)),
        ("scaler", MinMaxScaler()),
        ("Linear_model", Lasso(alpha=0.005, random_state=0)),
    ]
)