import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Categorical data missing value imputer."""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y = None):
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X):
        """Apply the transforms to the dataframe."""

        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna('Missing')

        return X
