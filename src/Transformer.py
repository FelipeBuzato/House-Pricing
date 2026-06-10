from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import json5

class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, config_path):
        super().__init__()
        self.config_path = config_path
        with open(self.config_path, 'r') as file:
            self.config = json5.load(file)
    

    def fit(self, X, y=None):
        self.feature_names_in_ = X.copy().columns.tolist()
        self.feature_names_out_ = self.transform(X.copy()).columns.tolist()
        return self
    

    def transform(self, X):
        ## Implement feature engineering here        
        X = X.copy()

        # New features
        X["SqFeet"] = X["1stFlrSF"] + X["2ndFlrSF"] + X["TotalBsmtSF"]
        X["Baths"] = X["FullBath"] + X["BsmtFullBath"] + (X["HalfBath"] + X["BsmtHalfBath"]) / 2
        X["Porch"] = X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"] + X["WoodDeckSF"]
        X["Age_Sold"] = X["YrSold"] - X["YearBuilt"]
        X["Age_Remod"] = X["YrSold"] - X["YearRemodAdd"]

        # Log transforms
        log_columns = self.config["log_transforms"]
        X[log_columns] = np.log1p(X[log_columns])

        return X