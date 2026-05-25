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
        return self
    

    def transform(self, X):
        ## Implement feature engineering here        
        X = X.copy()

        # Log transforms
        log_columns = self.config["log_transforms"]
        X[log_columns] = np.log1p(X[log_columns])

        return X