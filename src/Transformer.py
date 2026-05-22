from sklearn.base import BaseEstimator, TransformerMixin

class Transformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    

    def transform(self, X):
        ## Implement feature engineering here
        return X