from sklearn.compose import ColumnTransformer
from PreProcessor import PreProcessor
from sklearn.base import BaseEstimator, TransformerMixin

class CategoryCaster(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        super().__init__()
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for col in self.columns:
            X[col] = X[col].astype("category")

        return X

    def set_output(self, transform=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return input_features
    

class PreProcessorGB(PreProcessor):
    def build(self):

        self.categorical_cols = set(self.categorical_cols) | set(self.ordinal_1_cols) | set(self.ordinal_2_cols) | set(self.na_as_none_cols)
        self.numerical_cols = set(self.numerical_cols) | set(self.na_as_zero_cols)
        self.categorical_cols = list(self.categorical_cols)
        self.numerical_cols = list(self.numerical_cols)

        column_transformer = ColumnTransformer([
            ("num", "passthrough", self.numerical_cols),
            ("cat", CategoryCaster(self.categorical_cols), self.categorical_cols)
        ])

        column_transformer.set_output(transform="pandas")
        return column_transformer