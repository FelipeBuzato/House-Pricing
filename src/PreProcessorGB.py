from sklearn.compose import ColumnTransformer
from PreProcessor import PreProcessor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

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
    def build(self, model_name):
        
        if(model_name == "LightGBM"):
            return self.get_lightgbm_pipeline()
        elif(model_name == "XGBoost"):
            return self.get_xgb_pipeline()
        
        raise ValueError(f"Unknown Gradient Boosting model: {model_name}")


    def get_lightgbm_pipeline(self):

        categorical_cols = set(self.categorical_cols) | set(self.ordinal_1_cols) | set(self.ordinal_2_cols) | set(self.na_as_none_cols)
        numerical_cols = set(self.numerical_cols) | set(self.na_as_zero_cols)
        categorical_cols = list(categorical_cols)
        numerical_cols = list(numerical_cols)

        column_transformer = ColumnTransformer([
            ("num", "passthrough", numerical_cols),
            ("cat", CategoryCaster(categorical_cols), categorical_cols)
        ])

        column_transformer.set_output(transform="pandas")
        return column_transformer
    

    def get_xgb_pipeline(self):

        categorical_cols = self.categorical_cols + self.na_as_none_cols
        numerical_cols = self.numerical_cols + self.na_as_zero_cols
        ordinal_cols = self.ordinal_1_cols + self.ordinal_2_cols

        ord_1_categories = [categories[1:] for categories in self.ordinal_1_categories]
        ordinal_categories = ord_1_categories + self.ordinal_2_categories

        categorical_pipeline = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        ordinal_pipeline = Pipeline([
            ("encoder", OrdinalEncoder(categories=ordinal_categories, handle_unknown="use_encoded_value", unknown_value=-1)),
        ])

        column_transformer = ColumnTransformer([
            ("num", "passthrough", numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
            ("ord", ordinal_pipeline, ordinal_cols),
        ])

        column_transformer.set_output(transform="pandas")
        return column_transformer