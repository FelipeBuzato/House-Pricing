from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from PreProcessor import PreProcessor

class PreProcessorTree(PreProcessor):
    def build(self):
        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

        na_zero_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        na_none_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        ordinal_1_pipeline = Pipeline([
            ("inputer", SimpleImputer(strategy='constant', fill_value="None")),
            ("encoder", OrdinalEncoder(categories=self.ordinal_1_categories, handle_unknown="use_encoded_value", unknown_value=-1)),
        ])

        ordinal_2_pipeline = Pipeline([
            ("inputer", SimpleImputer(strategy='most_frequent')),
            ("encoder", OrdinalEncoder(categories=self.ordinal_2_categories, handle_unknown="use_encoded_value", unknown_value=-1)),
        ])

        column_transformer = ColumnTransformer([
            ("num", numerical_pipeline, self.numerical_cols),
            ("cat", categorical_pipeline, self.categorical_cols),
            ("ord_1", ordinal_1_pipeline, self.ordinal_1_cols),
            ("ord_2", ordinal_2_pipeline, self.ordinal_2_cols),
            ("na_as_none", na_none_pipeline, self.na_as_none_cols),
            ("na_as_zero", na_zero_pipeline, self.na_as_zero_cols)
        ])

        column_transformer.set_output(transform="pandas")
        return column_transformer
        