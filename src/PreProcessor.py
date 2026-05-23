from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
import json

class PreProcessor:
    def __init__(self, df):
        with open('config_preprocess.json', 'r') as file:
            self.config = json.load(file)
        column_groups = self.split_cols(df)

        self.numerical_cols = column_groups[0]
        self.categorical_cols = column_groups[1]
        self.ordinal_1_cols = column_groups[2]
        self.ordinal_2_cols = column_groups[3]
        self.na_as_none_cols = column_groups[4]
        self.num_as_zero_cols = column_groups[5]

        self.ordinal_1_categories = list(self.config["ordinal_1"].values())
        self.ordinal_2_categories = list(self.config["ordinal_2"].values())
        

    def split_cols(self, df):
        numeric_cols = set(df.select_dtypes(include="number").columns)
        categorical_cols = set(df.select_dtypes(include=["object", "category"]).columns)

        ordinal_1_cols = set(self.config["ordinal_1"].keys())
        ordinal_2_cols = set(self.config["ordinal_2"].keys())
        na_as_none_cols = set(self.config["na_as_none"])
        num_as_zero_cols = set(self.config["num_as_zero"])
        special_cols = (ordinal_1_cols | ordinal_2_cols | na_as_none_cols | num_as_zero_cols)
        
        numerical_cols = list(numeric_cols - special_cols)
        categorical_cols = list(categorical_cols - special_cols)
        ordinal_1_cols = list(ordinal_1_cols)
        ordinal_2_cols = list(ordinal_2_cols)
        na_as_none_cols = list(na_as_none_cols)
        num_as_zero_cols = list(num_as_zero_cols)

        return numerical_cols, categorical_cols, ordinal_1_cols, ordinal_2_cols, na_as_none_cols, num_as_zero_cols

    # Change handle unknown strategy!!!
    def build(self):
        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        num_as_zero_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        na_none_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        ordinal_1_pipeline = Pipeline([
            ("inputer", SimpleImputer(strategy='constant', fill_value="None")),
            ("encoder", OrdinalEncoder(categories=self.ordinal_1_categories, handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scaler", StandardScaler())
        ])

        ordinal_2_pipeline = Pipeline([
            ("inputer", SimpleImputer(strategy='most_frequent')),
            ("encoder", OrdinalEncoder(categories=self.ordinal_2_categories, handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scaler", StandardScaler())
        ])

        column_transformer = ColumnTransformer([
            ("num", numerical_pipeline, self.numerical_cols),
            ("cat", categorical_pipeline, self.categorical_cols),
            ("ord_1", ordinal_1_pipeline, self.ordinal_1_cols),
            ("ord_2", ordinal_2_pipeline, self.ordinal_2_cols),
            ("non_exist", na_none_pipeline, self.na_as_none_cols),
            ("num_as_zero", num_as_zero_pipeline, self.num_as_zero_cols)
        ])

        return column_transformer