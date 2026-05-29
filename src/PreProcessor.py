import json5

class PreProcessor:
    def __init__(self, config_path, df):
        self.config_path = config_path
        with open(config_path, 'r') as file:
            self.config = json5.load(file)
        column_groups = self.split_cols(df)

        self.numerical_cols = column_groups[0]
        self.categorical_cols = column_groups[1]
        self.ordinal_1_cols = column_groups[2]
        self.ordinal_2_cols = column_groups[3]
        self.na_as_none_cols = column_groups[4]
        self.na_as_zero_cols = column_groups[5]

        self.ordinal_1_categories = [self.config["ordinal_1"][col] for col in self.ordinal_1_cols]
        self.ordinal_2_categories = [self.config["ordinal_2"][col] for col in self.ordinal_2_cols]
        

    def split_cols(self, df):
        if('SalePrice' in df.columns):
            df = df.drop(columns=['SalePrice'])

        numeric_cols = set(df.select_dtypes(include="number").columns) | set(self.config["Features_added"])
        categorical_cols = set(df.select_dtypes(include=["object", "category"]).columns)

        ordinal_1_cols = set(self.config["ordinal_1"].keys())
        ordinal_2_cols = set(self.config["ordinal_2"].keys())
        na_as_none_cols = set(self.config["na_as_none"])
        na_as_zero_cols = set(self.config["na_as_zero"])
        cols_to_drop = set(self.config["cols_to_drop"])
        special_cols = (ordinal_1_cols | ordinal_2_cols | na_as_none_cols | na_as_zero_cols)

        if(special_cols.intersection(cols_to_drop)):
            raise ValueError("Remove all cols_to_drop from special_cols config.")
        
        numerical_cols = list(numeric_cols - special_cols - cols_to_drop)
        categorical_cols = list(categorical_cols - special_cols - cols_to_drop)
        ordinal_1_cols = list(ordinal_1_cols)
        ordinal_2_cols = list(ordinal_2_cols)
        na_as_none_cols = list(na_as_none_cols)
        na_as_zero_cols = list(na_as_zero_cols)

        return numerical_cols, categorical_cols, ordinal_1_cols, ordinal_2_cols, na_as_none_cols, na_as_zero_cols