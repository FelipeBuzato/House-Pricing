from sklearn.base import BaseEstimator, TransformerMixin

class InteractionsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names_in_ = X.copy().columns.tolist()
        self.feature_names_out_ = self.transform(X.copy()).columns.tolist()
        return self
    

    def transform(self, X):
        ## Implement feature engineering here        
        X = X.copy()

        # Quality and Condition interaction:
        #X["QualLivArea"] = X["num__OverallQual"] * X["num__GrLivArea"]
        #X["QualTotalSF"] = X["num__OverallQual"] * X["num__SqFeet"]
        #X["QualBsmtSF"] = X["num__OverallQual"] * X["num__TotalBsmtSF"]
        #X["QualGarage"] = X["num__OverallQual"] * X["num__GarageArea"]

        #X["GrLivArea_OverallQual"] = X["num__OverallQual"] * X["num__GrLivArea"]
        #X["OverallQual_OverallCond"] = X["num__OverallQual"] * X["num__OverallCond"]
        #X["ExterQual_ExterCond"] = X["ord_2__ExterQual"] * X["ord_2__ExterCond"]
        #X["BsmtQual_BsmtCond"] = X["ord_1__BsmtQual"] * X["ord_1__BsmtCond"]
        #X["GarageQual_GarageCond"] = X["ord_1__GarageQual"] * X["ord_1__GarageCond"]

        return X