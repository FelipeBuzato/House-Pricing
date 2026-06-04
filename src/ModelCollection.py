from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

class ModelCollection:
    def __init__(self):
        pass


    def get(self, model_name, params={}):

        if(model_name == "OLS"):
            return LinearRegression(**params)
        
        elif(model_name == "Ridge"):
            return Ridge(**params)
        
        elif(model_name == "Lasso"):
            return Lasso(**params)
        
        elif(model_name == "Random Forest"):
            return RandomForestRegressor(**params)
        
        elif(model_name == "Gradient Boosting"):
            return LGBMRegressor(**params)
        
        raise ValueError(f"Unknown model: {model_name}")