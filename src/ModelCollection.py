from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

class ModelCollection:
    def __init__(self):
        pass


    def get(self, model_name, params=None):

        if(model_name == "OLS"):
            return LinearRegression()
        
        elif(model_name == "Ridge"):
            return Ridge()
        
        elif(model_name == "Lasso"):
            return Lasso()
        
        elif(model_name == "Random Forest"):
            return RandomForestRegressor()
        
        elif(model_name == "Gradient Boosting"):
            return LGBMRegressor()
        
        raise ValueError(f"Unknown model: {model_name}")