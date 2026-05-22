from sklearn.linear_model import LinearRegression

class ModelCollection:
    def __init__(self):
        pass


    def get(self, model_name, params=None):

        if(model_name == "OLS"):
            return LinearRegression()
        
        raise ValueError(f"Unknown model: {model_name}")
        
