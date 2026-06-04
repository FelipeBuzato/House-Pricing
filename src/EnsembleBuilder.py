from PipelineBuilder import PipelineBuilder
from ModelCollection import ModelCollection
from sklearn.ensemble import VotingRegressor, StackingRegressor

DEFAULT_PARAMS = {
    "OLS": {},
    "Ridge": {"alpha": 1.5},
    "Lasso": {"alpha": 0.1},
    "Random Forest": {"n_estimators": 1000, "min_samples_leaf": 5, "max_features": 0.3, "max_depth": 20},
    "Gradient Boosting": {"colsample_bytree": 0.6, "learning_rate": 0.05, "num_leaves": 15, "subsample": 1, "min_child_samples": 5}
}

SHORT_NAMES = {
    "OLS": "ols",
    "Ridge": "ridge",
    "Lasso": "lasso",
    "Random Forest": "rf",
    "Gradient Boosting": "gb"
}


class EnsembleBuilder:
    def __init__(self, config_path, df):
        self.config_path = config_path
        self.df = df
        self.pipeline_builder = PipelineBuilder(self.config_path, self.df)
        self.model_collection = ModelCollection()

    
    def get_model_pipeline(self, model_name, params="DEFAULT"):
        if(params == "DEFAULT"):
            try:
                params = DEFAULT_PARAMS[model_name]
            except: 
                raise ValueError("Model name not found.")
        
        pipeline = self.pipeline_builder.build(model_name, params)
        short_name = SHORT_NAMES[model_name]

        return (short_name, pipeline)


    def build_voting(self, model_names, model_params={}, weights=None):
        if(weights is None):
            weights = [1] * len(model_names)

        if(len(weights) != len(model_names)):
            raise ValueError("Number of weights must match number of models.")

        estimators = []
        for model_name in model_names:
            params = "DEFAULT" if(model_name not in model_params) else model_params[model_name]
            estimators.append(self.get_model_pipeline(model_name=model_name, params=params))

        return VotingRegressor(estimators=estimators, weights=weights)
    

    def build_stacking(self, model_names, model_params={}, meta_model_name=None, meta_model_params={}):
        estimators = []
        for model_name in model_names:
            params = "DEFAULT" if(model_name not in model_params) else model_params[model_name]
            estimators.append(self.get_model_pipeline(model_name=model_name, params=params))
        
        if(meta_model_name is None):
            meta_model = self.model_collection.get("Ridge")
        else: 
            meta_model = self.model_collection.get(meta_model_name, meta_model_params)
        
        return StackingRegressor(estimators=estimators, final_estimator=meta_model)
