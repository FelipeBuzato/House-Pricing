from Transformer import Transformer
from InteractionsTransformer import InteractionsTransformer
from ModelCollection import ModelCollection
from sklearn.pipeline import Pipeline
from PreProcessorLinear import PreProcessorLinear
from PreProcessorTree import PreProcessorTree
from PreProcessorGB import PreProcessorGB

class PipelineBuilder:
    def __init__(self, config_path, df):
        self.config_path = config_path
        self.df = df


    def build(self, model_name, params={}):
        transformer = Transformer(self.config_path)
        interactions_transformer = InteractionsTransformer()
        model_collection = ModelCollection()
        model = model_collection.get(model_name, params)

        # linear models pipeline
        if(model_name in ["OLS", "Ridge", "Lasso"]):
            preprocessor = self.get_linear_preprocessor()
            pipeline = Pipeline([
                ("features", transformer),
                ("preprocess", preprocessor),
                ("interactions", interactions_transformer),
                ("model", model)
            ])

        # Random Forest pipeline
        elif(model_name in ["Random Forest"]):
            preprocessor = self.get_tree_preprocessor()
            pipeline = Pipeline([
                ("features", transformer),
                ("preprocess", preprocessor),
                ("model", model)
            ])

        # Gradient Boosting pipeline
        elif(model_name in ["LightGBM", "XGBoost"]):
            preprocessor = self.get_grad_boost_preprocessor(model_name)
            pipeline = Pipeline([
                ("features", transformer),
                ("preprocess", preprocessor),
                ("model", model)
            ])

        else: raise ValueError("Model name not found.")
    
        return pipeline
    

    def get_linear_preprocessor(self):
        preprocessor = PreProcessorLinear(self.config_path, self.df)
        return preprocessor.build()


    def get_tree_preprocessor(self):
        preprocessor = PreProcessorTree(self.config_path, self.df)
        return preprocessor.build()


    def get_grad_boost_preprocessor(self, model_name):
        preprocessor = PreProcessorGB(self.config_path, self.df)
        return preprocessor.build(model_name)