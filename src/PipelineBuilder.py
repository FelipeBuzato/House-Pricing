from sklearn.pipeline import Pipeline

class PipelineBuilder:
    def __init__(self, transformer, preprocessor, model):
        self.transformer = transformer
        self.preprocessor = preprocessor
        self.model = model


    def build(self):
        pipeline = Pipeline([
            ("features", self.transformer),
            ("preprocess", self.preprocessor),
            ("model", self.model)
        ])
        return pipeline