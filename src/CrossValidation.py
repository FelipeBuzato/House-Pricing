class CrossValidation:
    def __init__(self, cv_folds=5, model=None):
        self.cv_folds = cv_folds
        self.model = model


    def evaluate(self):
        ## Implement cross validation here
        pass

    
    def hyper_param_tune(self, params_grid):
        ## Implement cross validation and best param selection here
        pass