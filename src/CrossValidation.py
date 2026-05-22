from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
import numpy as np

class CrossValidation:
    def __init__(self, cv_folds=5, pipeline=None):
        self.cv_folds = cv_folds
        self.pipeline = pipeline
        self.kfold = KFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=42  # 'None' to shuffle the splits across every call
        )


    def evaluate(self, X, y):
        ## Implement cross validation here

        if(self.pipeline is None):
            raise ValueError("Define a pipeline before running cross-validation")
        
        scores = cross_val_score(
            self.pipeline,
            X,
            y,
            cv=self.kfold,
            scoring="neg_mean_squared_error",
            #n_jobs=-2  # to parallelize calculations across folds
        )

        scores_stats = {"mean_mse": -np.mean(scores), "std_mse": np.std(scores)}

        return scores_stats


    def hyper_param_tune(self, X, y, param_grid):
        ## Implement cross validation and best param selection here

        if(self.pipeline is None):
            raise ValueError("Define a pipeline before running cross-validation")
        
        search = GridSearchCV(
                    self.pipeline,
                    param_grid=param_grid,
                    cv=self.kfold,
                    scoring="neg_mean_squared_error",
                    refit=False
                    #n_jobs=-2  # to parallelize calculations across folds
        )
        search.fit(X, y)

        return search
