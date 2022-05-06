from types import SimpleNamespace
from typing import Dict, Any
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBRegressor

import carpricer.dataprep as dataprep
from carpricer.dataprep import transformations
from carpricer.train import evaluator

import pandas as pd
import numpy as np
import mlflow

from mlflow.models.signature import infer_signature


def fit_and_optimize(data: np.ndarray, labels: np.ndarray, base_model: Any, param_grid: Dict[str, Any], 
                     cv: int = 5, scoring_fit: str='neg_mean_squared_error'):
    gs = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid, 
        cv=cv, 
        n_jobs=-1, 
        scoring=scoring_fit,
        verbose=2
    )
    fitted_model = gs.fit(data, labels)
    
    return fitted_model


def train_regressor(input_dataset: str, params: SimpleNamespace):
    X_train, y_train, X_test, y_test = dataprep.train_test_split(input_dataset, params.data.test_size, params.data.label)

    X_train_transformed, transforms = transformations.scale_and_encode(X_train)
    X_test_transformed = transforms.transform(X_test)  

    base_model = XGBRegressor(silent=True, nthread=4, **vars(params.model.baseline))
    mlflow.log_params(vars(params.model.baseline))   

    print(f"Looking for best combination of parameters with objective {params.model.tune.objective}. Parameters are {params.model.tune.search.keys()}")
    search = fit_and_optimize(X_train_transformed, 
                              y_train, 
                              base_model=base_model,
                              param_grid=params.model.tune.search,
                              cv=params.model.tune.cv,
                              scoring_fit=params.model.tune.objective)

    print(f"Evaluating the performance of the model")
    evaluator.evaluate_search(search, plot_params_name=['learning_rate', 'max_depth'])

    best_model = search.best_estimator_
    mlflow.log_params(search.best_params_)

    metrics = evaluator.evaluate_regressor(best_model, X_test_transformed, y_test)
    mlflow.log_metrics(metrics)

    model_pipeline = Pipeline(steps=[('preprocessing', transforms),
                                     ('model', best_model)])

    signature = infer_signature(X_test, y_test)
    mlflow.sklearn.log_model("model", model_pipeline, signature=signature)

    return model_pipeline
