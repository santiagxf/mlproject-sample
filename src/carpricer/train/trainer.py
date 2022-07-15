import numpy as np
import mlflow

from types import SimpleNamespace
from typing import Dict, Any, List
from mlflow.models.signature import infer_signature
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBRegressor

import carpricer.dataprep as dataprep
from carpricer.dataprep import transformations
from carpricer.train import evaluator

def fit_and_optimize(data: np.ndarray, labels: np.ndarray, base_model: Any, param_grid: Dict[str, List[Any]], 
                     cv: int = 5, scoring_fit: str='neg_mean_squared_error') -> GridSearchCV:
    """
    Fits multiple models searching accross multiple model's configurations of hyperparameters.

    Parameters
    ----------
    data : np.ndarray
        The training dataset as it is expected to be submmited to the model.
    labels : np.ndarray
        The associated labels of the training dataset
    base_model : Any
        The base model to fit. This model needs to comply with the Scikit-Learn API.
    param_grid : Dict[str, List[Any]]
        Grid of parameter values to try.
    cv : int, optional
        Number of CV rounds to execute, by default 5
    scoring_fit : str, optional
        Optimization metric to guide and select best estimator, by default 'neg_mean_squared_error'

    Returns
    -------
    GridSearchCV
        The search results
    """
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


def train_regressor(data_path: str, params: SimpleNamespace, to_mlflow: bool = True ) -> Any:
    """
    Train and evaluate a regressor model using the input dataset and the given parameters, including
    hyper-parameter tunning. This method can log metrics, parameters and models using Mlflow.

    Parameters
    ----------
    data_path : str
        Path where the input dataset is placed.
    params : SimpleNamespace
        Training parameters and configuration.
    to_mlflow: bool
        If metrics, parameters and models should be logged with mlflow.

    Returns
    -------
    Any
        The best found ML pipeline.
    """
    X_train, y_train, X_test, y_test = dataprep.read_and_split(data_path, params.data.test_size, params.data.label)

    X_train_transformed, transforms = transformations.scale_and_encode(X_train)
    X_test_transformed = transforms.transform(X_test)  

    base_model = XGBRegressor(nthread=4, **vars(params.model.baseline))

    print(f"Looking for best combination of parameters with objective {params.model.tune.objective}. Parameters are {params.model.tune.search.to_dict().keys()}")
    search = fit_and_optimize(X_train_transformed, 
                              y_train, 
                              base_model=base_model,
                              param_grid=params.model.tune.search.to_dict(),
                              cv=params.model.tune.cv,
                              scoring_fit=params.model.tune.objective)

    print(f"Evaluating the performance of the model")
    evaluator.evaluate_search(search, plot_params_name=['learning_rate', 'max_depth'], to_mlflow = to_mlflow)

    best_model = search.best_estimator_
    metrics = evaluator.evaluate_regressor(best_model, X_test_transformed, y_test)
    model_pipeline = Pipeline(steps=[('preprocessing', transforms),
                                     ('model', best_model)])

    if to_mlflow:
        mlflow.log_params(params.model.baseline.to_dict())   
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model_pipeline, "model", signature=infer_signature(X_test, y_test.values))

    return model_pipeline
