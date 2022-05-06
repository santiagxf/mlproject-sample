from typing import Any, List, Dict
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV

import mlflow
import sklearn_evaluation
import numpy as np
import matplotlib.pyplot as plt


def evaluate_search(search: GridSearchCV, plot_params_name: List[str], subset_params_name: List[str] = None):
    """
    Evaluates a given parameters search by plotting the results in 2D plots, varying the different values
    on each axis. If the search contains more than 2 parameters, then one 2D graph is generated for each
    combination of additional parameters.

    Parameters
    ----------
    search : GridSearchCV
        The results of the parameter tunning
    plot_params_name : List[str]
        A list of parameters to plot in the graph. Exactly two values should be provided.
    subset_params_name : List[str], optional
        If provided, one 2D graph will be provided for each of the indicated parameters values combinations.
        Otherwise, one 2D graph will be generated for each combination of parameters that was used in the
        search (besides the one indicated on `plot_params_name`)
    """
    if len(plot_params_name) != 2:
        ValueError("Exactly two parameters have to be indicated for the evaluation (they are displayed on a grid).")

    for param in plot_params_name:
        if param not in search.cv_results_['params'][0].keys():
            ValueError(f"Parameter {param} wasn't used in the parameters search.")

    if subset_params_name:
        # Filter dictionary by the provided params
        graph_by_params = dict(filter(lambda item: item[0] in subset_params_name and len(item[1]) > 1, search.param_grid.items()))
    else:
        # Filter dictionary by the parameters not used on each plot
        graph_by_params = dict(filter(lambda item: item[0] not in plot_params_name and len(item[1]) > 1, search.param_grid.items()))

    # Gettings parameters and its values
    graph_by_params_values = graph_by_params.values()
    graph_by_params_keys = list(graph_by_params.keys())
    graph_by_params_dim = len(graph_by_params_keys)

    if graph_by_params_dim == 0:
        fig = plt.figure()
        ax = fig.add_subplot()
        sklearn_evaluation.plot.grid_search(search.cv_results_, 
                                            change=(tuple(plot_params_name)), 
                                            ax=ax)
        mlflow.log_figure(fig, f"{plot_params_name[0]}_{plot_params_name[1]}.png")
        ax.plot()
    else:
        for values in np.array(np.meshgrid(*graph_by_params_values)).T.reshape(-1, graph_by_params_dim):
            subset = { graph_by_params_keys[i]: [values[i]] for i in range(0, graph_by_params_dim) }

            fig = plt.figure()
            ax = fig.add_subplot()
            ax.set_title(', '.join([f"{item[0]} {item[1]}" for item in subset.items()]))
            sklearn_evaluation.plot.grid_search(search.cv_results_, 
                                                change=(tuple(plot_params_name)), 
                                                subset=subset, 
                                                ax=ax)
            mlflow.log_figure(fig, f"{plot_params_name[0]}_{plot_params_name[1]}_{'_'.join(values)}.png")
            ax.plot()

def evaluate_regressor(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluates the performance of a given model using regression metrics

    Parameters
    ----------
    model : Any
        Model to evaluate
    X_test : np.ndarray
        Input dataset to use for the evaluation.
    y_test : np.ndarray
        Corresponding true values for the evaluation dataset.

    Returns
    -------
    Dict[str, float]
        A dictionary of metrics
    """
    predictions = model.predict(X_test)

    metrics = {
        "eval_MAE": mean_absolute_error(predictions, y_test),
        "eval_MSE": mean_squared_error(predictions, y_test),
        "eval_R2": r2_score(predictions, y_test)
    }

    return metrics