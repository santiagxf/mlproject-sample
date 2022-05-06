from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def scale_and_encode(data: pd.DataFrame) -> Tuple[np.ndarray, ColumnTransformer]:
    """
    Scales numerical variables and encode categorical variables for a given dataset.

    Parameters
    ----------
    data : pd.DataFrame
        The data to transform

    Returns
    -------
    Tuple[np.ndarray, sklearn.compose.ColumnTransformer]
        A tuple containing the transformed data and the transformations that were applied to it.
    """
    pipe_cfg = {
        'num_cols': data.dtypes[data.dtypes == 'float64'].index.values.tolist(),
        'cat_cols': data.dtypes[data.dtypes == 'object'].index.values.tolist(),
    }

    transformations = ColumnTransformer([
        ('num_inputer', SimpleImputer(strategy='median'), pipe_cfg['num_cols']),
        ('cat_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), pipe_cfg['cat_cols'])
    ])

    data = transformations.fit_transform(data)
    return data, transformations
