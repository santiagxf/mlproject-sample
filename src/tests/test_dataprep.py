import pytest
import pandas as pd
import numpy as np

import carpricer.dataprep as dataprep
import carpricer.dataprep.transformations as txf
from sklearn.compose import ColumnTransformer

data_samples_path = '../data/car-prices/sample/automobile_prepared.csv'
data_samples = [pd.read_csv(data_samples_path)]
data_sample_target = 'lnprice'


@pytest.mark.parametrize("data", data_samples)
def test_scale_and_encode(data: pd.DataFrame):
    """ Unit test for carpricer.dataprep.transformations.scale_and_encode()
    """

    transformed, transforms = txf.scale_and_encode(data)

    # Check objects
    assert isinstance(transformed, np.ndarray)
    assert isinstance(transforms, ColumnTransformer)

    #Check shapes
    assert len(data.columns) == transformed.shape[1]
    assert len(data.index) == transformed.shape[0]

    # Check columns types
    transformed_df = pd.DataFrame(transformed)
    assert len(transformed_df.dtypes[transformed_df.dtypes == 'object'].index.values) == 0


def test_read():
    """ Unit test for carpricer.dataprep.transformations.scale_and_encode()
    """
    test_split = 0.3
    X_train, y_train, X_test, y_test = dataprep.read_and_split(data_samples_path, test_split, data_sample_target)

    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray) 
    assert isinstance(X_test, np.ndarray) 
    assert isinstance(y_test, np.ndarray) 
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert X_train.shape[1] == X_test.shape[1]
    
    if len(y_train) > 0:
        assert (len(y_test)/len(y_train)) ==  pytest.approx(test_split, abs=0.15)
