import pandas as pd
from sklearn.model_selection import train_test_split

def read_and_split(data_path:str, test_size: float, label: str):
    """
    Loads a dataset indicated in the path and splits it in train and test splits.

    Parameters
    ----------
    data_path : str
        Path where the data is located.
    test_size : float
        Proportion to be used for testing
    label : str
        Label column name

    Returns
    -------
    np.ndarray
        4 arrays containing train predictors, train labels, test predictor and test labels; in that order.
    """
    df = pd.read_csv(data_path).dropna()
    train, test = train_test_split(df, test_size=test_size)

    X_train = train.drop([label], axis=1)
    y_train = train[label].to_numpy()

    X_test = test.drop([label], axis=1)
    y_test = test[label].to_numpy()

    return X_train, y_train, X_test, y_test