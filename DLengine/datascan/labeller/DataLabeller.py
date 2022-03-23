import pandas as pd

from pandas import DataFrame
from pandas import Series

import numpy as np

from typing import Tuple

class DataLabeller:
    def __init__(self):
        pass

    def seperate_num_cat(X):
        cat_column = []
        num_column = []
        for column in X.columns:
            if (X[column].dtype == 'object'):
                X[column] = X[column].astype('category')
                X[column] = X[column].cat.codes
                cat_column.append(column)
            else:
                if(X[column].dtype == 'int64'):
                    if(pd.Series(X[column]).nunique()/pd.Series(X[column]).count() <0.003):
                        X[column] = X[column].astype('category')
                        X[column] = X[column].cat.codes
                        cat_column.append(column)
                    else:
                        num_column.append(column)
                else:
                    num_column.append(column)

        X_cat = X.filter(cat_column , axis=1)
        X_num = X.filter(num_column , axis=1)
        return X_cat, X_num

    def merge_features(t : Tuple[np.array, ...]):
        X_concat = np.hstack(t)
        return X_concat
        