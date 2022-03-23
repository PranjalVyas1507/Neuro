import pandas as pd

from pandas import DataFrame
from pandas import Series

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression, SelectPercentile


from typing import Tuple


class Correlations:
    def __init__(self, X : DataFrame, y: Series ):
        self.X = X
        self.y = y

    def sk_ANOVA(self,k=None):
        if self.X.shape[1] > 5:
            k = int(0.8*X.shape[1])
            X = SelectKBest(score_func=f_classif,k=k).fit_transform(self.X, self.y)

        return X

    def sk_pearson(self):
        if self.X.shape[1] > 5:
            k = int(0.8*X.shape[1])
            X = SelectKBest(score_func=f_regression,k=k).fit_transform(self.X, self.y)
        return X

    def sk_chi2(self):
        if self.X.shape[1] > 5:
            k = int(0.8*X.shape[1])
            X = SelectKBest(score_func=chi2,k=k).fit_transform(self.X, self.y)
        return X



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


    """
    Func: select_num_inputs
    Desc: Returns a dataframe consisting of the msot correlated input features

    Args : Self uses input df X and output df y.
    Returns : df
    """    
    def select_num_inputs(self):
        pass
