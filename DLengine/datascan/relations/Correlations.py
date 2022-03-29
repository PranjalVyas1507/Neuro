import pandas as pd

from pandas import DataFrame
from pandas import Series

from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression, SelectPercentile



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


    """
    Func: select_num_inputs
    Desc: Returns a dataframe consisting of the msot correlated input features

    Args : Self uses input df X and output df y.
    Returns : df
    """    
    def k_inputs(self):
        pass
