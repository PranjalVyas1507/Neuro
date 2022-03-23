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

    def sk_ANOVA(k=None):
        pass

    def sk_pearson(self):
        pass

    def sk_chi2(self):
        pass

    def select_num_inputs(self):
        pass
