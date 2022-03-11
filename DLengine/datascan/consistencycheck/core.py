from pandas import DataFrame
from pandas import Series

from typing import List, Union, Optional

class DataConsistencyCheck:
    def __init__(self, df: DataFrame, key = None):
        self.df = df
        self.key = key
    
    def find_null(self) :
        return self.df.isnull().sum()
    
    def fill_missing_values(self):
        for col in self.df.columns:
            if self.df[col].dtype == 'categorical':
                self.df[col] = self.df[col].fillna(self.df[col].mode().iloc[0])
            else:
                self.df[col] = self.df[col].fillna(self.df[col].mean().iloc[0])