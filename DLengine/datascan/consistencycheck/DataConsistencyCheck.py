from pandas import DataFrame
from pandas import Series

from typing import List, Union, Optional

class DataConsistencyCheck:
    def __init__(self, df: DataFrame, key = None):
        self.df = df
        self.key = key
    
    
    '''
    Total number of null values in a data frame
    '''
    def find_null(self) :
        return self.df.isnull().sum()
    
    def fill_missing_values(self, fill_value='mean'):
        cat_value = self.df[col].mode().iloc[0]
        num_value = self.df[col].mean().iloc[0]

        if fill_value is 'zero':
            cat_value = 'unknown_class'
            num_value = 0

        else:
            pass    
        for col in self.df.columns:
                if self.df[col].dtype == 'categorical':
                    self.df[col] = self.df[col].fillna(cat_value)
                else:
                    self.df[col] = self.df[col].fillna(num_value)    

    def check_duplicate_key(self):
        pass

    def check_duplicate_columns(self):
        pass
    
    def remove_duplicates(self):
        return self.df[self.df.duplicated(self.key, keep=False)]
    
    def del_datapoint(self):
        pass

