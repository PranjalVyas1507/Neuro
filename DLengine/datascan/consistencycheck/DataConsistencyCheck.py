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
    def count_null(self) :
        return self.df.isnull().sum()
    
    ###### remove null not needed as it shifts the individual dataseries
    ####def remove_null_value(self):
    #####    return 



    '''
    Args : self -> dataframe and key within the DDCobject,
            fill_value -> if fill value is 
            "mean" -- replace the null values of
                    the numerical columns with their mean/average 
                    with the mode values of the categorical columns
            "zero" -- replace  the null vales of numercial columns with 0.
                    With categorical columns, introduce a new category
                    "unknown class."
    '''  

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

    # self.key should not be null.
    def check_duplicate_key(self):
        if self.key is not None:
            if self.df[self.key].duplicated().any():
                return True
        

    def check_duplicate_rows(self):
        pass


    # Not Ideal. n^2    
    def check_duplicate_columns(self):
        duplicate_columns = set()
        for i in range(self.df.shape[1]):
            col = self.df.iloc[:,i]
            for j in range(i+1, self.df.shape[1]):
                next_col = self.df.iloc[:,j]
                if col.equals(next_col):
                    duplicate_columns.add(col)
        return duplicate_columns

    # self.key should not be null.
    def remove_duplicates(self):
        return self.df[self.df.duplicated(self.key, keep='last')]
    
    def del_datapoint(self):
        pass

