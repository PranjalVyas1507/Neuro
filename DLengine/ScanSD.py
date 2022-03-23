### ScanSD : Scans Structured Data for Deep learning
# X = Input features
# y = output feauture/ value


import os
import numpy as np
import pandas as pd

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression, SelectPercentile
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


from typing import Tuple

def unittest_DS(X, y, key=None):
    pass
    '''
    X ---->  input features
    y ---->  output label


    Read X as dataframe
    Read Y as dataseries

    Findout duplicates in X,
    (Need a primary key for this, non repeatables)


    Findout null in X,
    Findout null in y
    
    and store everthing in a dictionary
    ''' 



def find_null(X):
    '''
    Find out the null values in the dataset using pandas
    '''
    return X.isnull().sum()

def fill_missing_data(X):
    
    for col in X.columns:
        '''
        if categorical --->>>> replace with mode
        if numerical --->>>> replace with mean

        '''
        if X[col].dtype == 'object':
            X[col] = X[col].fillna(X[col].mode().iloc[0])
        
        else:
            X[col] = X[col].fillna(X[col].mean().iloc[0]) 
    

def remove_duplicates(X,key):
    '''
    Find out the duplicates in the dataset
    '''
    return X[X.duplicated(key, keep=False)]
def del_datapoint(X,y):
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


def feature_selection(X,y):
    X_cat, X_num = seperate_num_cat(X)

    if y.dtype == 'object':
        pass

    elif y.dtype == 'int64':
        pass


def sk_ANOVA(X,y):
    if X.shape[1] > 5:
        k = int(0.8*X.shape[1])
        X = SelectKBest(score_func=f_classif,k=k).fit_transform(X, y)

    return X

def sk_pearson(X,y):
    if X.shape[1] > 5:
        k = int(0.8*X.shape[1])
        X = SelectKBest(score_func=f_regression,k=k).fit_transform(X, y)

    return X

def sk_chi2(X,y):
    if X.shape[1] > 5:
        k = int(0.8*X.shape[1])
        X = SelectKBest(score_func=chi2,k=k).fit_transform(X, y)

    return X


def numerical_feature_distribution(X,y):
    " Find out the overall distribution of datapoints over its mean"

def categorical_feature_distribution(X):
    pass
    "Distribution of categorical datapoints"

def merge_features(t : Tuple[np.array, ...]):
    X_concat = np.hstack(t)
    return X_concat

def central_tendencies(dtyp):
    pass

