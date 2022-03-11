'''
The purpose of this file is to add an introspection model to
aid with model debugging and remidiation

Include SHAP as well as LIME values and visualizaiton
'''

import shap

def shap_explainer(model, X_train, y_train):
    explainer = shap.KernelExplainer(model)
    shap_values = explainer.shap_values(X_train)
    #shap.summary_plot(shap_values, X_train) 


def causalmodel_SD():
    '''
    This function is to create a causal model for Structured Data
    '''
    pass

def Bayesian_Model():
    pass 