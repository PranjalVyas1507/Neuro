import torch
import torch.optim as optim


import keras
import keras



def find_framework(fm):
    if(fm == 'keras'):
        return opt_keras()
    elif(fm == 'pytorch')
        return opt_pyt()



def opt_keras():
    if(parameters['optimization']== 'SGD'):
        optimizer = keras.optimizers.SGD(learning_rate=alpha)

    elif(parameters['optimization']== 'Adam'):
        optimizer = keras.optimizers.Adam(learning_rate=alpha)

    elif(parameters['optimization']== 'Adagrad'):
        optimizer = keras.optimizers.Adagrad(learning_rate=alpha)

    elif(parameters['optimization']== 'RMSProp'):
        optimizer = keras.optimizers.RMSprop(learning_rate=alpha)

    elif(parameters['optimization']== 'Adamax'):
        optimizer = keras.optimizers.Adamax(learning_rate=alpha)

    elif(parameters['optimization']== 'Custom_fn'):
        pass


def opt_pyt():
    pass
