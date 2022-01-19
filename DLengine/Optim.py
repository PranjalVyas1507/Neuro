import torch
import torch.optim as optim


import keras

def set_optimizer(framework, optim):
    if(framework == 'keras'):
        return opt_keras(optim)
    elif(framework == 'pytorch'):
        return opt_pyt(optim)



def opt_keras(optim):
    if(optim == 'SGD'):
        optimizer = keras.optimizers.SGD(learning_rate=alpha)

    elif(optim == 'Adam'):
        optimizer = keras.optimizers.Adam(learning_rate=alpha)

    elif(optim == 'Adagrad'):
        optimizer = keras.optimizers.Adagrad(learning_rate=alpha)

    elif(optim == 'RMSProp'):
        optimizer = keras.optimizers.RMSprop(learning_rate=alpha)

    elif(optim == 'Adamax'):
        optimizer = keras.optimizers.Adamax(learning_rate=alpha)

    elif(optim == 'Custom_fn'):
        pass

    return optimizer


def opt_pyt(optim):
    if(optim == 'SGD'):
        optimizer = torch.optim.SGD(model.parameters(), lr=alpha, weight_decay=0.00008)

    elif(optim == 'Adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr=alpha, weight_decay=0.00008)

    elif(optim == 'RMSProp'):
        optimizer = torch.optim.RMSprop(model.parameters(), lr=alpha, weight_decay=0.00008)

    elif(optim == 'Adagrad'):
        optimizer = torch.optim.Adagrad(model.parameters(), lr=alpha, weight_decay=0.00008)

    elif(optim == 'Adamax'):
        optimizer = torch.optim.Adamax(model.parameters(), lr=alpha, weight_decay=0.00008)

    elif(optim == 'Custom_fn'):
        pass

    return optimizer
