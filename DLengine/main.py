import sys, os
import json
import numpy as np
import time
#import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense

#from keras.models import Sequential
#from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


#import mxnet as mx
#from mxnet import nd, autograd, gluon
#from mxnet.gluon import nn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.autograd import Variable

from statistics import  mean



#bert import
import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer


import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

loss_stats = {
"loss": [],
"val_loss": [],
"accuracy" : [],
"val_accuracy" : [],
"y_train_inv": [],
"y_test_inv": [],
"y_pred_inv" : [],
"confusion_matrix" : [],
"classification_report" : []
}

w_n_b = {
 "layers" : [],
 "weights" : [],
 "biases" : [],
 "U" : [],
 "biases_hh" : []
}

input_file = ""

file_type = ""

y_transformer = RobustScaler()
f_transformer = RobustScaler()


def coder(parameters):

    #Necessary Parameters :
    """
                parameters[2] = Learning Rate / alpha
                parameters[3] = Train-Test split
                parameters[4] = Optimizers
                parameters[5] = Mini batch size
                parameters[6] = layers
                parameters[7] = Neurons per layer
                parameters[8] =
                parameters[10] = Input Parameters
                parameters[11] = Output Target
                parameters[12] = Train-Val Split
                parameters[13] = Dropouts for each layers

    """
    alpha = parameters['learning_rate']
    layers = int(parameters['layers'])
    layers_1 = int(parameters['layers']) + 1
    neurons = parameters['neurons']
    activationfunction = parameters['activation']
    dropouts = parameters['dropouts']
    target = parameters['target'].strip()
    global file_type
    try:
        code = open('DL_code.py', 'x')
        code.truncate(0)
        #py_out = open("DL_code.py", "a")
        code_string = ''
        with open('DL_code.py', 'a') as py_out:
                    #py_out.truncate(0)
                    if(parameters['framework']=='Keras'):
                        if(parameters['type']=='Classification'):
                            #Import Necessary libraries
                            py_out.write("import numpy as np\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.metrics import confusion_matrix\n\n\nimport keras\nfrom keras.models import Sequential\nfrom keras.layers import Dense\nfrom keras.layers import Dropout\n")

                            #Data Preprocessing
                            if(file_type == "excel"):
                                py_out.write("\ninput_frame = pd.read_excel("+str(parameters['filename'])+")\ntarget = \"" +target+"\"")
                            elif(file_type == "csv"):
                                py_out.write("\ninput_frame = pd.read_csv("+str(parameters['filename'])+")\ntarget = \"" +target+"\"")
                            elif(file_type == "json"):
                                py_out.write("\ninput_frame = pd.read_json("+str(parameters['filename'])+")\ntarget = \"" +target+"\"")

                            py_out.write("\nX = input_frame.drop(target, axis = 1)\ny = input_frame[target]\ncolumn_list =" +str(parameters['headers']))
                            py_out.write("\nX = X.filter(column_list, axis=1)")
                            #py_out.write("\nprint(\"Input file\")")
                            py_out.write("\nprint(X)")
                            #py_out.write("\nprint(\"Target\")")
                            py_out.write("\nprint(y)")
                            py_out.write("\nfor column in column_list:\n\tif(X[column].dtype == 'object'):\n\t\tX[column] = X[column].astype('category')\n\t\tX[column] = X[column].cat.codes\nif(y.dtype == 'object'):\n\ty = y.astype('category')\n\ty = y.cat.codes")
                            py_out.write("\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size = " + str(parameters['testsplit']) + ", random_state = 2)\nsc = StandardScaler()\nX_train = sc.fit_transform(X_train)\nX_test = sc.transform(X_test)")


                            #Deep Learning Model
                            if(parameters['optimization']== 'SGD'):
                                py_out.write("\noptimizer = keras.optimizers.SGD(learning_rate="+ str(alpha)+")\n")

                            elif(parameters['optimization']== 'Adam'):
                                py_out.write("\noptimizer = keras.optimizers.Adam(learning_rate="+ str(alpha)+")\n")

                            elif(parameters['optimization']== 'Adagrad'):
                                py_out.write("\noptimizer = keras.optimizers.Adagrad(learning_rate="+ str(alpha) +")\n")

                            elif(parameters['optimization']== 'RMSProp'):
                                py_out.write("\noptimizer = keras.optimizers.RMSprop(learning_rate="+ str(alpha) +")\n")

                            elif(parameters['optimization']== 'Adamax'):
                                py_out.write("\noptimizer = keras.optimizers.Adamax(learning_rate="+ str(alpha) +")\n")


                            py_out.write("classifier = Sequential()\n")

                            for i in range(layers+1):
                                if(i == 0):
                                    py_out.write("classifier.add(Dense(units ="+str(neurons[i]) +", kernel_initializer = 'uniform', activation = 'relu', input_dim = X_test.shape[1]))\n")
                                    py_out.write("classifier.add(Dropout("+str(dropouts[i])+"))\n")
                                elif(i == layers):
                                    py_out.write("classifier.add(Dense(units ="+str(neurons[i]) +", kernel_initializer = 'uniform', activation = 'sigmoid'))\n")
                                else:
                                    py_out.write("classifier.add(Dense(units ="+str(neurons[i]) +", kernel_initializer = 'uniform', activation = 'relu'))\n")
                                    py_out.write("classifier.add(Dropout("+str(dropouts[i])+"))\n")
                            # Model Training and evaluation

                            py_out.write("classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])\n")
                            py_out.write("history = classifier.fit(X_train, y_train, batch_size = " +str(parameters['batch_size'])+", epochs = "+str(parameters['epochs'])+", validation_split="+str(parameters['validsplit'])+")\n")
                            py_out.write("y_pred = classifier.predict(X_test)\ny_pred = (y_pred > 0.5)\ncm = confusion_matrix(y_test, y_pred)")
                            py_out.write("\nprint(\'confusion matrix:\')")
                            py_out.write("\nprint(cm)")

                        if(parameters['type']=='Time Series'):
                            #Import Necessary libraries
                            py_out.write("import numpy as np\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import RobustScaler\n\nimport keras\nfrom keras.models import Sequential\nfrom keras.layers import LSTM\nfrom keras.layers import Dropout\nfrom keras.layers import Dense\n")

                            #data_preprocessing
                            py_out.write("def create_dataset(X, y, time_steps=1):\n\tXs, ys = [], []\n\tfor i in range(len(X) - time_steps):\n\t\tv = X.iloc[i:(i + time_steps)].values\n\t\tXs.append(v)\n\t\tys.append(y.iloc[i + time_steps])\n\treturn np.array(Xs), np.array(ys)")
                            if(file_type == "excel"):
                                py_out.write("\ninput_frame = pd.read_excel("+str(parameters['filename'])+")\ntarget = \"" +target+"\"")
                            elif(file_type == "csv"):
                                py_out.write("\ninput_frame = pd.read_csv("+str(parameters['filename'])+")\ntarget = \"" +target+"\"")
                            elif(file_type == "json"):
                                py_out.write("\ninput_frame = pd.read_json("+str(parameters['filename'])+")\ntarget = \"" +target+"\"")
                            #py_out.write("\n
                            py_out.write("\ntarget = " + "\'" +target.strip()+ "\'")
                            py_out.write("\nX = input_frame\ny = input_frame[target]\ncolumn_list =" +str(parameters['headers']))
                            py_out.write("\nX = X.filter(column_list, axis=1)\n") #change this
                            py_out.write("for column in column_list:\n\tif(X[column].dtype == 'object'):\n\t\tX[column] = X[column].astype('category')\n\t\tX[column] = X[column].cat.codes\n")
                            py_out.write("test_size = int(len(X) * "+str(parameters['testsplit'])+")\ntrain_size = len(X) - test_size\ntrain, test = X.iloc[0:train_size], X.iloc[train_size:len(X)]\n")
                            py_out.write("y_transformer = RobustScaler()\nf_transformer = RobustScaler()\n")
                            py_out.write("y_transformer = y_transformer.fit(train[[target]])\ny_trn = y_transformer.transform(train[[target]])\ny_tst = y_transformer.transform(test[[target]])\nf_transformer = f_transformer.fit(train[column_list].to_numpy())\n")
                            py_out.write("train.loc[:, column_list] = f_transformer.transform(train[column_list].to_numpy())\ntest.loc[:, column_list] = f_transformer.transform(test[column_list].to_numpy())\ntime_steps = 10\ny_trn = pd.DataFrame(y_trn)\ny_tst = pd.DataFrame(y_tst)\nX_train, y_train = create_dataset(train, y_trn, time_steps)\nX_test, y_test = create_dataset(test, y_tst, time_steps)\n")

                            #Deep Learning Model
                            if(parameters['optimization']== 'SGD'):
                                py_out.write("optimizer = keras.optimizers.SGD(learning_rate="+ str(alpha)+")\n")

                            elif(parameters['optimization']== 'Adam'):
                                py_out.write("optimizer = keras.optimizers.Adam(learning_rate="+ str(alpha)+")\n")

                            elif(parameters['optimization']== 'Adagrad'):
                                py_out.write("optimizer = keras.optimizers.Adagrad(learning_rate="+ str(alpha)+")\n")

                            elif(parameters['optimization']== 'RMSProp'):
                                py_out.write("optimizer = keras.optimizers.RMSprop(learning_rate="+ str(alpha)+")\n")

                            elif(parameters['optimization']== 'Adamax'):
                                py_out.write("optimizer = keras.optimizers.Adamax(learning_rate="+ str(alpha)+")\n")

                            py_out.write("regressor = Sequential()\n")
                            for i in range(layers):
                                if(i == 0):
                                    if(layers-1 == 0):
                                        py_out.write("regressor.add(LSTM(units = "+str(neurons[i])+",return_sequences = False, input_shape = (X_train.shape[1], X_train.shape[2])))\n")
                                    else:
                                        py_out.write("regressor.add(LSTM(units = "+str(neurons[i])+",return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))\n")
                                elif(i == layers-1):
                                    py_out.write("regressor.add(LSTM(units = "+str(neurons[i])+", return_sequences = False))\n")
                                    py_out.write("regressor.add(Dropout("+str(dropouts[i])+"))\n")

                                else:
                                    # Adding remaining hidden layers
                                    py_out.write("regressor.add(LSTM(units = "+str(neurons[i])+",return_sequences = True ))\n")
                            #Model Training and evaluation
                            py_out.write("regressor.add(Dense(units = 1))\n")
                            py_out.write("regressor.compile(optimizer = optimizer, loss = 'mean_squared_error')\n")
                            py_out.write("history = regressor.fit(X_train, y_train, validation_split="+str(parameters['validsplit'])+", epochs="+str(parameters['epochs'])+", batch_size =" +str(parameters['batch_size'])+")\n")
                            py_out.write("y_pred = regressor.predict(X_test)\ny_train_inv = y_transformer.inverse_transform(y_train.reshape(1, -1))\ny_test_inv = y_transformer.inverse_transform(y_test.reshape(1, -1))\ny_pred_inv = y_transformer.inverse_transform(y_pred)\n")

                    if(parameters['framework']=='PyTorch'):
                        #Import Libraries
                        code_string += "import numpy as np\nimport pandas as pd\n"
                        code_string += "\nimport torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.utils.data import Dataset, DataLoader\nfrom torch.autograd import Variable\n"

                        if(parameters['type']=='Classification'):
                            layers_1 = layers_1 - 1
                            code_string += ("\nclass Dataset(Dataset):")

                            code_string += ("\n\tdef __init__(self, X_data, y_data):")
                            code_string += ("\n\t\tself.X_data = X_data")
                            code_string += ("\n\t\tself.y_data = y_data")

                            code_string += ("\n\tdef __getitem__(self, index):")
                            code_string += ("\n\t\treturn self.X_data[index], self.y_data[index]")

                            code_string += ("\n\tdef __len__ (self):")
                            code_string += ("\n\t\treturn len(self.X_data)")


                            code_string += ("\nfrom sklearn.preprocessing import StandardScaler\n")
                            code_string += ("\nfrom sklearn.model_selection import train_test_split\n")
                            #code_string += ("from sklearn.preprocessing import RobustScaler
                            code_string += ("from sklearn.metrics import confusion_matrix\n")


                            #Data Preprocessing
                            if(file_type == "excel"):
                                code_string +=("\ninput_frame = pd.read_excel("+str(parameters['filename'])+")\ntarget = \"" +target+"\"")
                            elif(file_type == "csv"):
                                code_string +=("\ninput_frame = pd.read_csv("+str(parameters['filename'])+")\ntarget = \"" +target+"\"")
                            elif(file_type == "json"):
                                code_string +=("\ninput_frame = pd.read_json("+str(parameters['filename'])+")\ntarget = \"" +target+"\"")

                            #code_string += ("input_frame = pd.read_json(file)\ntarget = \'" +target.strip()+"\'")
                            code_string += ("\nX = input_frame.drop(target, axis = 1)\ny = input_frame[target]\ncolumn_list =" +str(parameters['headers']))
                            code_string += ("\nX = X.filter(column_list, axis=1)\n")
                            code_string += ("for column in column_list:\n\tif(X[column].dtype == 'object'):\n\t\tX[column] = X[column].astype('category')\n\t\tX[column] = X[column].cat.codes\nif(y.dtype == 'object'):\n\ty = y.astype('category')\n\ty = y.cat.codes")
                            code_string += ("\nX_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size = " + str(parameters['testsplit']) + ", random_state = 2)\n")
                            code_string += "\nX_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size="+ str(parameters['validsplit'])+", random_state = 0)\n"
                            code_string += "sc = StandardScaler()\nX_train = sc.fit_transform(X_train)\nX_val = sc.transform(X_val)\nX_test = sc.transform(X_test)\n\nX_train , y_train = np.array(X_train), np.array(y_train)\nX_val, y_val = np.array(X_val), np.array(y_val)\nX_test, y_test = np.array(X_test), np.array(y_test)"


                            #Deep Learning model
                            code_string +="\nEPOCHS = "+str(parameters['epochs'])+"\n"
                            code_string +=("\nBATCH_SIZE = " + str(parameters['batch_size']))

                            code_string +="\ntrain_dataset = Dataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())\n"
                            code_string +="\nval_dataset = Dataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())"
                            code_string +="\ntest_dataset = Dataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())"

                            code_string +="\ntrain_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)"
                            code_string +="\nval_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)"
                            code_string +="\ntest_loader = DataLoader(dataset=test_dataset, batch_size=1)"

                            code_string +="\nlayers = []"
                            code_string +="\nlayers.append(nn.Linear(X_train.shape[1],"+str(neurons[0])+",bias=True))"

                            for i in range(layers_1):
                                #if(i == 0):
                                if(i == layers_1-1):
                                    code_string +="\nlayers.append(nn.Linear("+str(neurons[i])+",1,bias=True))"
                                    code_string +="\nlayers.append(nn.Sigmoid())"
                                else:
                                    code_string +="\nlayers.append(nn.Linear("+str(neurons[i])+","+str(neurons[i+1])+",bias=True))"
                                    code_string +="\nlayers.append(nn.Dropout("+str(dropouts[i])+"))"

                            code_string +="\nmodel = nn.Sequential(*layers)"
                            code_string +="\nlayers.clear()"

                            #Tuning, Training and validation
                            code_string +="\ndevice = torch.device(\"cuda:0\" if torch.cuda.is_available() else \"cpu\")"
                            code_string +="\nloss_func = nn.BCELoss()"

                            if(parameters['optimization']== 'SGD'):
                                code_string +="\noptimizer = torch.optim.SGD(model.parameters(), lr="+str(alpha)+", weight_decay=0.00008)"

                            elif(parameters['optimization']== 'Adam'):
                                code_string +="\noptimizer = torch.optim.Adam(model.parameters(), lr="+str(alpha)+", weight_decay=0.00008)"

                            elif(parameters['optimization']== 'RMSProp'):
                                code_string +="\noptimizer = torch.optim.RMSprop(model.parameters(), lr="+str(alpha)+", weight_decay=0.00008)"

                            elif(parameters['optimization']== 'Adagrad'):
                                code_string +="\noptimizer = torch.optim.Adagrad(model.parameters(), lr="+str(alpha)+", weight_decay=0.00008)"

                            elif(parameters['optimization']== 'Adamax'):
                                code_string +="\noptimizer = torch.optim.Adamax(model.parameters(), lr="+str(alpha)+", weight_decay=0.00008)"

                            code_string +="\ny_pred = []"
                            code_string +="\ny_actual = []"
                            code_string +="\nfor e in range(EPOCHS):"
                            code_string +="\n\ttrain_epoch_loss = 0"
                            code_string +="\n\tval_epoch_loss = 0"
                            code_string +="\n\ttrain_epoch_acc = 0"
                            code_string +="\n\tval_epoch_acc = 0"

                            code_string +="\n\tmodel.train()"
                            code_string +="\n\tfor X_train_batch, y_train_batch in train_loader:"
                            code_string +="\n\t\tX_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)"
                            code_string +="\n\t\ty_train_pred = model(X_train_batch)"
                            code_string +="\n\t\toptimizer.zero_grad()"
                            code_string +="\n\t\ty_actual.append(y_train_batch.unsqueeze(1))"
                            code_string +="\n\t\ttrain_loss = loss_func(y_train_pred, y_train_batch.unsqueeze(1))"
                            code_string +="\n\t\ttrain_loss.backward()"
                            code_string +="\n\t\toptimizer.step()"
                            code_string +="\n\t\ty_pred_tag = (y_train_pred > 0.5).float()"
                            code_string +="\n\t\tacc = ((y_pred_tag == y_train_batch.unsqueeze(1)).sum().float())/y_train_batch.shape[0]"
                            code_string +="\n\t\ttrain_epoch_loss += train_loss.item()"
                            code_string +="\n\t\ttrain_epoch_acc += acc.item()"
                            code_string +="\n\twith torch.no_grad():"
                            code_string +="\n\t\tmodel.eval()"
                            code_string +="\n\t\tfor X_val_batch, y_val_batch in val_loader:"
                            code_string +="\n\t\t\tX_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)"
                            code_string +="\n\t\t\ty_val_pred = model(X_val_batch)"
                            code_string +="\n\t\t\tval_loss = loss_func(y_val_pred, y_val_batch.unsqueeze(1))"
                            code_string +="\n\t\t\tval_epoch_loss += val_loss.item()"
                            code_string +="\n\t\t\ty_pred_tag = (y_val_pred > 0.5).float()"
                            code_string +="\n\t\t\tacc = ((y_pred_tag == y_val_batch.unsqueeze(1)).sum().float())/y_val_batch.shape[0]"
                            code_string +="\n\t\t\tval_epoch_acc += acc.item()"
                            code_string +="\n\tprint(train_epoch_loss/len(train_loader))"
                            code_string +="\n\tprint(val_epoch_loss/len(val_loader))"
                            code_string +="\n\tprint(train_epoch_acc/len(train_loader))"
                            code_string +="\n\tprint(val_epoch_acc/len(val_loader))"

                            # Test-set Prediction
                            code_string +="\ny_pred_list = []"
                            code_string +="\ny_test_list = []"
                            code_string +="\nmodel.eval()"
                            code_string +="\nfor X_test_batch, y_test_batch in test_loader:"
                            code_string +="\n\tX_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)"
                            code_string +="\n\ty_pred = model(X_test_batch)"
                            code_string +="\n\ty_pred_tag = (y_pred > 0.5).float()"
                            code_string +="\n\ty_pred_list.append(y_pred)"
                            code_string +="\n\ty_test_list.append(y_test_batch)"
                            code_string +="\ny_pred_list = [a.squeeze().tolist() for a in y_pred_list ]"
                            code_string +="\ny_pred_list = [round(a) for a in y_pred_list]"
                            code_string +="\ny_test_list = [a.squeeze().tolist() for a in y_test_list ]"

                            code_string +="\ncm = confusion_matrix(y_test_list, y_pred_list)"
                            code_string +="\nprint(cm)"

                            py_out.write(code_string)

                        if(parameters['type']=='Time Series'):
                                layers_1 = layers_1 - 1
                                #code_string += ("\nfrom sklearn.preprocessing import StandardScaler\n")
                                code_string += ("\nfrom sklearn.preprocessing import RobustScaler\n")
                                #code_string += ("from sklearn.metrics import confusion_matrix\n")
                                code_string += ("\nclass Dataset(Dataset):")

                                code_string += ("\n\tdef __init__(self, X_data, y_data):")
                                code_string += ("\n\t\tself.X_data = X_data")
                                code_string += ("\n\t\tself.y_data = y_data")

                                code_string += ("\n\tdef __getitem__(self, index):")
                                code_string += ("\n\t\treturn self.X_data[index], self.y_data[index]")

                                code_string += ("\n\tdef __len__ (self):")
                                code_string += ("\n\t\treturn len(self.X_data)")
                                code_string+= "\ndef create_dataset(X, y, time_steps=1):"
                                code_string+= "\n\tXs, ys = [], []"
                                code_string+= "\n\tfor i in range(len(X) - time_steps):"
                                code_string+= "\n\t\tv = X.iloc[i:(i + time_steps)].values"
                                code_string+= "\n\t\tXs.append(v)"
                                code_string+= "\n\t\tys.append(y.iloc[i + time_steps])"
                                code_string+= "\n\treturn np.array(Xs), np.array(ys)"

                                code_string+=""
                                code_string+= "\nclass Regressor_LSTM(nn.Module):"
                                code_string+="\n\tdef __init__(self,input_dim, seq_len):"
                                code_string+="\n\t\t"+"super(Regressor_LSTM, self).__init__()"
                                code_string+="\n\t\tself.input_dim = input_dim"
                                code_string+="\n\t\tself.seq_length = seq_len"
                                code_string+="\n\t\tself.IP_Layer = nn.LSTM(input_size=self.input_dim,hidden_size="+str(neurons[0])+",dropout="+str(dropouts[0])+")"
                                code_string+="\n\t\tself.Out_Layer = nn.Linear(" +str(neurons[layers_1-1])+",1,bias=True)"

                                code_string+="\n\tdef forward(self,X, batch):"
                                for i in range(layers_1):
                                    code_string+="\n\t\th_x_"+str(i)+" = Variable(torch.zeros(1, batch," + str(neurons[i]) + "))"
                                    code_string+="\n\t\tc_x_"+str(i)+" = Variable(torch.zeros(1, batch," + str(neurons[i]) + "))"
                                    if(i==0):
                                        code_string+="\n\t\tout"+str(i)+", (h_x_"+str(i+1)+", c_x_"+str(i+1)+") = self.IP_Layer(X.view(self.seq_length,len(X),-1), (h_x_"+str(i)+",c_x_"+str(i)+"))"
                                    else:
                                        code_string+="\n\t\tself.regress"+str(i)+" = nn.LSTM(input_size="+str(neurons[i-1])+",hidden_size="+str(neurons[i])+",dropout="+str(dropouts[i])+")"
                                        code_string+="\n\t\tout"+str(i)+", (h_x_"+str(i+1)+", c_x_"+str(i+1)+") = self.regress"+str(i)+"(out"+str(i-1)+",(h_x_"+str(i)+", c_x_"+str(i)+"))"
                                code_string+="\n\t\tout = self.Out_Layer(out"+str(layers_1-1)+"[-1].view(batch,-1))"

                                code_string+="\n\t\treturn out.view(-1)\n"

                                #Data Preprocessing
                                if(file_type == "excel"):
                                    code_string +=("\ninput_frame = pd.read_excel("+str(parameters['filename'])+")\ntarget = \"" +target+"\"")
                                elif(file_type == "csv"):
                                    code_string +=("\ninput_frame = pd.read_csv("+str(parameters['filename'])+")\ntarget = \"" +target+"\"")
                                elif(file_type == "json"):
                                    code_string +=("\ninput_frame = pd.read_json("+str(parameters['filename'])+")\ntarget = \"" +target+"\"")
                                    code_string += ("\nX = input_frame\ny = input_frame[target]\ncolumn_list =" +str(parameters['headers']))
                                code_string += ("\nX = X.filter(column_list, axis=1)")

                                code_string += ("\ntest_size = int(len(X) *"+str(parameters['testsplit'])+")")
                                code_string += ("\ntrain_size = len(X) - test_size")
                                code_string += ("\ntrain, test = X.iloc[0:train_size], X.iloc[train_size:len(X)]")
                                code_string += ("\nval_size = int(train_size *"+str(parameters['validsplit'])+")")
                                code_string += ("\nval, train = train.iloc[0:val_size], train.iloc[val_size:train_size]")

                                code_string += ("\ny_transformer = RobustScaler()")
                                code_string += ("\nf_transformer = RobustScaler()")
                                code_string += ("\ny_transformer = y_transformer.fit(train[[target]])")
                                code_string += ("\ny_trn = y_transformer.transform(train[[target]])")
                                code_string += ("\ny_tst = y_transformer.transform(test[[target]])")
                                code_string += ("\ny_val = y_transformer.transform(val[[target]])")
                                code_string += ("\nf_transformer = f_transformer.fit(train[column_list].to_numpy())")
                                code_string += ("\ntrain.loc[:, column_list] = f_transformer.transform(train[column_list].to_numpy())")
                                code_string += ("\ntest.loc[:, column_list] = f_transformer.transform(test[column_list].to_numpy())")
                                code_string += ("\nval.loc[:, column_list] = f_transformer.transform(val[column_list].to_numpy())")
                                code_string += ("\ny_train_inv = y_transformer.inverse_transform(y_val.reshape(1,-1)).tolist()")
                                code_string += ("\nytrain_inv = y_transformer.inverse_transform(y_trn.reshape(1,-1)).tolist()")
                                code_string += ("\ny_train_inv[0].extend(ytrain_inv[0])")
                                code_string += ("\ny_test_inv = y_transformer.inverse_transform(y_tst.reshape(1,-1)).tolist()")

                                code_string += ("\ntime_steps = 10")
                                code_string += ("\ny_trn = pd.DataFrame(y_trn)")
                                code_string += ("\ny_tst = pd.DataFrame(y_tst)")
                                code_string += ("\ny_val = pd.DataFrame(y_val)")
                                code_string += ("\nX_train, y_train = create_dataset(train, y_trn, time_steps)")
                                code_string += ("\nX_test, y_test = create_dataset(test, y_tst, time_steps)")
                                code_string += ("\nX_val, y_val = create_dataset(val, y_val, time_steps)")

                                code_string += ("\nEPOCHS = "+str(parameters['epochs']))
                                code_string += ("\nBATCH_SIZE = "+str(parameters['batch_size']))

                                code_string += ("\ntrain_dataset = Dataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())")
                                code_string += ("\nval_dataset = Dataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())")
                                code_string += ("\ntest_dataset = Dataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())")

                                code_string += ("\ntrain_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)")
                                code_string += ("\nval_loader = DataLoader(dataset=val_dataset, batch_size=16, drop_last=True)")
                                code_string += ("\ntest_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False )")

                                code_string += ("\ndevice = torch.device(\"cuda:0\" if torch.cuda.is_available() else \"cpu\")")
                                code_string += ("\nloss_func = nn.MSELoss()")
                                code_string += ("\nregressor_model = Regressor_LSTM(X_train.shape[2], 10)")

                                if(parameters['optimization']== 'SGD'):
                                    code_string += ("\noptimizer = torch.optim.SGD(regressor_model.parameters(), lr="+str(alpha)+", weight_decay=0.00008)")

                                elif(parameters['optimization']== 'Adam'):
                                    code_string += ("\noptimizer = torch.optim.Adam(regressor_model.parameters(), lr="+str(alpha)+", weight_decay=0.00008)")

                                elif(parameters['optimization']== 'RMSProp'):
                                    code_string += ("\noptimizer = torch.optim.RMSprop(regressor_model.parameters(), lr="+str(alpha)+", weight_decay=0.00008)")

                                elif(parameters['optimization']== 'Adagrad'):
                                    code_string +=("\noptimizer = torch.optim.Adagrad(regressor_model.parameters(), lr="+str(alpha)+", weight_decay=0.00008)")

                                elif(parameters['optimization']== 'Adamax'):
                                    code_string +=("\noptimizer = torch.optim.Adamax(regressor_model.parameters(), lr="+str(alpha)+", weight_decay=0.00008)")

                                code_string +=("\nfor e in range(EPOCHS):")
                                code_string +=("\n\ttrain_epoch_loss = 0")
                                code_string +=("\n\tregressor_model.train()")
                                code_string +=("\n\tfor X_train_batch, y_train_batch in train_loader:")
                                code_string +=("\n\t\tX_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)")
                                code_string +=("\n\t\toptimizer.zero_grad()")
                                code_string +=("\n\t\ty_train_pred = regressor_model(X_train_batch,BATCH_SIZE)")

                                code_string +=("\n\t\ttrain_loss = loss_func(y_train_pred, y_train_batch.unsqueeze(1))")
                                code_string +=("\n\t\ttrain_loss.backward()")
                                code_string +=("\n\t\toptimizer.step()")

                                code_string +=("\n\t\ttrain_epoch_loss += train_loss.item()")
                                code_string +=("\n\twith torch.no_grad():")
                                code_string +=("\n\t\tval_epoch_loss = 0")
                                code_string +=("\n\t\tregressor_model.eval()")
                                code_string +=("\n\t\tfor X_val_batch, y_val_batch in val_loader:")
                                code_string +=("\n\t\t\tX_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)")
                                code_string +=("\n\t\t\ty_val_pred = regressor_model(X_val_batch,16)")


                                code_string +=("\n\t\t\tval_loss = loss_func(y_val_pred, y_val_batch.unsqueeze(1))")
                                code_string +=("\n\t\t\tval_epoch_loss += val_loss.item()")
                                code_string +=("\n\tprint(train_epoch_loss/len(train_loader))")
                                code_string +=("\n\tprint(val_epoch_loss/len(val_loader))")

                                code_string +=("\ny_pred_list = []")
                                code_string +=("\nfor X_test_batch, y_test_batch in test_loader:")
                                code_string +=("\n\tX_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)")
                                code_string +=("\n\ty_test_pred = regressor_model(X_test_batch,1)")
                                code_string +=("\n\ty_pred_list.append(y_test_pred)")
                                code_string +=("\ny_pred_inv = y_transformer.inverse_transform(pd.DataFrame(y_pred_list))")

                                py_out.write(code_string)

        #py_out.close()
        toelectronmain("Final_Message : Python-Code generated")
        #with open('debug.json', 'w') as fp:
            #json.dump(str(code_string), fp)

    except Exception as e:
        toelectronmain("Error Encountered")
        with open('debug.json', 'w') as fp:
            json.dump(str(e), fp)



class Regressor_LSTM(nn.Module):
    def __init__(self, layers, neurons, dropouts, input_dim, seq_len):
        super(Regressor_LSTM, self).__init__()
        self.layers = int(layers)
        self.neurons = neurons
        self.dropouts = dropouts
        self.input_dim = input_dim
        #self.batch_size = batch
        self.seq_length = seq_len
        self.out = 0

        self.IP_Layer = nn.LSTM(input_size=self.input_dim,hidden_size=int(self.neurons[0]),dropout=float(dropouts[0]))

        self.Out_Layer = nn.Linear(int(self.neurons[self.layers-1]),1,bias=True)
        #self.relu = nn.ReLU()

    #def hidden_states(b):
        #self.hidden_state = [each.detach() for each in self.hidden_state]


    def forward(self,X, batch):

        #X = X.view(self.seq_length,self.batch_size,self.input_dim)
        #hidden_states(batch)
        self.hidden_state=[]
        for i in range(self.layers):
            h_x = Variable(torch.zeros(1,batch, int(self.neurons[i])))
            c_x = Variable(torch.zeros(1, batch, int(self.neurons[i])))
            hidden = (h_x,c_x)
            hidden = [each.detach() for each in hidden]
            self.hidden_state.append(hidden)

        for i in range(self.layers):
            if(i==0):
                self.out, self.hidden_state[i] = self.IP_Layer(X.view(self.seq_length,len(X),-1),self.hidden_state[i])
            else:
                self.regress = nn.LSTM(input_size=int(self.neurons[i-1]),hidden_size=int(self.neurons[i]),dropout=float(self.dropouts[i]))
            #h_1 = Variable(torch.zeros(1, self.batch_size, self.neurons[i]))
            #c_1 = Variable(torch.zeros(1, self.batch_size, self.neurons[i]))
                self.out, self.hidden_state[i] = self.regress(self.out.view(self.seq_length,len(X),-1), self.hidden_state[i])
            self.hidden_state[i] = [each.detach() for each in self.hidden_state[i]]
         #self.relu(out)
        #out = out.view(-1, int(self.neurons[self.layers-1]))
        self.out = self.Out_Layer(self.out.view(self.seq_length,len(X),int(self.neurons[self.layers-1]))[-1])
        #self.out = self.ReLU(self.out)

        return self.out


class DataCleansing:
  DATA_COLUMN = 'email'     #parameters['headers']
  LABEL_COLUMN = 'label'    #target = parameters['target']

  def __init__(self, train, test, tokenizer : FullTokenizer, classes, max_seq_len = 125):
    self.tokenizer = tokenizer
    self.max_seq_len = 0
    self.classes = classes


    train, test = map(lambda df: df.reindex(df[DataCleansing.DATA_COLUMN].str.len().sort_values().index),[train , test])
    ((self.train_x, self.train_y), (self.test_x, self.test_y)) = map(self.prepare_data, [train, test])

    #print("max seq_len", self.max_seq_len)
    self.max_seq_len = min(self.max_seq_len, max_seq_len)
    self.train_x, self.test_x = map(self.pad, [self.train_x, self.test_x])


  def prepare_data(self, df):
    x,y = [], []

    for _, row in df.iterrows():
      text, label = row[DataCleansing.DATA_COLUMN], row[DataCleansing.LABEL_COLUMN]
      tokens = self.tokenizer.tokenize(str(text))
      tokens =  ["[CLS]"] + tokens + ["[SEP]"]
      token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
      self.max_seq_len = max(self.max_seq_len, len(token_ids))
      x.append(token_ids)
      y.append(self.classes.index(label))

    return np.array(x), np.array(y)

  def pad(self, ids):
    x= []
    for input_ids in ids:
      input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
      input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
      x.append(np.array(input_ids))

    return np.array(x)



class TextClassificationDataset(Dataset):
  def __init__(self, data, target, tokenizer, max_len):
    self.data = data
    self.target = target
    self.tokenizer = tokenizer
    self.max_len = max_len
    #print(target.dtype)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, items):
    data = str(self.data[items])
    target = self.target[items]
    encoding = self.tokenizer.encode_plus(data, max_length=self.max_len, add_special_tokens=True, return_token_type_ids=False, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')

    return {
        #'text' : data,
        'input_ids' : encoding['input_ids'].flatten(),
        'attention_masks' : encoding['attention_mask'].flatten(),
        'targets' : torch.tensor(target, dtype = torch.long)
     }

def create_dataset(df, tokenizer, max_len, batch_size):
    dataset = TextClassificationDataset(data= df[headers].to_numpy(),target = df[target].to_numpy(),tokenizer = tokenizer, max_len = max_len)
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)# parameters['batch_size']




def tf_text_classifier(max_seq_len, bert_ckpt_file):

  input_ids = keras.layers.Input(shape=(max_seq_len, ), dtype='int32', name="input_ids")
  bert_output = bert(input_ids)


  cls_out = keras.layers.Lambda(lambda seq:seq[:,0,:])(bert_output)
  cls_out = keras.layers.Dropout(0.05)(cls_out)
  cls_out = keras.layers.Dense(units=786,activation = 'tanh')(cls_out)
  cls_out = keras.layers.Dropout(0.05)(cls_out)
  cls =keras.layers.Dense(units=len(classes), activation='softmax')(cls_out)

  model = keras.Model(inputs= input_ids, outputs = cls)
  model.build(input_shape=(None, max_seq_len))

  load_stock_weights(bert, bert_ckpt_file)

  return model

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def data_preprocessing(file, parameters):
    # Importing the dataset
    #toelectronmain("In data preprocessing")
    global error_occured
    toelectronmain("Display_Message :Preprocessing data")
    try:
        global input_file
        input_frame = input_file.copy()
        #toelectronmain("input_file")
        #toelectronmain(input_file)
        #toelectronmain(input_frame)

        target = parameters['target'].strip()
        #toelectronmain('targets')
        #toelectronmain(target)


        if(type(parameters['headers'])==str):
            #toelectronmain('parameters_headers is a string')
            temp = parameters['headers']
            parameters['headers'] = list()
            parameters['headers'].append(temp)

        temp = list()
        for header in parameters['headers']:
            header1 = header.strip()
            temp.append(header1)
            #header = header.replace(" ","")
            #toelectronmain(header)
        del parameters['headers'][:]
        parameters['headers'] = temp.copy()
        toelectronmain("headers:")
        toelectronmain(parameters['headers'])
        toelectronmain("temp:")
        toelectronmain(temp)

    except Exception as e:
        toelectronmain("Error Encountered" + e)
        error_occured = True
        #with open('debug1.json', 'w') as fp:
            #json.dump(str(e), fp)
    ##print(input_frame.info())
        #column_list = list(self.dataset.columns)
    #input_frame.drop(target,1)
    #print(parameters[1])
    if(parameters['type']=='Time Series'):

        try:
            toelectronmain("Display_Message :Preparing data for Time Series Prediction")
            X = input_frame.copy()
            y = input_frame[target]

            #print(input_frame.head())
            #print(X,y)
            inx = list(X.columns)
            column_list = parameters['headers']
            #print(column_list)
            drop = True
            #print(column_list)
            for column1 in inx:
                for column2 in column_list:
                    if(column1==column2):
                        drop = False
                        break
                if(drop==True):
                    X = X.drop(column1, axis=1)
                drop = True

            toelectronmain("Display_Message: Creating train, validation and test set")
            test_size = int(len(X) * float(parameters['testsplit']))
            train_size = len(X) - test_size
            train, test = X.iloc[0:train_size], X.iloc[train_size:len(X)]

            global y_transformer
            global f_transformer

            y_transformer = y_transformer.fit(train[[target]])
            y_trn = y_transformer.transform(train[[target]])
            #print(y_trn.shape)
            y_tst = y_transformer.transform(test[[target]])
            #print(y_tst.shape)

            #f_transformer = RobustScaler()
            f_transformer = f_transformer.fit(train[column_list].to_numpy())
            train.loc[:, column_list] = f_transformer.transform(train[column_list].to_numpy())
            test.loc[:, column_list] = f_transformer.transform(test[column_list].to_numpy())

            time_steps = 10
            # reshape to [samples, time_steps, n_features]
            y_trn = pd.DataFrame(y_trn)
            y_tst = pd.DataFrame(y_tst)
            X_train, y_train = create_dataset(train, y_trn, time_steps)
            X_test, y_test = create_dataset(test, y_tst, time_steps)
            #print(X_train.shape,y_train.shape)

            return(X_train, X_test, y_train, y_test)

        except Exception as e:
            error_occured = True
            toelectronmain("Error Encountered:" + e)
            with open('debug.json', 'w') as fp:
                json.dump(str(e), fp)

    if(parameters['type']=='Classification') or (parameters['type']=='MultiClass'):
        try:
            if(parameters['type']=='Classification'):
                toelectronmain("Display_Message :Preparing data for binary classiification")
            else:
                toelectronmain("Display_Message :Preparing data for multiclass classiification")

            X = input_frame.drop(target, axis = 1)
            y = input_frame[target]
            #print(input_frame.head())
            #print(X,y)
            inx = list(X.columns)
            toelectronmain(inx)
                    #print(type(parameters[10]))
                    #print(inx)
            column_list = parameters['headers']
            for name in column_list:
                if(name==target):
                    column_list.remove(target)
            #with open('debug.json', 'w') as fp:
                #json.dump(str(column_list), fp)

                    #print(column_list)
            drop = True
                    #print(column_list)
            for column1 in inx:
                for column2 in column_list:
                    if(column1==column2):
                        drop = False
                        break
                if(drop==True):
                    X = X.drop(column1, axis=1)
                            #print(type(column1))
                            #print(input_frame.head())
                            #print(type(column1))
                drop = True

            for column in column_list:
                if(X[column].dtype == 'object'):
                    X[column] = X[column].astype('category')
                    X[column] = X[column].cat.codes
            if(y.dtype == 'object'):
                y = y.astype('category')
                y = y.cat.codes

            if(parameters['type'] == 'MultiClass'):
                encoder = LabelEncoder()
                encoder.fit(y)
                y_encoded = encoder.transform(y)
                y_categorical = keras.utils.to_categorical(y_encoded)


            toelectronmain("Display_Message : Generating train, validation and test sets")
            X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size = float(parameters['testsplit']), random_state = 2)

                    #print(X_train)
            sc = StandardScaler()
                    #print(X_train)
            X_train = sc.fit_transform(X_train)
                    #print(X_train)
            X_test = sc.transform(X_test)
                    #print(X_test)
                    #print(X_train.shape, y_train.shape)
            return(X_train, X_test, y_train, y_test)

        except Exception as e:
            error_occured = True
            toelectronmain("Error Encountered:"+e)
            #with open('debug1.json', 'w') as fp:
                #json.dump(str(e), fp)
    #checkforreset()

def tf_ann(parameters):
    toelectronmain("Display_Message : Setting up keras for binary classification")
    global loss_stats
    global w_n_b
    global error_occured
    #print(type(parameters[10]))
    X_train, X_test, y_train, y_test = data_preprocessing('data.json', parameters)
    toelectronmain("Display_Message : Processed data")
    #print(type(X_train),type(y_train))
    #checkforreset()
    with open('debug.json', 'w') as fp:
        json.dump(str(y_train.shape), fp)

    try:
        alpha = float(parameters['learning_rate'])
        #print(alpha)

        layers = int(parameters['layers'])
        #print(layers)

        neurons = parameters['neurons']
        #print(neurons)

        activationfunction = parameters['activation']
        #print(activationfunction)

        dropouts = parameters['dropouts']
        #print(droputs)

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
        #checkforreset()
        #print(type(layers))
        toelectronmain("Display_Message : Initialising Neural Network for Binary Classification")
        # Initialising the ANN
        classifier = Sequential()

        for i in range(layers+1):
            #print(i)
            # Adding the input layer and the first hidden layer
            if(i == 0):
                #print(neurons[i])
                classifier.add(Dense(units = int(neurons[i]), kernel_initializer = 'uniform', activation = 'relu', input_dim = X_test.shape[1]))
                classifier.add(Dropout(float(dropouts[i])))
                #print(i)
                #print(neurons[i])
            elif(i == layers):
                classifier.add(Dense(units = int(neurons[i]), kernel_initializer = 'uniform', activation = 'sigmoid'))
                #print(neurons[i])
            else:
                # Adding remaining hidden layers
                classifier.add(Dense(units = int(neurons[i]), kernel_initializer = 'uniform', activation = 'relu'))
                classifier.add(Dropout(float(dropouts[i])))
                #print(neurons[i])
            #classifier.add(Dropout(float(dropouts[i])))
        #print(classifier.summary())
        #checkforreset()
                # Compiling the ANN
        classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
        #print(optimizer)
        #Fitting the ANN to the Training set
        #checkforreset()
        toelectronmain("Display_Message: Fitting Neural Network Onto training set")
        history = classifier.fit(X_train, y_train, batch_size = int(parameters['batch_size']), epochs = int(parameters['epochs']), validation_split=float(parameters['validsplit']))

        toelectronmain("Display_Message : Evaluating neural network on the test set")
        y_pred = classifier.predict(X_test)
        y_pred = (y_pred > 0.5)
        toelectronmain("Display_Message: Generating Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

    except Exception as e:
        error_occured = True
        toelectronmain("Error Encountered")
    #history = classifier.evaluate(X_test,)
    #checkforreset()
    #with open('predict.json', 'w') as fp:
        #json.dump(str(y_pred), fp)
    try:
        i=0
        for layer in classifier.layers:
            w_n_b['layers'].append(layer.name)
            if(layer.name.find("dropout")==-1):
                w_n_b['weights'].append((layer.get_weights()[0].transpose()).tolist())
                w_n_b['biases'].append((layer.get_weights()[1].transpose()).tolist())
            i= i + 1
        with open('weights.json','w') as fp :
            json.dump(w_n_b,fp)
    except Exception as e:
        error_occured = True
        toelectronmain("Error Encountered")
        #with open('debug.json', 'w') as fp:
        #    json.dump(str(e), fp)

    #checkforreset()
    try:
        loss_stats["loss"] = history.history['loss']
        loss_stats["val_loss"] = history.history['val_loss']
        loss_stats["accuracy"] = history.history['accuracy']
        loss_stats["val_accuracy"] = history.history['val_accuracy']
        loss_stats["confusion_matrix"] = np.array(pd.DataFrame(cm)).tolist()

        toelectronmain("Error Encountered")
        with open('result.json', 'w') as fp:
            json.dump(loss_stats, fp)

        toelectronmain("Final_Message : Check Result")
    except Exception as e:
        error_occured = True
        toelectronmain("Error Encountered")
    #toelectronmain("Final_Message : Kill Script")

def tf_rnn(parameters):
    #print(X_train.shape,y_train.shape,type(parameters[5]))
    global error_occured
    try:
        global w_n_b
        toelectronmain("Display_Message : Setting up Keras for Time series analysis")
        #print(parameters[1])
        #dataset_train = pd.read_json('data.json')
        X_train, X_test, y_train, y_test = data_preprocessing('data.json', parameters)
        toelectronmain("Display_Message :Processed data, ready for tuning")

        #print('Backfromprocessing')
        #checkforreset()

        global loss_stats
        alpha = float(parameters['learning_rate'])
        #print(alpha)

        layers = int(parameters['layers'])
        #print(layers)

        neurons = parameters['neurons']
        #print(neurons)

        activationfunction = parameters['activation']
        #print(activationfunction)

        dropouts = parameters['dropouts']

        #checkforreset()
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

        #checkforreset()
        #print(optimizer)
        # Part 2 - Building the RNN
        # Initialising the RNN
        regressor = Sequential()
        toelectronmain("Display_Message : Initilalizing Neural Network")
        for i in range(layers):
            #print(i)
            if(i == 0):
                #print(neurons[i])
                # Adding the first LSTM layer and some Dropout regularisation
                # datapoint,timesteps,dimensions
                #print(X_train.shape)
                if(layers-1 == 0):
                    regressor.add(LSTM(units = int(neurons[i]), return_sequences = False, input_shape = (X_train.shape[1], X_train.shape[2])))
                else :
                    regressor.add(LSTM(units = int(neurons[i]), return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))

                regressor.add(Dropout(float(dropouts[i])))
                #print(neurons[i])
            elif(i == layers-1):
                regressor.add(LSTM(units = int(neurons[i]), return_sequences = False))
                regressor.add(Dropout(float(dropouts[i])))
                #print(neurons[i])
            else:
                # Adding remaining hidden layers
                regressor.add(LSTM(units = int(neurons[i]), return_sequences = True ))
                regressor.add(Dropout(float(dropouts[i])))
                #regressor.add(Dropout(0.2))
                #print(neurons[i]
        #print(dropouts)
        #checkforreset()
        # Adding the output layer
        regressor.add(Dense(units = 1))
        #print(regressor)
        # Compiling the RNN
        regressor.compile(optimizer = optimizer, loss = 'mean_squared_error')
        #print(regressor)
        # Fitting the RNN to the Training set
        #checkforreset()
        toelectronmain("Display_Message: Fitting Neural Network onto the training set")
        history = regressor.fit(X_train, y_train, validation_split=float(parameters['validsplit']), epochs=int(parameters['epochs']), batch_size = int(parameters['batch_size']))
    except Exception as e:
        error_occured = True
        toelectronmain("Error Encountered")
        #with open('debug.json', 'w') as fp:
        #    json.dump(str(e), fp)
        #print(parameters[5])
    try:
        i=0
        #checkforreset()
        for layer in regressor.layers:
            w_n_b['layers'].append(layer.name)
            if(layer.name.find("droput")==-1):
                if(layer.name.find("lstm")!=-1):
                    w_n_b['weights'].append((layer.get_weights()[0].transpose()).tolist())
                    w_n_b['biases'].append((layer.get_weights()[2].transpose()).tolist())
                    w_n_b['U'].append((layer.get_weights()[1].transpose()).tolist())
                if(layer.name.find("dense")!=-1):
                    w_n_b['weights'].append((layer.get_weights()[0].transpose()).tolist())
                    w_n_b['biases'].append((layer.get_weights()[1].transpose()).tolist())
            i= i + 1
        with open('weights.json','w') as fp :
            json.dump(w_n_b,fp)
    #print(json.dumps(str(history.history)))
        loss_stats["loss"] = history.history['loss']
        loss_stats["val_loss"] = history.history['val_loss']
        #loss_stats['accuracy'] = history.history['accuracy']
        #loss_stats['val_accuracy'] = history.history['val_accuracy']

        global y_transformer
        toelectronmain("Display_Message : Predicting future values for the test set")
        y_pred = regressor.predict(X_test)
        loss_stats["y_train_inv"] = y_transformer.inverse_transform(y_train.reshape(1, -1)).tolist()
        loss_stats["y_test_inv"] = y_transformer.inverse_transform(y_test.reshape(1, -1)).tolist()
        loss_stats["y_pred_inv"] = y_transformer.inverse_transform(y_pred).tolist()
        with open('result.json', 'w') as fp:
            json.dump(loss_stats, fp)
        toelectronmain("Error Encountered")
        toelectronmain("Final_Message :Check Result")
        #toelectronmain("Final_Message : Kill Script")

    except Exception as e:
        error_occured = True
        toelectronmain("Error Encountered")


def tf_multiclass(parameters):
    toelectronmain("Display_Message : Setting up keras for binary classification")
    global loss_stats
    global w_n_b
    global error_occured
    #print(type(parameters[10]))
    X_train, X_test, y_train, y_test = data_preprocessing('data.json', parameters)
    toelectronmain("Display_Message : Processed data")
    #print(type(X_train),type(y_train))
    #checkforreset()
    try:
                alpha = float(parameters['learning_rate'])
                #print(alpha)

                layers = int(parameters['layers'])
                #print(layers)

                neurons = parameters['neurons']
                #print(neurons)

                activationfunction = parameters['activation']
                #print(activationfunction)

                dropouts = parameters['dropouts']
                #print(droputs)

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
                #checkforreset()
                #print(type(layers))
                toelectronmain("Display_Message : Initialising Neural Network for Classification")
                # Initialising the ANN
                classifier = Sequential()

                for i in range(layers+1):
                    if(i == 0):
                        classifier.add(Dense(units = int(neurons[i]), activation = 'relu', input_dim = X_test.shape[1]))
                        classifier.add(Dropout(float(dropouts[i])))
                    elif(i == layers):
                        classifier.add(Dense(units = y_test.shape[1], activation = 'relu'))

                    else:
                        # Adding remaining hidden layers
                        classifier.add(Dense(units = int(neurons[i]), activation = 'relu'))
                        classifier.add(Dropout(float(dropouts[i])))

                classifier.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

                toelectronmain("Display_Message: Fitting Neural Network Onto training set")
                history = classifier.fit(X_train, y_train, batch_size = int(parameters['batch_size']), epochs = int(parameters['epochs']), validation_split=float(parameters['validsplit']))

                toelectronmain("Display_Message : Evaluating neural network on the test set")
                y_predict = np.argmax(classifier.predict(X_test),axis=1)
                y_actual = np.argmax(y_test,axis=1)

                i=0
                for layer in classifier.layers:
                    w_n_b['layers'].append(layer.name)
                    if(layer.name.find("dropout")==-1):
                        w_n_b['weights'].append((layer.get_weights()[0].transpose()).tolist())
                        w_n_b['biases'].append((layer.get_weights()[1].transpose()).tolist())
                    i= i + 1
                with open('weights.json','w') as fp :
                    json.dump(w_n_b,fp)


                cm =confusion_matrix(y_actual,y_predict)
                loss_stats["loss"] = history.history['loss']
                loss_stats["val_loss"] = history.history['val_loss']
                loss_stats["accuracy"] = history.history['accuracy']
                loss_stats["val_accuracy"] = history.history['val_accuracy']
                loss_stats["confusion_matrix"] = np.array(pd.DataFrame(cm)).tolist()

                with open('result.json', 'w') as fp:
                    json.dump(loss_stats, fp)
                checkoutputfiles()
                toelectronmain("Final_Message : Check Result")



    except Exception as e:
        toelectronmain("Error Encountered")
        toelectronmain(str(e))
        error_occured = True



def tf_nlp_classify(lines):
    try:
        global loss_stats
        alpha = float(parameters['learning_rate'])
        #print(alpha)
        activationfunction = parameters['activation']
        #print(activationfunction)

        #print(droputs)

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

        global error_occured
        global input_file
        input_frame = input_file.copy()
        bert_model_name="uncased_L-12_H-768_A-12"
        bert_ckpt_dir = os.path.join("model/", bert_model_name)
        bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
        bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")

        tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir,"vocab.txt"))

        with tf.io.gfile.GFile(bert_config_file,mode='r') as reader:
            bc = StockBertConfig.from_json_string(reader.read())
            bert_params = map_stock_config_to_params(bc)
            bert_params.adapter_size = None
            bert = BertModelLayer.from_params(bert_params, name="bert")

        test_size = int(len(input_file) * float(parameters['testsplit']))
        train_size = len(input_file) - test_size
        train, test = input_file.iloc[0:train_size], input_file.iloc[train_size:len(input_file)]

        classes = train.label.unique().tolist()
        data = DataCleansing(train, test, tokenizer, classes, max_seq_len=128)
        model = tf_text_classifier(data.max_seq_len, bert_ckpt_file)

        model.compile(optimizer= optimizer, loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        #parameters['optimization'], parameters['learning_rate']

        history = model.fit(x=data.train_x, y=data.train_y, validation_split=float(parameters['validsplit']), batch_size=int(parameters['batch_size']), epochs=int(parameters['epochs']))
        #, parameters['epochs'], parameters['batch_size']

        pred_y = model.predict(data.test_x)
        y_pred = np.argmax(y_pred,axis=-1)
        cm = confusion_matrix(data.test_y, y_pred)

        loss_stats["loss"] = history.history['loss']
        loss_stats["val_loss"] = history.history['val_loss']
        loss_stats["accuracy"] = history.history['accuracy']
        loss_stats["val_accuracy"] = history.history['val_accuracy']
        loss_stats["confusion_matrix"] = np.array(pd.DataFrame(cm)).tolist()

        with open('result.json', 'w') as fp:
            json.dump(loss_stats, fp)
            checkoutputfiles()
            toelectronmain("Final_Message : Check Result")

    except Exception as e:
        toelectronmain("Error Encountered")
        toelectronmain(str(e))
        error_occured = True


def tf_nlp_predict(lines):
    try:
        pass
    except Exception as e:
        raise


def pyt_preprocessing(file, parameters):
    global error_occured
    try:
        #checkforreset()
        toelectronmain("Display_Message : Preprocessing data")
        global input_file
        input_frame = input_file.copy()

        target = parameters['target'].strip()
        #toelectronmain('targets')
        #toelectronmain(target)


        if(type(parameters['headers'])==str):
            temp = parameters['headers']
            parameters['headers'] = list()
            parameters['headers'].append(temp)

        temp = list()
        for header in parameters['headers']:
            header1 = header.strip()
            temp.append(header1)
            #header = header.replace(" ","")
            #toelectronmain(header)
        del parameters['headers'][:]
        parameters['headers'] = temp.copy()
        #toelectronmain("headers:")
        #toelectronmain(parameters['headers'])
        #toelectronmain("temp:")
        #toelectronmain(temp)


        X = input_frame.drop(target, axis = 1)
        y = input_frame[target]
        inx = list(X.columns)
        #print(type(parameters[10]))
        #print(inx)
        column_list = parameters['headers']
        for name in column_list:
            if(name==target):
                column_list.remove(target)
        #print(column_list)
        drop = True
        #print(column_list)
        for column1 in inx:
            for column2 in column_list:
                if(column1==column2):
                    drop = False
                    break
            if(drop==True):
                X = X.drop(column1, axis=1)
            drop = True

        for column in column_list:
            if(X[column].dtype == 'object'):
                X[column] = X[column].astype('category')
                X[column] = X[column].cat.codes
        if(y.dtype == 'object'):
            y = y.astype('category')
            y= y.cat.codes
        #checkforreset()
        toelectronmain("Display_Message :Splitting into train,test and dev set")
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=float(parameters['testsplit']), random_state = 0)
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=float(parameters['validsplit']), random_state = 0)

        #checkforreset()
        sc = StandardScaler()
        #sc = MinMaxScaler(feature_range = (0, 1))
        X_train = sc.fit_transform(X_train)
        X_val = sc.transform(X_val)
        X_test = sc.transform(X_test)

        X_train , y_train = np.array(X_train), np.array(y_train)
        X_val, y_val = np.array(X_val), np.array(y_val)
        X_test, y_test = np.array(X_test), np.array(y_test)

        return(X_train , y_train , X_val, y_val, X_test, y_test)

    except Exception as e:
        error_occured = True
        toelectronmain(e)
        #with open('debug.json', 'w') as fp:
        #    json.dump(str(e), fp)


class Dataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
        return len(self.X_data)

def pyt_ANN(parameters):
    #print(parameters[0],parameters[1])
    target = parameters['target'].strip()
    alpha = float(parameters['learning_rate'])
    #print(alpha)

    layers_1 = int(parameters['layers'])
    #print(layers)

    neurons = parameters['neurons']
    #print(neurons)

    activationfunction = parameters['activation']
    #print(activationfunction)

    dropouts = parameters['dropouts']
    global error_occured
    global loss_stats
    global w_n_b
    toelectronmain("Display_Message : Setting up pytorch for Binary Classification")
    X_train , y_train , X_val, y_val, X_test, y_test = pyt_preprocessing('data.json', parameters)
    toelectronmain("Display_Message :Prepared Data for Deep Learning")

    EPOCHS = int(parameters['epochs'])
    BATCH_SIZE = int(parameters['batch_size'])

    #checkforreset()
    try:
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        y_val = y_val.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_test = y_test.astype(np.float32)
        train_dataset = Dataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        val_dataset = Dataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
        test_dataset = Dataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1)

        #try :
        layers = []
        layers.append(nn.Linear(X_train.shape[1],int(neurons[0]),bias=True))
        #layers.append(nn.ReLU())
        toelectronmain("Display_Message : Initializing Neural Network for Binary classification")
        #checkforreset()
        for i in range(layers_1):
            #if(i == 0):
            if(i == layers_1-1):
                layers.append(nn.Linear(int(neurons[i]),1,bias=True))
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.Linear(int(neurons[i]),int(neurons[i+1]),bias=True))
                #layers.append(nn.ReLU())
                layers.append(nn.Dropout(float(dropouts[i])))
        model = nn.Sequential(*layers)

        layers.clear()


        #except Exception as e:


        #NUM_FEATURES = len(X.columns)
        #try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loss_func = nn.BCELoss()
        #loss_func = nn.BCEWithLogitsLoss()
            #loss_func = nn.MSELoss()
            #learning_rate = 0.0015
        #checkforreset()
        if(parameters['optimization']== 'SGD'):
            optimizer = torch.optim.SGD(model.parameters(), lr=alpha, weight_decay=0.00008)

        elif(parameters['optimization']== 'Adam'):
            optimizer = torch.optim.Adam(model.parameters(), lr=alpha, weight_decay=0.00008)

        elif(parameters['optimization']== 'RMSProp'):
            optimizer = torch.optim.RMSprop(model.parameters(), lr=alpha, weight_decay=0.00008)

        elif(parameters['optimization']== 'Adagrad'):
            optimizer = torch.optim.Adagrad(model.parameters(), lr=alpha, weight_decay=0.00008)

        elif(parameters['optimization']== 'Adamax'):
            optimizer = torch.optim.Adamax(model.parameters(), lr=alpha, weight_decay=0.00008)


        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        toelectronmain("Display_Message :Training Neural Network")
        y_pred = []
        y_actual = []
        #checkforreset()
        for e in range(EPOCHS):
            EPOCH_display = "Epoch :"+ str(e) +"/"+str(EPOCHS)
            toelectronmain(EPOCH_display)
            train_epoch_loss = 0
            val_epoch_loss = 0
            train_epoch_acc = 0
            val_epoch_acc = 0

            model.train()
            for X_train_batch, y_train_batch in train_loader:
                X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)

                y_train_pred = model(X_train_batch)
                optimizer.zero_grad()
                y_actual.append(y_train_batch.unsqueeze(1))
                train_loss = loss_func(y_train_pred, y_train_batch.unsqueeze(1))
                train_loss.backward()
                optimizer.step()

                #y_pred_tag = torch.round(torch.sigmoid(y_train_pred)
                y_pred_tag = (y_train_pred > 0.5).float()
                acc = ((y_pred_tag == y_train_batch.unsqueeze(1)).sum().float())/y_train_batch.shape[0]
                train_epoch_loss += train_loss.item()
                train_epoch_acc += acc.item()

                # VALIDATION
            with torch.no_grad():

                model.eval()
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                    y_val_pred = model(X_val_batch)

                    val_loss = loss_func(y_val_pred, y_val_batch.unsqueeze(1))
                    val_epoch_loss += val_loss.item()

                    #y_pred_tag = torch.round(torch.sigmoid(y_val_pred))
                    y_pred_tag = (y_val_pred > 0.5).float()
                    acc = ((y_pred_tag == y_val_batch.unsqueeze(1)).sum().float())/y_val_batch.shape[0]
                    val_epoch_acc += acc.item()

            toelectronmain("loss:  " + str(train_epoch_loss/len(train_loader)) + "\tVal loss:  " +str(val_epoch_loss/len(val_loader)))


            loss_stats["loss"].append(train_epoch_loss/len(train_loader))
            loss_stats["val_loss"].append(val_epoch_loss/len(val_loader))

            loss_stats["accuracy"].append(train_epoch_acc/len(train_loader))
            loss_stats["val_accuracy"].append(val_epoch_acc/len(val_loader))


        y_pred_list = []
        y_test_list = []
        toelectronmain("Display_Message :Evaluating Neural Network on test set")
        #checkforreset()
        model.eval()
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            y_pred = model(X_test_batch)
            #y_pred = torch.round(torch.sigmoid(y_pred))
            y_pred_tag = (y_pred > 0.5).float()
            y_pred_list.append(y_pred)
            y_test_list.append(y_test_batch)


        y_pred_list = [a.squeeze().tolist() for a in y_pred_list ]
        y_pred_list = [round(a) for a in y_pred_list]
        y_test_list = [a.squeeze().tolist() for a in y_test_list ]

        toelectronmain("Display_Message :Generating Confusion Matrix")
        cm = confusion_matrix(y_test_list, y_pred_list)
        #toelectronmain("Error Encountered")
        #check the data-types and convert lists to numpy arrays

        loss_stats["confusion_matrix"] = np.array(pd.DataFrame(cm)).tolist()

        model_wnb = model.state_dict()
        #checkforreset()
        for i in range(len(model)):
            w_n_b['layers'].append(str(model[i]))
            if(str(model[i]).find("Linear")!=-1):
                str1 = str(i) + ".weight"
                w_n_b['weights'].append(model_wnb[str1].tolist())
                str1 = str(i) + ".bias"
                w_n_b['biases'].append(model_wnb[str1].tolist())

        with open('weights.json','w') as fp :
            json.dump(w_n_b,fp)
        with open('result.json', 'w') as fp:
            json.dump(loss_stats, fp)
        toelectronmain("Display_Message :Check Result")
    except Exception as e:
        error_occured = True
        toelectronmain("Error Encountered"+str(e))
        #with open('debug1.json', 'w') as fp:
        #    json.dump(str(e), fp)

def pyt_RNN(parameters):
    global error_occured
    try:
        toelectronmain("Display_Message: Setting up PyTorch for time series analysis")
        global input_file
        global w_n_b
        input_frame = input_file.copy()

        target = parameters['target'].strip()
        #toelectronmain('targets')
        #toelectronmain(target)
        #checkforreset()
        if(type(parameters['headers'])==str):
            temp = parameters['headers']
            parameters['headers'] = list()
            parameters['headers'].append(temp)

        temp = list()
        for header in parameters['headers']:
            header1 = header.strip()
            temp.append(header1)
                #header = header.replace(" ","")
                #toelectronmain(header)
        del parameters['headers'][:]
        parameters['headers'] = temp.copy()


        X = input_frame
        y = input_frame[target]
        alpha = float(parameters['learning_rate']) # alpha : learning_rate
        #print(alpha)

        layers_1 = int(parameters['layers'])
        #print(layers)

        neurons = parameters['neurons']
        #print(neurons)

        activationfunction = parameters['activation']
        #print(activationfunction)

        dropouts = parameters['dropouts']
        global loss_stats

        #print(input_frame.head())
        #print(X,y)
        #with open('debug2.json', 'a') as fp:
        #    json.dump(str(type(parameters['headers'])),fp)
        #    json.dump(str(X.columns), fp)

        inx = list(X.columns)
        #print(type(parameters[10]))
        #print(inx)
        column_list = parameters['headers']
        #json.dump(str(column_list), fp)
        #print(column_list)
        drop = True
        #print(column_list)

        #checkforreset()
        for column1 in inx:
            for column2 in column_list:
                if(column1==column2):
                    drop = False
                    break
            if(drop==True):
                X = X.drop(column1, axis=1)
                #print(type(column1))
                #print(input_frame.head())
                #print(type(column1))
            drop = True
        #X = X.filter(column_list, axis=1)
        #with open('debug.json', 'w') as fp:
            #json.dump(str(drop), fp)
        #with open('debug2.json', 'a') as fp:
        #    json.dump(str(X.columns), fp)


        #checkforreset()
        toelectronmain("Display_Message : Splitting data into train, dev and test set")
        test_size = int(len(X) * float(parameters['testsplit']))
        train_size = len(X) - test_size
        train, test = X.iloc[0:train_size], X.iloc[train_size:len(X)]
        val_size = int(train_size * float(parameters['validsplit']))
        train_size = train_size - val_size
        train,val = train.iloc[0:train_size], train.iloc[train_size: (train_size+val_size)]

        #print(test.shape)
        #print(type(target))


    except Exception as e:
        error_occured = True
        toelectronmain("Error Encounterd : Pytorch LSTM")
    try:
        #y_transformer = RobustScaler()
        global y_transformer
        global f_transformer
        y_transformer = y_transformer.fit(train[[target]])
        y_trn = y_transformer.transform(train[[target]])
        #print(y_trn.shape)
        y_tst = y_transformer.transform(test[[target]])
        #print(y_tst.shape)
        y_val = y_transformer.transform(val[[target]])

        #f_transformer = RobustScaler()
        f_transformer = f_transformer.fit(train[column_list].to_numpy())
        train.loc[:, column_list] = f_transformer.transform(train[column_list].to_numpy())
        test.loc[:, column_list] = f_transformer.transform(test[column_list].to_numpy())
        val.loc[:, column_list] = f_transformer.transform(val[column_list].to_numpy())
        loss_stats["y_train_inv"] = y_transformer.inverse_transform(y_trn.reshape(1,-1)).tolist()
        ytrain_inv = y_transformer.inverse_transform(y_val.reshape(1,-1)).tolist()
        loss_stats["y_train_inv"][0].extend(ytrain_inv[0])
        loss_stats["y_test_inv"] = y_transformer.inverse_transform(y_tst.reshape(1,-1)).tolist()

        time_steps = 10
        # reshape to [samples, time_steps, n_features]
        y_trn = pd.DataFrame(y_trn)
        y_tst = pd.DataFrame(y_tst)
        y_val = pd.DataFrame(y_val)
        X_train, y_train = create_dataset(train, y_trn, time_steps)
        X_test, y_test = create_dataset(test, y_tst, time_steps)
        X_val, y_val = create_dataset(val, y_val, time_steps)
    except Exception as e:
        error_occured = True
        with open('debug.json', 'w') as fp:
            json.dump(str(e), fp)

    try:
        EPOCHS = int(parameters['epochs'])
        BATCH_SIZE = int(parameters['batch_size'])

        train_dataset = Dataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        val_dataset = Dataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
        test_dataset = Dataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE,shuffle=False, drop_last=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #loss_func = nn.BCELoss()
        loss_func = nn.MSELoss()
        #learning_rate = 0.0015
        #model = nn.ModuleList()

        #for i in range(layers_1):
            #if(i==0):
                #model.append(nn.LSTM(input_size=X_train.shape[2],hidden_size=int(neurons[0]),dropout=float(dropouts[i])))
        toelectronmain("Display_Message : Initialising Neural Network")
        regressor_model = Regressor_LSTM(layers_1, neurons, dropouts, X_train.shape[2], 10)
        with open('debug.json', 'w') as fp:
            json.dump(str(X_train.shape), fp)

        if(parameters['optimization']== 'SGD'):
            optimizer = torch.optim.SGD(regressor_model.parameters(), lr=alpha, weight_decay=0.00008)

        elif(parameters['optimization']== 'Adam'):
            optimizer = torch.optim.Adam(regressor_model.parameters(), lr=alpha, weight_decay=0.00008)

        elif(parameters['optimization']== 'RMSProp'):
            optimizer = torch.optim.RMSprop(regressor_model.parameters(), lr=alpha, weight_decay=0.00008)

        elif(parameters['optimization']== 'Adagrad'):
            optimizer = torch.optim.Adagrad(regressor_model.parameters(), lr=alpha, weight_decay=0.00008)

        elif(parameters['optimization']== 'Adamax'):
            optimizer = torch.optim.Adamax(regressor_model.parameters(), lr=alpha, weight_decay=0.00008)

    except Exception as e:
        error_occured = True
        toelectronmain("Error Encountered")

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        toelectronmain("Display_Message :Training Neural Network")
        #checkforreset()
        for e in range(EPOCHS):
            EPOCH_display = "Epoch :"+ str(e) +"/"+str(EPOCHS)
            toelectronmain(EPOCH_display)
            train_epoch_loss = 0
            #hidden_state = [each.detach() for each in hidden]
            #global regressor_model
            regressor_model.train()
            for X_train_batch, y_train_batch in train_loader:
                X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
                optimizer.zero_grad()
                #X_train_batch = X_train_batch.permute(1,0,2)
                with open('debug.json', 'w') as fp:
                    json.dump(str(X_train_batch.shape), fp)

                y_train_pred = regressor_model(X_train_batch,BATCH_SIZE)

                #hidden = [each.detach() for each in hidden]

                #hidden = Variable(hidden.data, requires_grad=True)
                #hidden = hidden.detach()

                train_loss = loss_func(y_train_pred, y_train_batch.unsqueeze(1))
                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item()
                #n_correct += (t2orch.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()
                #n_total += batch.batch_size
                #train_acc = n_correct/n_total

                # VALIDATION
            with torch.no_grad():
                val_epoch_loss = 0

                regressor_model.eval()
                for X_val_batch, y_val_batch in val_loader:
                    #h_x = Variable(torch.zeros(1, BATCH_SIZE, int(neurons[0])))
                    #c_x = Variable(torch.zeros(1, BATCH_SIZE, int(neurons[0])))
                    #hidden = (h_x,c_x)

                    X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                    #X_val_batch = X_val_batch.permute(1,0,2)
                    y_val_pred = regressor_model(X_val_batch,BATCH_SIZE)


                    val_loss = loss_func(y_val_pred, y_val_batch.unsqueeze(1))
                    val_epoch_loss += val_loss.item()
            loss_stats["loss"].append(train_epoch_loss/len(train_loader))
            loss_stats["val_loss"].append(val_epoch_loss/len(val_loader))
            toelectronmain("loss:  " + str(train_epoch_loss/len(train_loader)) + "\tVal loss:  " + str(val_epoch_loss/len(val_loader)))

        y_pred_list = []
        #y_test_list = []
        toelectronmain("Display_Message :Testing the neural net on the test dataset")
        #checkforreset()
        for X_test_batch, y_test_batch in test_loader:
            #h_x = Variable(torch.zeros(1, 1, int(neurons[0])))
            #c_x = Variable(torch.zeros(1, 1, int(neurons[0])))
            #hidden = (h_x,c_x)

            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            #X_test_batch = X_test_batch.permute(1,0,2)
            y_test_pred = regressor_model(X_test_batch,1)
            y_pred_list.append(y_test_pred)
            #y_test_list.append(y_test_batch)
        loss_stats["y_pred_inv"] = y_transformer.inverse_transform(pd.DataFrame(y_pred_list)).tolist()
        #loss_stats["y_test_inv"] = y_transformer.inverse_transform(y_test_inv.reshape(1,-1)).tolist()

        #checkforreset()
        model_wnb = regressor_model.state_dict()
        for key in regressor_model.state_dict():
            if(str(key).find("weight_ih")!=-1):
                w_n_b['weights'].append(model_wnb[key].tolist())
            elif(str(key).find("weight_hh")!=-1):
                w_n_b['U'].append(model_wnb[key].tolist())
            elif(str(key).find("weight")!=-1):
                w_n_b['weights'].append(model_wnb[key].tolist())
            if(str(key).find("bias_ih")!=-1):
                w_n_b['biases'].append(model_wnb[key].tolist())
            elif(str(key).find("bias_hh")!=-1):
                w_n_b['biases_hh'].append(model_wnb[key].tolist())
            elif(str(key).find("bias")!=-1):
                w_n_b['biases'].append(model_wnb[key].tolist())
        for name , layer in regressor_model.named_children():
            w_n_b['layers'].append(str(layer))

        with open('weights.json','w') as fp :
            json.dump(w_n_b,fp)
        with open('result.json', 'w') as fp:
            json.dump(loss_stats, fp)
        toelectronmain("Error Encountered")
        toelectronmain("Display_Message: Check Result")


    except Exception as e:
        toelectronmain("Error Encountered" + str(e))
        error_occured = True


def pyt_multiclass(parameters):
    try:
        target = parameters['target'].strip()
        alpha = float(parameters['learning_rate'])
        #print(alpha)

        layers_1 = int(parameters['layers'])
        #print(layers)

        neurons = parameters['neurons']
        #print(neurons)

        activationfunction = parameters['activation']
        #print(activationfunction)

        dropouts = parameters['dropouts']
        global error_occured
        global loss_stats
        global w_n_b
        toelectronmain("Display_Message : Setting up pytorch for Binary Classification")
        X_train , y_train , X_val, y_val, X_test, y_test = pyt_preprocessing('data.json', parameters)
        toelectronmain("Display_Message :Prepared Data for Deep Learning")

        EPOCHS = int(parameters['epochs'])
        BATCH_SIZE = int(parameters['batch_size'])
        NUM_FEATURES = len(X_train.columns)

        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        y_val = y_val.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_test = y_test.astype(np.float32)

        train_dataset = Dataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        val_dataset = Dataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
        test_dataset = Dataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1)

        layers = []
        layers.append(nn.Linear(X_train.shape[1],int(neurons[0]),bias=True))
        #layers.append(nn.ReLU())
        toelectronmain("Display_Message : Initializing Neural Network for Binary classification")
        #checkforreset()
        for i in range(layers_1):
            #if(i == 0):
            if(i == layers_1-1):
                layers.append(nn.Linear(int(neurons[i]),1,bias=True))
                layers.append(nn.LogSoftmax())
            else:
                layers.append(nn.Linear(int(neurons[i]),int(neurons[i+1]),bias=True))
                #layers.append(nn.ReLU())
                layers.append(nn.Dropout(float(dropouts[i])))
        model = nn.Sequential(*layers)

        layers.clear()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loss_func = nn.CrossEntropyLoss()
        #loss_func = nn.BCEWithLogitsLoss()
            #loss_func = nn.MSELoss()
            #learning_rate = 0.0015
        #checkforreset()
        if(parameters['optimization']== 'SGD'):
            optimizer = torch.optim.SGD(model.parameters(), lr=alpha, weight_decay=0.00008)

        elif(parameters['optimization']== 'Adam'):
            optimizer = torch.optim.Adam(model.parameters(), lr=alpha, weight_decay=0.00008)

        elif(parameters['optimization']== 'RMSProp'):
            optimizer = torch.optim.RMSprop(model.parameters(), lr=alpha, weight_decay=0.00008)

        elif(parameters['optimization']== 'Adagrad'):
            optimizer = torch.optim.Adagrad(model.parameters(), lr=alpha, weight_decay=0.00008)

        elif(parameters['optimization']== 'Adamax'):
            optimizer = torch.optim.Adamax(model.parameters(), lr=alpha, weight_decay=0.00008)


        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        toelectronmain("Display_Message :Training Neural Network")


        model.train()
        y_pred = []
        y_actual = []
        #checkforreset()
        for e in range(EPOCHS):
            train_epoch_loss = 0
            val_epoch_loss = 0
            train_epoch_acc = 0
            val_epoch_acc = 0

            model.train()
            for X_train_batch, y_train_batch in train_loader:
                X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
                y_train_pred = model(X_train_batch)
                optimizer.zero_grad()
                #print(y_train_batch.shape)
                #print(y_train_pred.shape)
                #y_actual.append(y_train_batch.unsqueeze(1))
                train_loss = loss_func(y_train_pred, y_train_batch)
                train_loss.backward()
                optimizer.step()
                #print("y_train_pred: "+(str(y_train_pred)))
                y_pred_softmax = torch.log_softmax(y_train_pred,dim = 1)
                #print("y_pred_softmax: "+(str(y_pred_softmax)))
                #y_pred_tag = (y_train_pred > 0.5).float()
                _, y_pred_tag = torch.max(y_pred_softmax, dim = 1)
                correct_pred = (y_pred_tag == y_train_batch).float()
                acc = correct_pred.sum()/len(correct_pred)
                train_epoch_loss += train_loss.item()
                train_epoch_acc += acc.item()
            print(train_epoch_acc/len(train_loader))
            print(train_epoch_loss/len(train_loader))
            #print

            model.eval()

            with torch.no_grad():
              for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch)
                val_loss = loss_func(y_val_pred, y_val_batch)
                y_pred_softmax = torch.log_softmax(y_val_pred,dim = 1)
                _, y_pred_tag = torch.max(y_pred_softmax, dim = 1)
                correct_pred = (y_pred_tag == y_val_batch).float()
                acc = correct_pred.sum()/len(correct_pred)
                val_epoch_loss += train_loss.item()
                val_epoch_acc += acc.item()
            print(val_epoch_acc/len(val_loader))
            print(val_epoch_loss/len(val_loader))


        y_pred_list = []
        y_test_list = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(X_batch)
                y_pred_softmax = torch.log_softmax(y_pred,dim = 1)
                _, y_pred_tag = torch.max(y_pred_softmax, dim = 1)
                y_pred_list.append(y_pred_tag.cpu().numpy())
                y_test_list.append(y_batch)
        y_pred_list = [ a.squeeze().tolist() for a in y_pred_list  ]
        y_test_list = [ a.squeeze().tolist() for a in y_test_list  ]


        cm = confusion_matrix(y_test, y_pred_list)
        #print(cm)
        #check the data-types and convert lists to numpy arrays

        loss_stats["confusion_matrix"] = np.array(pd.DataFrame(cm)).tolist()

        model_wnb = model.state_dict()
        #checkforreset()
        for i in range(len(model)):
            w_n_b['layers'].append(str(model[i]))
            if(str(model[i]).find("Linear")!=-1):
                str1 = str(i) + ".weight"
                w_n_b['weights'].append(model_wnb[str1].tolist())
                str1 = str(i) + ".bias"
                w_n_b['biases'].append(model_wnb[str1].tolist())

        with open('weights.json','w') as fp :
            json.dump(w_n_b,fp)
        with open('result.json', 'w') as fp:
            json.dump(loss_stats, fp)
        toelectronmain("Display_Message :Check Result")

    except Exception as e:
        #raise
        error_occured = True
        toelectronmain("Error Encountered"+str(e))
        #with open('debug1.json', 'w') as fp:
        #    json.dump(str(e), fp)

        #print(parameters[0],parameters[1])



def pyt_textclassify(parameters):
    global error_occured
    global input_file
    try:

        target = parameters['target'].strip()
        alpha = float(parameters['learning_rate'])

        df = input_file
        df.fillna(0)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
        tokenizer  = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

        le = LabelEncoder()
        df[parameters['headers']] = le.fit_transform(df[parameters['headers']])

        trainval, test = train_test_split(df,test_size=float(parameters['testsplit']), random_state = 42)  #
        train, val = train_test_split(trainval,test_size=float(parameters['validsplit']), random_state = 42) #

        train_dataset = create_dataset(train, tokenizer,128 , int(parameters['batch_size']) )
        val_dataset = create_dataset(val, tokenizer, 128, int(parameters['batch_size']))
        test_dataset = create_dataset(test, tokenizer, 128, 1)

        class TextClassificationModel(nn.Module):
            def __init__(self, num_classes):
                super(TextClassificationModel, self).__init__()
                self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict = False)
                self.drop = nn.Dropout(p=0.3)
                self.out = nn.Linear(self.bert.config.hidden_size, num_classes)

            def forward(self, input_ids, attention_mask):
                _, pooled_output = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
                )
                output = self.drop(pooled_output)
                return self.out(output)

        num_classes = len(df[target].unique())
        model = TextClassificationModel(num_classes)
        model = model.to(device)

        EPOCHS = int(parameters['epochs'])


        if(parameters['optimization']== 'SGD'):
            optimizer = torch.optim.SGD(model.parameters(), lr=alpha, weight_decay=0.00008)

        elif(parameters['optimization']== 'Adam'):
            optimizer = torch.optim.Adam(model.parameters(), lr=alpha, weight_decay=0.00008)
        elif(parameters['optimization']== 'RMSProp'):
            optimizer = torch.optim.RMSprop(model.parameters(), lr=alpha, weight_decay=0.00008)
        elif(parameters['optimization']== 'Adagrad'):
            optimizer = torch.optim.Adagrad(model.parameters(), lr=alpha, weight_decay=0.00008)
        elif(parameters['optimization']== 'Adamax'):
            optimizer = torch.optim.Adamax(model.parameters(), lr=alpha, weight_decay=0.00008)
        #parameters['optimization'] , parameters['learning_rate']
        total_steps = len(train_dataset) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=total_steps)

        loss_fn  = nn.CrossEntropyLoss().to(device)

        model.train()


        for e in range(EPOCHS):
            losses = []
            correct_prediction = 0
            val_losses = []
            val_correct_prediction = 0

            for d in train_dataset:
                input_ids = d['input_ids'].to(device)
                attention_masks = d['attention_masks'].to(device)
                targets = d["targets"].to(device)

                outputs = model(input_ids,attention_masks)

                _, preds = torch.max(outputs, dim=1)
                loss = loss_fn(outputs, targets)

                correct_prediction += torch.sum(preds == targets)
                losses.append(loss.item())

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            #print(correct_prediction.double() /  len(train), np.mean(losses))


            model.eval()
            with torch.no_grad():

                for d in val_dataset:
                    input_ids = d["input_ids"].to(device)
                    attention_mask = d["attention_masks"].to(device)
                    targets = d["targets"].to(device)

                    outputs = model(input_ids=input_ids,attention_mask=attention_mask)
                    _, preds = torch.max(outputs, dim=1)

                    loss = loss_fn(outputs, targets)

                    val_correct_prediction += torch.sum(preds == targets)
                    val_losses.append(loss.item())

                #print(val_correct_prediction.double() / len(val), np.mean(val_losses))
                loss_stats["loss"].append(np.mean(losses))
                loss_stats["val_loss"].append(np.mean(val_losses))

                loss_stats["accuracy"].append(correct_prediction.double()/len(train))
                loss_stats["val_accuracy"].append(val_correct_prediction.double() / len(val))


            model.eval()
            y_pred_list = []
            y_test_list = []


            with torch.no_grad():
                for d in test_dataset:
                    input_ids = d["input_ids"].to(device)
                    attention_mask = d["attention_masks"].to(device)
                    targets = d["targets"].to(device)

                    outputs = model(input_ids=input_ids,attention_mask=attention_mask)
                    _, preds = torch.max(outputs, dim=1)

                    y_pred_list.append(preds)
                    y_test_list.append(targets)


                y_pred_list = [ a.squeeze().tolist() for a in y_pred_list  ]
                y_test_list = [ a.squeeze().tolist() for a in y_test_list  ]

                cm = confusion_matrix(y_test_list, y_pred_list)
                loss_stats["confusion_matrix"] = np.array(pd.DataFrame(cm)).tolist()
                #print(cm)

    except Exception as e:
        #toelectronmain(e)
        toelectronmain("Error Encountered" + str(e))
        error_occured = True


def pyt_textpredict(parameters):
    pass


def find_extensions_headers():
    filepathflag = False
    global input_file
    global file_type
    global error_occured
    path = os.path.dirname(__file__)
    path = os.path.join(path,'path.txt').replace("\\","/")
    filepath = None
    #filename, file_extension = os.path.splitext(filepath)
    #filepath = str(filepath.encode('unicode_escape'))
    while(filepathflag == False):
        if(os.path.isfile(path) == True):
            #toelectronmain("In if")
            try:
                time.sleep(1)
                f = open("DLengine/path.txt",'r')
                filepath = f.read()
                #toelectronmain(type(parameters))
                filepathflag = True
                f.close()
            except Exception as e:
                error_occured = True
                toelectronmain("Error Encountered")
                toelectronmain(e)
    os.remove(path)
    try:
        filepath = filepath.strip()
        filepath = filepath.strip('\"')
        if (filepath.lower().endswith(('.xls','.xlsx','.xlsm')) == True):
            #print("in condition 1")
            #sys.stdout.flush()
            input_file = pd.read_excel(filepath)
            file_type = "excel"
        elif (filepath.lower().endswith('.csv') == True):
            input_file = pd.read_csv(filepath)
            file_type = "csv"
            #print("in condition 2")
            #sys.stdout.flush()
        elif (filepath.lower().endswith('.json') == True):
            input_file = pd.read_json(filepath)
            file_type = "json"
            #print("in condition 3")
            #sys.stdout.flush()
        elif (filepath.lower().endswith('.tsv') == True):
            input_file = pd.read_table(filepath)
            file_type = "tsv"
            #print("in condition 4")
            #sys.stdout.flush()
        input_file.fillna(0)
        headers = list(input_file.columns)
        return headers
    except Exception as e:
        error_occured = True
        toelectronmain("Error Encountered")
        toelectronmain(e)

#def #checkforreset():
    #toelectronmain("\nChecking for restarts\n")
    #ath = os.path.dirname(__file__)
    #path = os.path.join(path,'reset.dat').replace("\\","/")
    #if(os.path.isfile(path) == True):
    #    toelectronmain("Notification : Restarted")
    #    start()


#Sending data back main.js
def toelectronmain(args):
    print(args)
    sys.stdout.flush()


def read_in():
    lines = sys.stdin.readlines()
    return json.loads(lines[0])

def read_parameters():
    paramsflag = True
    path = os.path.dirname(__file__)
    global parameters
    global error_occured

    #toelectronmain(type(path))
    #toelectronmain(os.path.join(path,'params.json'))
    toelectronmain("Display_Message : Waiting for hyperparameter input......")
    path = os.path.join(path,'params.json').replace("\\","/")
    #toelectronmain(path)
    #toelectronmain(type(path))
    while(paramsflag == True):
        if(os.path.isfile(path) == True):
            toelectronmain("Display_Message : Reading hyperparameters.....")
            time.sleep(3)
            #toelectronmain("In if")
            try:
                f = open("DLengine/params.json",'r')
                parameters = json.load(f)
                #toelectronmain(parameters)
                #toelectronmain(type(parameters))
                paramsflag == False
                f.close()
            except Exception as e:
                error_occured = True
                toelectronmain("Error Encountered")
                toelectronmain(e)
            os.remove(path)
            return parameters

#def start():
#    while True:
#        try:
#                sys.stdout.flush()
#                toelectronmain("Notification : Input_Features__" + str(find_extensions_headers()))
#                ##checkforreset()
#                lines = read_parameters()
##                #checkforreset()
#                if(lines['framework']=='Keras'):
#                    if(lines['type']=='Classification'):
#                        tf_ann(lines)
                        #print(lines[5])
#                    if(lines['type']=='Time Series'):
#                        tf_rnn(lines)
                #if(lines[0]=='Gluon'):
                    #if(lines[1]=='Classification'):
                        #tf_ann(lines)
                    #if(lines[1]=='Time Series'):
                        #tf_rnn(lines)
#                if(lines['framework']=='PyTorch'):
#
#                    if(lines['type']=='Classification'):
#                        pyt_ANN(lines)
#                    if(lines['type']=='Time Series'):
#                        pyt_RNN(lines)
                #checkforreset()
#                toelectronmain("Display_Message : Generating python autocode")
##                toelectronmain("Final_Message : Kill Script")
#        except Exception as e:
#                toelectronmain(e)


def reset():
    global w_n_b
    global loss_stats

    loss_stats = {
    "loss": [],
    "val_loss": [],
    "accuracy" : [],
    "val_accuracy" : [],
    "y_train_inv": [],
    "y_test_inv": [],
    "y_pred_inv" : [],
    "confusion_matrix" : []
    }


    w_n_b = {
     "layers" : [],
     "weights" : [],
     "biases" : [],
     "U" : [],
     "biases_hh" : []
    }

def checkoutputfiles():
    global w_n_b
    global loss_stats

    for a in loss_stats['loss']:
        if(a != a):
            loss_stats['loss'] = []
            toelectronmain("Error Encountered")

    for a in loss_stats['val_loss']:
        if(a != a):
            loss_stats['val_loss'] = []
            toelectronmain("Error Encountered")

    for a in loss_stats['accuracy']:
        if(a != a):
            loss_stats['accuracy'] = []
            toelectronmain("Error Encountered")

    for a in loss_stats['val_accuracy']:
        if(a != a):
            loss_stats['val_accuracy'] = []
            toelectronmain("Error Encountered")

    for a in loss_stats['y_train_inv']:
        if(a != a):
            loss_stats['y_train_inv'] = []
            toelectronmain("Error Encountered")

    for a in loss_stats['y_test_inv']:
        if(a != a):
            loss_stats['y_test_inv'] = []
            toelectronmain("Error Encountered")

    for a in loss_stats['y_pred_inv']:
        if(a != a):
            loss_stats['y_pred_inv'] = []
            toelectronmain("Error Encountered")

    for a in loss_stats['confusion_matrix']:
        if(a != a):
            loss_stats['confusion_matrix'] = []
            toelectronmain("Error Encountered")

def main():
    #receive the filepath as an input stream from main.js
    #find the input features
    #Return the input features back to the user
    while True:
        try:
                global error_occured
                error_occured = False
                sys.stdout.flush()
                toelectronmain("Notification : Input_Features__" + str(find_extensions_headers()))
                lines = read_parameters()
                os.system('cls' if os.name == 'nt' else 'clear')
                if(lines['framework']=='Keras'):
                    if(lines['type']=='Classification'):
                        tf_ann(lines)
                        #print(lines[5])
                    elif(lines['type']=='Time Series'):
                        tf_rnn(lines)
                    elif(lines['type']=='MultiClass'):
                        tf_multiclass(lines)
                    elif(lines['type']=='Text Classification'):
                        tf_nlp_classify(lines)
                    elif(lines['type']=='Text Prediction'):
                        tf_nlp_predict(lines)

                if(lines['framework']=='PyTorch'):
                    if(lines['type']=='Classification'):
                        pyt_ANN(lines)
                    elif(lines['type']=='Time Series'):
                        pyt_RNN(lines)
                    elif(lines['type']=='MultiClass'):
                        pyt_multiclass(lines)
                    elif(lines['type']=='Text Classification'):
                        pyt_textclassify(lines)
                    elif(lines['type']=='Text Prediction'):
                        pyt_textpredict(lines)


                checkoutputfiles()
                toelectronmain("Display_Message : Generating python autocode")
                if(error_occured is False):
                    coder(lines)
                toelectronmain("Final_Message : Kill Script")
                reset()
        except Exception as e:
                toelectronmain('Error Encountered : Please Check file or hyperparameter settings')



#start process
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        toelectronmain(e)
