from jupyterthemes import jtplot
from tensorflow import set_random_seed

# Constantes
SEED = 42
set_random_seed(SEED)

# Importacion de las bibliotecas necesarias

from sklearn.metrics import mean_squared_error 
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from keras import regularizers
from tensorflow import keras
import numpy as np
import pandas as pd

#########################################################################################
# Esta función se encarga de crear un modelo de MLP sin regularización los parametros   #
# de entrada son:                                                                       #
# neuronas (número de neuronas en cada capa) ejemplo.- [10, 12, 13] en este caso se     #
# no se agrega a la lista el numero de neuronas de salida                               #
# activations (funciones de activación en cada capa) ejemplo.- ['relu', 'relu', 'relu'] #
# de igual manera no se incluye la en la lista la funcion de transferencia de la capa   #
# de salida.                                                                            #
# m es el número de entradas a la red, y show sirve para mostrar o no el modelo         #
#                                                                                       #
#########################################################################################

def ModMLPN(neurons, activations, m, show=True): # Modelo de la red neuronal a utilizar
    # Se comienza a declarar la red neuronal
    if neurons.shape != activations.shape:
        print('No corresponde numero de neuronas y funciones de activación')
        return 0
    inputs = keras.Input(shape=(m))
    x = layers.Dense(neurons[0], activation=activations[0])(inputs)
    for i in range(1, len(neurons)):   
            x = layers.Dense(neurons[i], activation=activations[i])(x)
    outputs = layers.Dense(1, activation='linear')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='forecasting')
    
    if(show):
        model.summary()
    
    return model

#########################################################################################
# En esta funcion la unica diferencia con respecto a la anterior es que en este modelo  #
# incluye la regularización L2.                                                         #
# lamda es la penalizacion que se le da a los pesos                                     #
#########################################################################################

def ModMLPR(neurons, activations, m, lamda=0.0001, show=True): # Modelo de la red neuronal a utilizar
    # Se comienza a declarar la red neuronal
    if neurons.shape != activations.shape:
        print('No corresponde numero de neuronas y funciones de activación')
        return 0
    inputs = keras.Input(shape=(m))
    x = layers.Dense(neurons[0], activation=activations[0], activity_regularizer=regularizers.l2(lamda))(inputs)
    for i in range(1, len(neurons)):   
            x = layers.Dense(neurons[i], activation=activations[i], activity_regularizer=regularizers.l2(lamda))(x)
    outputs = layers.Dense(1, activation='linear', activity_regularizer=regularizers.l2(lamda))(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='forecasting')
    
    if(show):
        model.summary()
    
    return model

########################################################################################
# Para unificar las dos funciones anteriores se creo esta por tanto cuando no se le da #
# una lamda regresa un modelo sin regularización                                       #
########################################################################################

def ModMLP(neuronas, activaciones, m, lamda=0, show=True):
    neuronas = np.array(neuronas)
    activaciones = np.array(activaciones)
    
    if 0 == lamda:
        model = ModMLPN(neuronas, activaciones, m)
    else:
        model = ModMLPR(neuronas, activaciones, m, lamda)
    return model

########################################################################################
# En esta función se entrena el modelo, cuando no se le pasa el parametro patien el    #
# entrenamiento no utiliza el early-stopping                                           #
########################################################################################


def fitMLP(model, X_train, y_train, X_val, y_val, X_test, y_test, epoc=30, patien=-1):
    
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=['mae', 'mse'])  
    
    if -1 == patien:
        history = model.fit(X_train, y_train, batch_size=64, epochs=epoc, validation_data=(X_val, y_val))
    else:
        callback = EarlyStopping(monitor='val_loss', patience=patien, restore_best_weights=True)
        history = model.fit(X_train, y_train, batch_size=64, callbacks=[callback], epochs=epoc, validation_data=(X_val, y_val))

    test_scores = model.evaluate(X_test, y_test, batch_size=64)
    print('Test loss:', test_scores[0])
    print('Test mae:', test_scores[1])
    print('Test mse:', test_scores[2])
    
    return history

#########################################################################################
# La funcion agrega ruido a una muestra determinada, esta funcion es un pequeño modulo  #
# que es utilizada por otra función que se encarga de adicionar el ruido al train_set   #
#########################################################################################

def addNoise(train_set, rep, mean, sigma, i):
    i = int(i/rep)
    
    l = len(train_set[0])
    e = np.random.normal(mean, sigma, size=(l))
    Nueva_muestra = train_set[i,:] + e
    
    return Nueva_muestra

########################################################################################
# Al igual que las función anterior esta tambien es un submodulo de otra que explico   #
# más adelante.                                                                        #
########################################################################################


def returnTarget(y_train, rep, i):
    i = int(i/rep)
    return y_train[i]
    

########################################################################################
# Esta función se encarga de adicionar el ruido al train_set sus parametros de entrada #
# son:                                                                                 #
# SIGMA la cual es utilizada para la funcion de ruido normal                           #
# NUMREP que indica la cantidad de veces que se utiliza cada vector para generar       #
# muestras                                                                             #
########################################################################################


def trainNoise(X_train, y_train, SIGMA, NUMREP=2):
    
    X_train_e =  [addNoise(X_train, NUMREP, 0, SIGMA,i)
               for i in range(len(X_train)*NUMREP)
             ]
    y_train_e = [returnTarget(y_train, NUMREP, i)
               for i in range(len(X_train)*NUMREP)
             ]
    X_train_e = np.array(X_train_e)
    y_train_e = np.array(y_train_e)

    return X_train_e, y_train_e

########################################################################################
# Funciones de error, únicamente reciben el y_target y el y_predict                    #
########################################################################################

def SMAPE(y_target, y_predic):
    
    porcent = 100/len(y_target)
    smape = abs(y_target - y_predic) / (y_target + y_predic)
    smape = smape.sum() * porcent
    
    return smape

def MAPE(y_target, y_predic):
    
    porcent = 100/len(y_target)
    mape = abs(y_target - y_predic) / y_predic
    mape = mape.sum() * porcent
    
    return mape


########################################################################################
# Esta función se utiliza para mostrar la gráfica de y_target vs y_predict             #
########################################################################################

def graphPrediction(Y_inv, Y_pred_inv):
    plt.figure(figsize=(16, 8))
    plt.xlabel('Prediction vs Real')
    plt.ylabel('Magnitude')
    plt.plot(Y_inv,marker='.', label="Real")
    plt.plot(Y_pred_inv, 'r', label="Prediction")
    plt.legend(loc="lower right")
    plt.show()


#######################################################################################
# Esta funcion se encarga de plotear las graficas del historial de entrenamiento      #
#######################################################################################

def plotHistory(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
  
    plt.figure(figsize=(16, 12))
    plt.subplot(221)
    #plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error LOSS')
    plt.plot(hist['epoch'], hist['loss'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_loss'],
             label = 'Val Error')

    #plt.ylim([0,20])
    plt.legend(loc="upper right")

    plt.subplot(222)
    plt.xlabel('Epoch')
    plt.ylabel('mean_absolute_error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train MAE')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label = 'Val MAE')
    #plt.ylim([0,20])
    plt.legend(loc="upper right")
    
    plt.subplot(223)
    plt.xlabel('Epoch')
    plt.ylabel('mean_squared_error')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train MSE')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label = 'Val MSE')
    #plt.ylim([0,20])
    plt.legend(loc="upper right")
  
    plt.show()