SEED = 42
PNLN = 0.0001
m = 64
tau = 1
delta = 2

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
#########################################################################################
# Esta funcion es la encargada de crear la base de datos a partir de una serie de timpo #
# recive Ts que es la serie de tiempo, m que es tamaño de la ventana, tau que es el     #
# desplazamiento de la ventana, y delta que el la cantidad de datos a predecir          #
#########################################################################################

def TS2DBS(TS, m, tau, delta):
    N = TS.size
    DBSize = N-delta-(m-1)*tau
    DB = np.array([TS[slice(i,i+m*tau,tau)] for i in range(DBSize)])
    Y = np.array([TS[slice(i+m*tau,i+m*tau+delta,tau)] for i in range(DBSize)])
    return DB,Y

########################################################################################
# La funcion recive una serie de tiempo la proporcion tomada de los datos para el test #
# set y la proporcion del test set que ha de ser utilizada para crear al validationset #
# ademas internamente estandariza la serie de tiempo de entrada para mejorar el rendi- #
# miento durante el entrenamiento.                                                     #
########################################################################################

def GenersConjuntos(serie, test_siz, val_size):
    # Estazndarizacion de los Datos
    serie_std = (serie - serie.mean()) / serie.std()
    # Creacion de la base de datos
    X,y = TS2DBS(serie_std,m,tau,delta)
    # Cracion del Train_set
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_siz, random_state=SEED)
                    
    # Cracion del validation set
    X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=val_size, random_state=SEED)
    
    return X_train, X_val, X_test, y_train, y_val, y_test 


#######################################################################################
# Esta funcion se encarga de plotear las graficas del historial de entrenamiento      #
#######################################################################################

def plot_history(history):
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

    plt.subplot(224)
    #plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Precisión')
    plt.plot(hist['epoch'], hist['acc'],
             label='Train ACC')
    plt.plot(hist['epoch'], hist['val_acc'],
             label = 'Val ACC')
    #plt.ylim([0,20])
    plt.legend(loc="lower right")

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