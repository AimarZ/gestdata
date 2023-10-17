import numpy as np
import pandas as pd
import copy

# STANDARIZE

def __standarize_vector(x):
    """
    Esta función estandariza los valores de un vector numérico
    """ 
    if x.dtype.kind not in 'biufc':
      raise Exception("Los valores tienen que ser numéricos")
    return((x-np.mean(x))/np.std(x))

def __standarize_tabla(x):
    """
    Esta función estandariza los valores de las columnas numéricas de un dataframe
    """ 
    aux = x.select_dtypes(include='number')    
    aux = aux.apply(__standarize_vector, axis=0) 
    result = copy.deepcopy(x)
    result[aux.columns] = aux
    return result
  
def standarize(x):
    """
    Esta función estandariza los valores de un vector y las columnas de
    un dataframe para que la media de cada uno sea 0 y la desviación estándar 1

    :param x: Vector o pd.DataFrame que estandarizar
    :return: Array estandarizado de la misma dimensión que x
    """ 
    if isinstance(x, np.ndarray) or isinstance(x, pd.Series):
      return __standarize_vector(x)  
    elif isinstance(x, pd.DataFrame):
      return __standarize_tabla(x)
    else:
      raise Exception("x tiene que ser un np.array, un pd.Series o un pd.DataFrame")
        

# NORMALIZE

def __normalize_vector(x):
    """
    Esta función normaliza los valores de un vector numérico
    """ 
    if x.dtype.kind not in 'biufc':
      raise Exception("Los valores tienen que ser numéricos")
    return((x-np.min(x))/np.max(x-np.min(x)))

def __normalize_tabla(x):
    """
    Esta función normaliza los valores de las columnas numéricas de un dataframe
    o matriz
    """ 
    aux = x.select_dtypes(include='number')    
    aux = aux.apply(__normalize_vector, axis=0) 
    result = copy.deepcopy(x)
    result[aux.columns] = aux
    return result

def normalize(x):
    """
    Esta función normaliza los valores de un vector y las columnas de
    un dataframe para que esten en un rango [0,1]

    :param x: Vector o pd.DataFrame que normalizar
    :return: Array normalizado de la misma dimensión que x
    """ 
    if isinstance(x, np.ndarray) or isinstance(x, pd.Series):
      return __normalize_vector(x)  
    elif isinstance(x, pd.DataFrame):
      return __normalize_tabla(x)
    else:
      raise Exception("x tiene que ser un np.array, un pd.Series o un pd.DataFrame")