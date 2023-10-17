import numpy as np
import itertools
import pandas as pd

def is_binary(x):
  """
  Esta función sirve para sabe si un vector contiene datos binarios o no. 

  :param x: Vector tipo np.array o pd.Series a analizar
  :return: True si x contiene sólo valores binarios, False en el caso contrario
  """ 
  if not (isinstance(x,np.ndarray) or isinstance(x,pd.Series)):
    raise Exception("x tiene que ser un np.array o un pd.Series")
  return all([elem in [0,1] for elem in x])

def __conditional_entropy_part(comb,X,Y):
  """
  Función de ayuda para calcular la entropía condicional parcial dadas 
  2 columnas (X,Y) y la combinación de 2 valores que se analiza
  """ 
  a = comb[0]
  b = comb[1]
  # Mira que posiciones tienen los valores que se definen en comb
  condX = (np.logical_or(np.array(X).astype('U')==str(a),np.array(X)==a))
  condY = (np.logical_or(np.array(Y).astype('U')==str(b),np.array(Y)==b))
  both = np.logical_and(condX,condY)
  # Si no hay ninguna 'row' que cumple las 2 condiciones, la entropía condicional es 0
  suma = np.sum(both)
  if suma==0:
    return(float(0))
  # Aplica la fórmula de entropía condicional
  prob_both = suma/len(X)
  prob_cond = suma/np.sum(condY)
  return prob_both*np.log2(1/prob_cond)

def conditional_entropy(X,Y):
  """
  Esta función calcula la entropía condicional entre 2 vectores

  :param X: Primer vector (condicionado)
  :param Y: Segundo vector (condicionante)
  :return: Valor numérico de la entropía condicional
  """ 
  # Recoge que valores puede coger cada columna
  try:
    ux = X.unique()
    uy = Y.unique()
  except:
    ux = np.unique(X)
    uy = np.unique(Y)
  # Calcula todas las combinaciones de valores posibles
  combs = np.asarray(list(itertools.product(ux, uy)))
  
  # Para cada combinación, calcula la entropía condicional parcial y suma todas
  return np.sum(np.apply_along_axis(__conditional_entropy_part, 1, combs, X,Y))