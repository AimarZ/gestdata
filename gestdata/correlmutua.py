import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from gestdata.extras import conditional_entropy
from gestdata.metrics import entropy



def __coeficiente_correlacion(comb, x):
  """
  Calcula el coeficiente de correlación r dada un array x, 2 índices de columna
  en comb, y cuales son las columnas numéricas especificadas en bool_numeric
  """ 
  i = comb[0]
  j = comb[1]
  a = x.iloc[:,i]
  b = x.iloc[:,j]
  covar = a.cov(b)
  cor_coef = covar/a.std()/b.std()
  new_name = a.name + "-" + b.name
  return (cor_coef, new_name)

def correlacion(x):
  """
  Esta función calcula los coeficientes de correlación de Pearson 
  de entre todos los pares de columnas numéricas de un array

  :param x: pd.DataFrame sobre el cual calcular las correlaciones
  :return: pd.DataFrame que contiene todas las correlaciones entre las columnas numéricas de x
  """ 
  if not isinstance(x, pd.DataFrame):
    raise Exception("x debe ser un pd.DataFrame")
  # aux = columnas numéricas de x
  aux = x.select_dtypes(include='number')   
  num_numeric = aux.shape[1]
  if num_numeric < 2:
    raise Exception("Se necesitan al menos 2 variables numéricas")
  # Para cada combinación de 2 columnas numéricas, calcula la correlación entre ellas
  combs = np.asarray(list(itertools.combinations(list(range(0,num_numeric)), 2)))
  corrs_names = np.apply_along_axis(__coeficiente_correlacion, 1, combs, aux)
  corrs = corrs_names[:,0].astype(float)
  new_names = corrs_names[:,1]
  result = pd.DataFrame(columns = new_names)
  result.loc[0] = corrs
  return result

def mutual_information_vectors(X,Y):
  """
  Esta función calcula la información mutua entre 2 vectores

  :param X: np.array o pd.Series sobre el que calcular la información mutua
  :param Y: np.array o pd.Series sobre el que calcular la información mutua
  
  :return: float con el valor de la información mutua entre X e Y
  """ 
  return entropy(X) - conditional_entropy(X,Y)


def __aux_mutual_information_vectors(comb,x):
  """
  Esta función calcula la información mutua entre 2 vectores del
  pd.DataFrame x, especificados en la tupla comb. Además, 
  devuelve el nombre de las variables que se han utilizado
  """ 
  X = x.iloc[:,comb[0]]
  Y = x.iloc[:,comb[1]]
  new_name = X.name + "-" + Y.name
  mut_info = entropy(X) - conditional_entropy(X,Y)
  return (mut_info, new_name)


def mutual_information(x, only_categorical=False):
  """
  Esta función calcula la información mutua entre las columnas de un dataframe

  :param x: pd.DataFrame sobre el que calcular las informaciones mutuas
  :param only_categorical: Si calcular la información mutua entre todas las columnas (False por default) o sólo con las categóricas (True)
  :return: pd.DataFrame que recoge los valores de información mutua entre las columnas de x
  """ 
  if not isinstance(x, pd.DataFrame):
    raise Exception("x debe ser un pd.DataFrame")
  keep_columns = x.columns
  # Si sólo se quiere calcular para las variables categóricas
  if only_categorical:
    numeric_df = x.select_dtypes(include='number') 
    # Nos quedamos con las columnas que no son numéricas
    keep_columns = [elem for elem in keep_columns if elem not in numeric_df.columns ]
  if len(keep_columns) <2:
    raise Exception("Se necesitan al menos 2 columnas")
  aux = x[keep_columns]
  # Calcula todas las combinaciones de 2 entre las columnas
  combs = np.asarray(list(itertools.combinations(list(range(0,len(keep_columns))), 2)))
  # Calcula la información mutua y el nombre de salida para cada combinación
  mutinfo_names = np.apply_along_axis(__aux_mutual_information_vectors, 1, combs, aux)
  mutinfo = mutinfo_names[:,0].astype(float)
  new_names = mutinfo_names[:,1]
  result = pd.DataFrame(columns = new_names)
  result.loc[0] = mutinfo  
  return result




def mutual_correlation(x):
  """
  Esta función calcula la información mutua entre las columnas 
  categóricas de un dataframe, y la correlación entre las numéricas 

  :param x: pd.DataFrame sobre el cual calcular la información mutua y las correlaciones
  :return: pd.DataFrame numérico que recoge los valores de información mutua y correlación entre todas las columnas de x
  """ 
  if not isinstance(x, pd.DataFrame):
    raise Exception("x tiene que ser un pd.DataFrame")
  numeric_df = x.select_dtypes(include='number') 
  # Prepara la estructura de la matriz de correlación/info.mutua resultado
  result = np.empty((x.shape[1],x.shape[1]))
  result.fill(np.nan)                  
  result = pd.DataFrame(result, columns=x.columns, index = x.columns)
  # Revisa cuantas columnas numéricas y categóricas hay.
  # Si no hay suficientes numéricas, sólo se tendrá que calcular la información mutua.
  # Si no hay suficientes categóricas, sólo se tendrá que calcular la correlación.
  if numeric_df.shape[1]>1:
    metrics = correlacion(x)
  else:
    metrics = None
  if (len(x.columns) - len(numeric_df.columns)) > 1:
    if metrics is None:
      metrics = mutual_information(x,only_categorical = True)
    else:
      metrics = pd.concat([metrics, mutual_information(x,only_categorical = True)], axis=1) 
  
  if metrics is None:
    raise Exception("No hay suficientes columnas numéricas ni categóricas")
  
  # Rellena la matriz de resultado teniendo en cuenta los nombres
  names = metrics.columns
  for i,name in enumerate(names):
    indexers = name.split("-")
    value = metrics[name][0]
    result.loc[indexers[0],indexers[1]] = value
    result.loc[indexers[1],indexers[0]] = value
  
  # La diagonal se compondrá de las entropías para las categóricas,
  # y un coeficiente de correlación=1 para las numéricas
  numeric_columns = numeric_df.columns
  for col in result:
    if col in numeric_columns:
      result.loc[col,col] = 1
    else:
      result.loc[col,col] = entropy(x[col])
  return result

# HEATMAP

def corr_plot(x):
  """
  Esta función visualiza los valores numérico de un dataframe en un gráfico de tipo heatmap. 
  Está pensado para ser utilizado sobre una matriz de correlacion/información mutua,
  pero puede ser utilizado con cualquier otro dataframe.

  :param x: pd.DataFrame numérico a visualizar
  """ 
  plt.clf()
  sns.heatmap(x, cmap = "RdBu_r", vmax=1, annot=True, fmt='.4f')
  plt.show()



