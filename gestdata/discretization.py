import numpy as np
import pandas as pd
import copy
from gestdata.correlmutua import mutual_information_vectors
from gestdata.extras import is_binary


def discretize(x, cut_points):
  """
  Esta función discretiza los valores de un vector dados los puntos de corte

  :param x: Vector numérico a discretizar
  :param cut_points: Vector numérico con los de puntos de corte
  :return: pandas.core.arrays.categorical.Categorical con los valores de x discretizados en intérvalos
  """ 
  return pd.cut(x, bins=[float("-Inf")]+list(cut_points)+[float("Inf")])

def discretizeEW_vector(x, num_bins):
  """
  Esta función discretiza todos los valores numéricos
  de un vector siguiendo el algoritmo "equal width"

  :param x: Vector numérico sobre el que aplicar el algoritmo
  :param num_bins: Número de intervalos con los que discretizar
  :return: pandas.core.arrays.categorical.Categorical con los valores de x discretizados en intérvalos
  """ 
  if x.dtype.kind not in 'biufc':
    raise Exception("Los valores tienen que ser numéricos")
  minimo = np.min(x)
  maximo = np.max(x)
  tamaño = (maximo-minimo)/num_bins
  puntos_corte = np.full(num_bins-1,minimo) + np.arange(1,num_bins) * tamaño
  return(discretize(x,puntos_corte))

def __discretizeEW_tabla(x, num_bins):
  """
  Esta función discretiza los valores de las columnas numéricas de un
  dataframe usando el algoritmo "equal width"
  """ 
  # Prepara el resultado con la misma estructura que x pero rellenado de NA
  aux = x.select_dtypes(include='number') 
  aux = aux.apply(discretizeEW_vector, axis=0, args=(num_bins,))
  result = copy.deepcopy(x)
  result[aux.columns] = aux
  return(result)

def discretizeEW(x, num_bins):
  """
  Esta función discretiza todos los valores numéricos
  de un vector o un dataframe siguiendo el algoritmo "equal width"

  :param x: Vector numérico o un pd.DataFrame sobre el que aplicar el algoritmo
  :param num_bins: Número de intervalos con los que discretizar
  :return: Si x es un vector, devolverá un objeto de tipo pandas.core.arrays.categorical.Categorical con los valores de x discretizados en intérvalos. Si x es un pd.DataFrame, devolverá un pd.DataFrame con los valores numéricos discretizados
  """ 
  if num_bins<1:
    raise Exception("El parámetro num_bins tiene que ser mayor que 0")
  if isinstance(x, np.ndarray) or isinstance(x, pd.Series):
    return discretizeEW_vector(x,num_bins)  
  elif isinstance(x, pd.DataFrame):
    return __discretizeEW_tabla(x,num_bins)
  else:
    raise Exception("x tiene que ser un np.array, un pd.Series o un pd.DataFrame")


def discretizeEF_vector(x, num_bins):
  """
  Esta función discretiza todos los valores numéricos
  de un vector siguiendo el algoritmo "equal frequency"

  :param x: Vector numérico sobre el que aplicar el algoritmo
  :param num_bins: Número de intervalos con los que discretizar
  :return: pandas.core.arrays.categorical.Categorical con los valores de x discretizados en intérvalos
  """ 
  if x.dtype.kind not in 'biufc':
    raise Exception("Los valores tienen que ser numéricos")
  
  puntos_corte = pd.Series(x).quantile(np.arange(0,1,1/num_bins)[1:])
  return(discretize(x,puntos_corte))

def __discretizeEF_tabla(x, num_bins):
  """
  Esta función discretiza los valores de las columnas numéricas de un
  dataframe usando el algoritmo "equal frequency"
  """ 
  aux = x.select_dtypes(include='number')
  aux = aux.apply(discretizeEF_vector, axis=0, args=(num_bins,))
  result = copy.deepcopy(x)
  result[aux.columns] = aux
  return(result)

def discretizeEF(x, num_bins):
  """
  Esta función discretiza todos los valores numéricos
  de un vector o un dataframe siguiendo el algoritmo "equal frequency"

  :param x: Vector numérico o un pd.DataFrame sobre el que aplicar el algoritmo
  :param num_bins: Número de intervalos con los que discretizar
  :return: Si x es un vector, devolverá un objeto de tipo pandas.core.arrays.categorical.Categorical con los valores de x discretizados en intérvalos. Si x es un pd.DataFrame, devolverá un pd.DataFrame con los valores numéricos discretizados
  """ 
  if num_bins<2:
    raise Exception("El parámetro num_bins tiene que ser mayor que 1")
  if isinstance(x, np.ndarray) or isinstance(x, pd.Series):
    return discretizeEF_vector(x,num_bins)  
  elif isinstance(x, pd.DataFrame):
    return __discretizeEF_tabla(x,num_bins)
  else:
    raise Exception("x tiene que ser un np.array, un pd.Series o un pd.DataFrame")


def entropy_binning(X, Y, num_bins, tries=5, random_cut=False, offset=None):
    """
    Esta función discretiza los valores de un vector con el objetivo de maximizar la información mutua
    respecto a otra variable. 

    :param X: Vector numérico a discretizar
    :param Y: Vector respecto al cual se quiere maximizar la información mutua
    :param num_bins: Número de intervalos con los que discretizar
    :param tries: Cuantos intentos de discretización se harán con distintos puntos de corte
    :param random_cut: Valor boolear que especifica si los puntos de corte se calcularán de manera aleatoria o no
    :param offset: En el caso de que no haya aleatoriedad, en cada intento se incrementarán los puntos de corte por este valor numérico. Por defecto es None, y el offset será la desviación estándar de los datos dividido por el numéro de intentos
    :return: Lista con 3 valores: 
      1- pandas.core.arrays.categorical.Categorical con los valores de x discretizados en intérvalos
      2- Lista de puntos de corte calculados
      3- El valor de información mutua máxima obtenida
    """ 
    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise ValueError("X and Y must be numpy arrays")
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same length")
    if not is_binary(Y):
        raise ValueError("The target variable Y must be binary")
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("The feature variable X must be numeric")

    # Se calculan los valores mínimo y máximo de X
    min_val = np.min(X)
    max_val = np.max(X)

    best = -1
    result = None

    # Si random_cut es False, se calculan los puntos de corte de manera equiespaciada
    if not random_cut:
        bin_size = (max_val - min_val) / num_bins
        if offset is None:
            offset = np.std(X) / tries
        lower_bound = min_val - offset * ((tries - 1) // 2)

    # Se realizan los intentos de discretización
    for i in range(tries):
        # Si random_cut es True, se calculan los puntos de corte de manera aleatoria
        if random_cut:
            cut_points = np.sort(np.random.uniform(min_val, max_val, num_bins - 1))
        else:
            # Se calculan los puntos de corte equiespaciados
            cut_points = np.array([lower_bound + j * bin_size for j in range(1, num_bins)])
            lower_bound += offset

        # Se discretiza X y se calcula la información mutua
        X_discr = np.digitize(X, cut_points)
        mutual_info = mutual_information_vectors(X_discr, Y)
        
        # Se guarda el resultado si es el mejor hasta el momento
        if mutual_info > best:
            best = mutual_info
            result = [X_discr, cut_points, mutual_info]

    result[0] = discretize(X,result[1])

    return result