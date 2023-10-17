import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from gestdata.extras import is_binary

# Para que no salten unos errores absurdos de deprecación de pandas
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning) 

def __varianza_vector(x):
    """
    Esta función calcula la varianza de un vector numérico
    """ 
    if x.dtype.kind not in 'biufc':#"biufc"  
        raise Exception("x tiene que ser un vector numérico")
    return np.sum((x-np.mean(x))**2) / (len(x)-1)

def __varianza_tabla(x):
    """
    Esta función calcula la varianza de las columnas numéricas de un dataframe
    """ 
    aux =  x.select_dtypes(include='number') 
    varianzas = aux.apply(__varianza_vector, axis=0) 
    return varianzas

def varianza(x):
    """
    Esta función calcula la varianza de un vector numérico o de 
    cada columna numérica de un dataframe

    :param x: Vector o pd.DataFrame de valores sobre los cuales calcular la varianza
    :return: En el caso de que x sea un vector, devolverá el valor de varianza. Si no, devolverá un vector que recoja las varianzas de las columnas numéricas de x
    """
    if 0 in x.shape:
        raise Exception("Alguna dimensión de x es 0")
    if isinstance(x, np.ndarray) or isinstance(x,pd.Series):
      return __varianza_vector(x)  
    elif isinstance(x, pd.DataFrame):
      return __varianza_tabla(x)
    else:
      raise Exception("x tiene que ser un np.array, un pd.Series o un pd.DataFrame")

def __entropy_vector(x):
    """
    Esta función calcula la entropía de un vector
    """ 
    probs = pd.value_counts(x)/len(x)
    logits = -probs * np.log2(probs)
    entropia = np.sum(logits)
    return entropia

def __entropy_tabla(x):
    """
    Esta función calcula la entropía para cada columna de un dataframe
    """ 
    return x.apply(__entropy_vector, axis=0) 

def entropy(x):
    """
    Esta función calcula la entropía de un vector o de cada una de 
    las columnas de una matriz o dataframe

    :param x: Vector o pd.DataFrame sobre el que calcular la entropía
    :return: En el caso de que x sea un vector, devolverá el valor de entropía. Si no, devolverá un vector que recoja las entropías de las columnas de x
    """ 
    if 0 in x.shape:
        raise Exception("Alguna dimensión de x es 0")
    if isinstance(x, np.ndarray) or isinstance(x, pd.Series):
      return __entropy_vector(x)  
    elif isinstance(x, pd.DataFrame):
      return __entropy_tabla(x)
    else:
      raise Exception("x tiene que ser un np.array, un pd.Series o un pd.DataFrame")

def ROC(x,attribute_variable,predict_variable):
    """
    Esta función calcula los puntos de la curva ROC para 
    la predicción de una variable binaria y visualiza el gráfico

    :param x: Todos los datos recogidos en un pd.DataFrame
    :param attribute_variable: Nombre de la columna que corresponde a los atributos numéricos que se utilizan para hacer la predicción
    :param predict_variable: Nombre de la columna binaria que se quiere predecir 
    :return: Las coordenadas X e Y de los puntos del gráfico ROC recogidos en un pd.DataFrame
    """ 
    if not isinstance(x, pd.DataFrame):
      raise Exception("x debe ser un pd.DataFrame")
    # Intenta utilizar el nombre de la variable objetivo para indexar
    try:
        binary_variable = x[predict_variable]
    except:
        raise Exception("El nombre de la variable objetivo no es correcto")
    if not is_binary(binary_variable):
        raise Exception("La variable objetivo no es de tipo binario")
    # Intenta utilizar el nombre de la variable atributo para indexar
    try:
        attributes = x[attribute_variable]
    except:
        raise Exception("El nombre de la variable de atributos no es correcto")
    
    if attributes.dtype.kind not in 'biufc':
        raise Exception("Los atributos tienen que ser numéricos")
    
    # Ordena el dataframe teniendo en cuenta los atributos
    x_sorted = x.sort_values(by = attribute_variable)
    binary_variable = np.array(x_sorted[predict_variable])
    attributes = np.array(x_sorted[attribute_variable])
    # aux = min(variable de atributo)-1
    aux = attributes[0] -1
    coords_x, coords_y = [],[]
    # El numero de puntos será igual al número de 'rows'+1
    for i in range(x.shape[0]+1):
      # Haz una predicción suponiendo un punto de corte
      punto_corte = ([aux]+list(attributes))[i]
      predictions = attributes>punto_corte
      # Mira si la predicción y el objetivo son iguales
      equal_comp = predictions == binary_variable
      # Calcula las métricas
      TP = np.sum(equal_comp[predictions])
      TN = np.sum(equal_comp[np.logical_not(predictions)])
      FP = np.sum(np.logical_not(equal_comp[predictions]))
      FN = np.sum(np.logical_not(equal_comp[np.logical_not(predictions)]))
      TPR = TP/(TP+FN)
      FPR = FP/(FP+TN)
      # Añade el TPR y FPR a las coordenadas
      coords_x.append(FPR)
      coords_y.append(TPR)
    coords = pd.DataFrame({"x":coords_x,"y":coords_y})
    plt.plot("x","y",'', data=coords, marker='o')
    plt.xlabel('1-specifity')
    plt.ylabel('sensitivity')
    plt.show()
    return(coords)


def AUC(df):
    """
    Esta función calcula el área de debajo de la curva ROC, es decir,
    el valor AUC (Area Under the Curve)

    :param df: pd.DataFrame que contiene las coordenadas X e Y de los puntos que conforman la curva ROC
    :return: Valor que corresponde al AUC
    """
    # Calcula la integral dados los puntos (x1,y1), (x2,y2) ...
    x = np.array(df.iloc[:,0])
    y = np.array(df.iloc[:,1])
    delta_x = np.diff(x)
    y = y.reshape((y.shape[0],1))
    mean_y = np.mean(np.concatenate((y[:-1],y[1:]),axis=1), axis = 1)
    return abs(np.sum(delta_x*mean_y))   



def metricas(x,attribute_variable=None,predict_variable=None):
    """
    Esta función calcula las varianzas de las columnas numéricas y
    las entropías de las categóricas de un dataframe. Si se especifican 
    una variable predictora y una columna objetivo, también calcula la curva 
    ROC y el valor AUC.

    :param x: Todos los datos recogidos en un pd.DataDrame
    :param attribute_variable: Nombre de la columna que corresponde a los atributos que se utilizan para hacer la predicción (None por defecto)
    :param predict_variable: Nombre de la columna binaria que se quiere predecir (None por defecto)
    :return: pd.DataFrame que contiene todas las métricas
    """
    if not isinstance(x, pd.DataFrame):
        raise Exception("x debe ser un pd.DataFrame")
        
    # Calcula la varianza de las columnas numéricas    
    vari = varianza(x)
    result = pd.DataFrame(columns = vari.index)
    result.loc[0] = vari
    # Busca las columnas categóricas
    categorical_cols = []
    for col_name in x.columns:
        if col_name not in result.columns:
            categorical_cols.append(col_name)
    # Calcula la entropía de las columnas categóricas   
    entrp = entropy(x[categorical_cols])
    
    result[list(entrp.index)] = entrp
    
    # Calcula el ROC y AUC si se quiere
    if attribute_variable != None and predict_variable != None:
        coords = ROC(x,attribute_variable,predict_variable)
        result["AUC"] = AUC(coords)     
    
    return result[x.columns] 


def filtrar(x,condition,use_entropy=True,use_varianza=False):
    """
    Esta función elimina las columnas de un dataframe que no 
    cumplan una condición dada respectiva a su varianza o entropía.
    Si únicamente se utiliza la varianza para filtrar, las variables categóricas
    no resultaran afectadas. De igual manera, utilizando la entropía las columnas
    numéricas se mantendrán.

    :param x: pd.DataFrame que se quiere filtrar
    :param condition: Condición que debe cumplir la métrica de una columna para que se conserve
    :param use_entropy: Si se quiere utilizar la entropía como métrica de filtrado (True por defecto)
    :param use_varianza: Si se quiere utilizar la varianza como métrica de filtrado (False por defecto)
    :return: pd.DataFrame con las columnas filtradas
    """
    if not isinstance(x, pd.DataFrame):
        raise Exception("x debe ser un pd.DataFrame")
    if type(condition) != type("str"):
        raise Exception("La condición tiene que ser de tipo 'str'")
    if use_entropy == False and use_varianza==False:
        raise Exception("No se ha especificado ninguna métrica")
    # Calcular las métricas dependiendo de los parámetros
    if use_entropy and use_varianza:
        metrics = metricas(x)
        # Se han considerado todas las columnas
        not_taken = np.array([])
    elif use_entropy==True:
        # Calcular la entropía solo para las variables categóricas
        entrp = entropy(x.select_dtypes(exclude='number'))
        metrics = pd.DataFrame(columns = entrp.index)
        metrics.loc[0] = entrp
        # No se han considerado las columnas numéricas
        not_taken = np.array(x.select_dtypes(include='number').columns)
    else:
        vari = varianza(x)
        metrics = pd.DataFrame(columns = vari.index)
        metrics.loc[0] = vari
        # No se han considerado las columnas categóricas
        not_taken = np.array(x.select_dtypes(exclude='number').columns)
    # Trata de evaluar la condición dada
    try:
        # Crea una línea string para ejecutar para crear una máscara boolear de las columnas que cumples la condición
        statement = "np.array([elem "+condition+" for elem in metrics.iloc[0,:]])"
        # Ejecuta el string como si fuera una orden y asigna el resultado a mask
        mask = eval(statement)
        # Consigue los nombres de las columnas que se mantendrán
        keep_columns = np.array(metrics.columns)[mask]
    except:
        raise Exception("La condición especificada no es válida")
    # Añadir las columnas que no se han utilizado para evaluar la condición
    keep_columns = np.concatenate((keep_columns,not_taken),axis=0)
    # Ordenar las columnas para que estén en el mismo orden que en el input
    keep_columns =  keep_columns[np.argsort([list(x.columns).index(item) for item in keep_columns])]
    return x[keep_columns]