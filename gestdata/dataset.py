import numpy as np
import pandas as pd

class Dataset:
  """La clase Dataset sirve para almacenar datos en una estructura de tabla. Su funcionamiento es similar al de un DataFrame de Pandas, pero más limitado.
  El Dataset está formado por una lista de columnas, cada una de las cuales es una lista de valores. Todos los valores de una columna tienen que ser del mismo tipo.
  La clase Dataset sólo admite los tipos de datos int, float, bool y str."""

  allowed_types = [int,float,bool,str]
  def __init__(self):
    """Constructor de la clase Dataset. Crea un Dataset vacío."""
    self._data = []
    self.colnames = []
    # Diccionario que relaciona el nombre de una columna con su índice, esto permite acceder a las columnas por su nombre en tiempo O(1)
    self._name2col = {}
    self.dtypes = []
    self.shape = [0,0]
  
  def __str__(self):
    """Método para imprimir el Dataset."""
    s = "Dataset of shape: "+ str(self.shape) + "\n"
    s += "Data by columns:\n"
    for i in range(len(self._data)):
      s+= f" {self.colnames[i]} = {str(self._data[i])} , dtype =  {self.dtypes[i].__name__}\n"
    s = s[:-1]
    return(s)
    
  def __slice2indexes(self,slc,max):
    """Método auxiliar para convertir un slicing en una lista de índices."""
    try:
      return(list(range(slc.start if slc.start != None else 0, slc.stop if slc.stop != None else max, slc.step if slc.step != None else 1)))
    except:
      raise Exception("Los índices del slicing no son correctos")
  
  def add_col(self,data, name = None):
    """Método para añadir una columna al Dataset.

    :param data: Python list de valores de la columna a añadir. Todos los valores tienen que ser del mismo tipo.
    :name: nombre de la columna, opcional
    """ 

    if not isinstance(data,list):
      raise Exception("El parámetro data tiene que ser una Python list")
    if len(data)<1:
      raise Exception("data es una lista vacía. No hay nada para añadir")
    if self.shape[0] != 0 and self.shape[0] != len(data):
      raise Exception(f"La columna tiene que tener {self.shape[0]} valores")

    for i in range(len(data)-1+(len(data)==1)):
      if type(data[i]) not in self.allowed_types:
        raise Exception("Sólo se admiten valores de tipo: "+ str(self.allowed_types))
      if len(data)>1 and type(data[i]) != type(data[i+1]):
        raise Exception("Todos los valores de data tienen que ser del mismo tipo")
    
    self._data.append(np.array(data))
    index = self.shape[1]
    # Si no se especifica nombre, se le asigna uno por defecto
    if name == None:
      name = f"V{index}"
    self.colnames.append(str(name))
    self._name2col[str(name)] = index
    self.shape[1]+=+1
    self.shape[0] = len(data)
    self.dtypes.append(type(data[0]))


  def add_cols(self,data,names=None):
    """Método para añadir varias columnas al Dataset.

    :param data: Lista de Python lists que contienen los valores de las columnas a añadir
    :names: Lista de nombres de las columnas, opcional
    """ 
    if not isinstance(data,list):
      raise Exception("El parámetro data tiene que ser una Python list")
    if len(data)<1:
      raise Exception("data es una lista vacía. No hay nada para añadir")
      
    for i in range(len(data)):
      name = None if names==None else names[i]
      self.add_col(data[i],name)
  
  def add_row(self,data):
    """Método para añadir una fila al Dataset.
    
    :param data: Python list de valores de la fila a añadir.
    """

    if not isinstance(data,list):
      raise Exception("El parámetro data tiene que ser una Python list")
    if len(data)<1:
      raise Exception("data es una lista vacía. No hay nada para añadir")
    if self.shape[1] != 0 and self.shape[1] != len(data):
      raise Exception(f"La fila tiene que tener {self.shape[1]} valores")
    for i in range(len(data)):
      if type(data[i]) not in self.allowed_types:
        raise Exception("Sólo se admiten valores de tipo: "+ str(self.allowed_types))
      if self.dtypes != []:
        if type(data[i]) != self.dtypes[i]:
          raise Exception(f"El valor para la columna {self.colnames[i]} tiene que ser de tipo {self.dtypes[i].__name__}")
    
    for i,elem in enumerate(data):
      if self.dtypes == []:
        self._data.append(np.array(elem))
      else:
        self._data[i] = np.append(self._data[i], elem)
    # Si no hay nombres de columnas, se les asigna uno por defecto V0, V1, ...
    if self.colnames == []:
      for i in range(len(data)):
        self.colnames.append(f"V{i}")
        self._name2col[self.colnames[i]] = i
        self.dtypes.append(type(data[i]))
    self.shape[1] = len(data)
    self.shape[0]+=1
    
  def add_rows(self,data):
    """Método para añadir varias filas al Dataset. 
    
    :param data: Lista de Python lists que contienen los valores de las filas a añadir
    """
    if not isinstance(data,list):
      raise Exception("El parámetro data tiene que ser una Python list")
    if len(data)<1:
      raise Exception("data es una lista vacía. No hay nada para añadir")
      
    for i in range(len(data)):
      self.add_row(data[i])
  
  def remove_col(self,i):
    """Método para eliminar una columna del Dataset.
    
    :param i: Índice de la columna a eliminar. También se puede indicar el nombre de la columna.
    """
    if type(i) == str:
      # Sólo 1 índice, string
      i= self._name2col[i]
    self._data.pop(i)
    self.dtypes.pop(i)
    self.shape[1]-=1
    self._name2col.pop(self.colnames[i])
    self.colnames.pop(i)
  
  def remove_cols(self,l):
    """Método para eliminar varias columnas del Dataset.
    
    :param l: Lista de índices de las columnas a eliminar. También se pueden indicar los nombres de las columnas.
    """
    indexes = []
    for elem in l:
      if type(elem)==str:
        indexes.append(self.colnames.index(elem))
      else:
        if elem<0:
          elem = len(self)+elem
        if elem<0:
          raise Exception(f"Un índice negativo es demasiado pequeño. El mínimo admitido es {-len(self)}")
        indexes.append(elem)
    indexes = sorted(indexes,reverse=True)
    for i in indexes:
      self.remove_col(i)
    
  def remove_row(self,i):
    """Método para eliminar una fila del Dataset.
    
    :param i: Índice de la fila a eliminar.
    """
    self.shape[0]-=1
    for j in range(len(self._data)):
      self._data[j]=np.delete(self._data[j],i)
  
  def remove_rows(self,l):
    """Método para eliminar varias filas del Dataset.
    
    :param l: Lista de índices de las filas a eliminar.
    """
    l_sorted = []
    for elem in l:
      if elem<0:
        elem = self.shape[0]+elem
        if elem<0:
          raise Exception(f"Un índice negativo es demasiado pequeño. El mínimo admitido es {-len(self)}")
      l_sorted.append(elem)
    l_sorted = sorted(l_sorted,reverse=True)
    for i in l_sorted:
      self.remove_row(i)
  
  def __len__(self):
    """Método para definir el tamaño del Dataset. El tamaño se condidera como el número de columnas."""
    return len(self._data)
  
  def __getitem__(self, pos):
    """Método para acceder a los elementos del Dataset. Permite la indexación de un Dataset mediante corchetes. Admite el uso de 1 índice o una tupla de 2 índices. El primer índice indica la columna y el segundo la fila. Además, las columnas se pueden indexar mediante el nombre.
    También se puede indexar mediante slicing para acceder a varias filas y/o columnas. Si el output es multidimensional, devuelve un Dataset. Si el output es unidimensional, devuelve una lista de valores. Si el output es un valor único, devuelve el valor.
    
    :param pos: Indexación única o tupla de indexaciones. Estos pueden ser: un índice, nombre de columna (en el caso de la columna) o un slicing. Si se indica un índice o un nombre de columna, se devuelve una lista con los valores de la columna. Si se indica una tupla (i,j), se devuelve el elemento de la fila i y la columna j. Si se indica una tupla (slice,slice), se devuelve un Dataset con las filas y columnas indicadas por los slices.
    :return: Lista de valores, Dataset o valor único, dependiendo del tipo de indexación."""
    # Si es un slicing único, hay que devolver un Dataset con las columnas indicadas por el slicing
    if isinstance(pos, slice):
      # Consigue los índices de las columnas
      indexlist = self.__slice2indexes(pos,max=len(self))
      # Si sólo hay 1 índice, devuelve una lista con los valores de la columna
      if len(indexlist)==1:
        pos = indexlist[0]
      # Si hay más de 1 índice, devuelve un Dataset con las columnas indicadas por el slicing
      else:
        # Crea un Dataset vacío
        result = Dataset()
        # Añade las columnas indicadas por el slicing
        result.add_cols([self[ind] for ind in indexlist], names = self.colnames[pos])
        return result
    # Si es un índice único, devuelve una lista con los valores de la columna
    if type(pos) == int:
      # Sólo 1 índice, int
      return self._data[pos].tolist()
    # Si es un nombre de columna, devuelve una lista con los valores de la columna
    elif type(pos) == str:
      # Sólo 1 índice, string
      return self._data[self._name2col[pos]].tolist()
    # Si es una tupla de 2 elementos, hay que revisar cada uno
    elif len(pos)==2:
      i,j = pos
      # Si el primer elemento es un slicing
      if isinstance(i, slice):
        # Prepara un Dataset vacío
        result = Dataset()
        indexlist1 = self.__slice2indexes(i,max=len(self))
        # Si el segundo elemento es un slicing, devuelve un Dataset con las filas y columnas indicadas por los slices
        if isinstance(j, slice):
          # Para eso, añade las filas enteras de las columnas indicadas por el slicing mediante una llamada recursiva
          result.add_cols([self[ind,j] for ind in indexlist1])
        # Si el segundo elemento es un índice, devuelve una lista con los valores de la columna indicada por el slicing
        else:
          # Para eso, añade la fila indicada por el índice j mediante una llamada recursiva
          result.add_row([self[ind][j] for ind in indexlist1])
        # Renombra las columnas del Dataset resultante 
        result.rename_cols(self.colnames[i])
        return result
      # Si el primer elemento es un nombre único de columna
      if type(i) == str:
        # Convertimos el nombre en un índice
        i = self._name2col[i]
      # Si el segundo elemento es un slicing
      if isinstance(j, slice):
        # Devuelve una lista con los valores de la columna indicada por el índice i y y las filas del slicing j
        indexlist = self.__slice2indexes(j,max=self.shape[0])
        return self._data[i][indexlist].tolist()
      return self._data[i].tolist()[j]
    raise Exception("Parámetro de indexación no válido. Se esperaba un índice, un nombre de columna o un slicing (o una tupla de 2 de estos elementos).")

  def rename_cols(self,l):
    """Método para renombrar las columnas del Dataset."""
    if len(l) != len(self.colnames):
      raise Exception(f"La lista de nombres tiene que tener {len(self.colnames)} valores")
    for i,elem in enumerate(l):
      name = str(elem)
      # Hay que actualizar el diccionario
      self._name2col[name] = self._name2col.pop(self.colnames[i])
      self.colnames[i] = name

  def from_dataframe(self,df):
    """Método para cargar los datos de un DataFrame de Pandas en un Dataset.
    
    :param df: DataFrame de Pandas que contiene los datos."""
    if not isinstance(df,pd.DataFrame):
      raise Exception("El parámetro df tiene que ser un DataFrame de Pandas")
    self.add_cols([df[colname].tolist() for colname in df.columns],names=df.columns.tolist())
  
  def to_dataframe(self):
    """Método para convertir un Dataset en un DataFrame de Pandas.
    
    :return: DataFrame de Pandas que contiene los mismos datos del Dataset."""
    dic = {}
    for i in range(len(self.colnames)):
        dic[self.colnames[i]] = self._data[i]
    return pd.DataFrame(dic,columns=self.colnames)