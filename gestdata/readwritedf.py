import pandas as pd
import csv


def write_dataframe(x,file_path):
    """
    Esta función escribe y guarda un pd.DataFrame de entrada en el
    fichero especificado. El formato es el siguiente (los símbolos <> indican que
    la información es opcional):
    
    númerofilas,númerocolumnas<,nombre>
    #Rownames: nombrefila1,nombrefila2,nombrefila3...
    #Colnames: nombrecolumna1,nombrecolumna2,nombrecolumna3...
    datos,de,columna,1,separados,por,coma
    datos,de,columna,2,separados,por,coma
    datos,de,columna,3,separados,por,coma
    ...
  
    :param x: pd.DataFrame a guardar. También puede ser una lista, donde el primer elemento especifica el nombre y el segundo el dataset a guardar
    :param file_path: Dirección y nombre del fichero en donde guardar los datos
    """
    # Si es una lista, hay que desglosarla
    if type(x)==list:
        if type(x[0])==str:
            name = ","+x[0]
        else:
            raise Exception("El primer elemento de la lista debe ser un string")
        x = x[1]   
    else:
        name = ""
    if not isinstance(x, pd.DataFrame):
        raise Exception("Sólo se admiten pd.DataFrames para guardar")
    if 0 in x.shape:
        raise Exception("Alguna dimensión de x es 0")
    with open(file_path,"w") as f:
        # Escribe las línea siguiendo el formato
        f.write(",".join([str(x.shape[0]),"".join([str(x.shape[1]),name])])+"\n")
        f.write("#Rownames: "+",".join([str(elem) for elem in list(x.index)])+"\n")
        f.write("#Colnames: "+",".join([str(elem) for elem in list(x.columns)])+"\n")
        lines = "\n".join([",".join([str(elem) for elem in list(x[colname])]) for colname in x])
        f.write(lines)

def read_dataframe(file_path):
    """
    Esta función lee un dataset desde un fichero especificado que haya sido 
    creado con el método gemelo write_dataset
  
    :param file_path: Dirección y nombre del fichero a leer
    :return: pd.DataFrame construido a partir de los datos leídos en el fichero. Si el dataset está nombrado, devolverá una lista donde el primer elemento especificará el nombre y el segundo el dataset
    """ 
    with open(file_path,"r") as f:
        # Separa la primera línea por comas
        line1_info = f.readline().replace("\n","").split(",")
        if len(line1_info)<2:
            raise Exception("La primera línea tiene que tener uno de los formatos:\n",
         "númerodefilas,númerodecolumnas\n",
         "númerodefilas,númerodecolumnas,nombre")
        # Consigue las dimensiones del dataset
        size = [int(elem) for elem in line1_info[:2]]
        num_rows, num_cols = size
        # Descubre si se ha escrito el nombre del dataset. En tal caso, guardalo
        var_name = None 
        if len(line1_info)>2:
            var_name= line1_info[2]   
        result = pd.DataFrame()
        
        # Consigue los nombres de las filas
        line2_info = f.readline().replace("\n","").split(" ")
        if line2_info[0] != "#Rownames:":
            raise Exception("El formato de la segunda línea tiene que ser: #Rownames: nombresdefilas")
        rownames = line2_info[1].split(",")
        if len(rownames) != num_rows:
            raise Exception("La dimensión leída no corresponde con el número de filas")
        
        # Consigue los nombres de las columnas
        line3_info = f.readline().replace("\n","").split(" ")
        if line3_info[0] != "#Colnames:":
            raise Exception("El formato de la tercera línea tiene que ser: #Colnames: nombresdecolumnas")
        colnames = line3_info[1].split(",")
        if len(colnames) != num_cols:
            raise Exception("La dimensión leída no corresponde con el número de filas")

        # Lee los datos del dataset columna a columna y guárdalos
        data = f.readlines()
        for i,col in enumerate(data):
            result[colnames[i]] = col.replace("\n","").split(",")
            
        result.index = rownames
        if var_name != None:
            return [var_name,result]
        return result

def dataframe2csv(x,file_path):
    """
    Esta función escribe y guarda un pd.DataFrame de entrada en el
    fichero de tipo csv especificado. El formato es el siguiente:
    
    ,nombrecolumna1,nombrecolumna2,nombrecolumna3...
    nombrefila1,datos,de,fila,1,separados,por,coma
    nombrefila2,datos,de,fila,2,separados,por,coma
    nombrefila3,datos,de,fila,3,separados,por,coma
    ...
  
    :param x: pd.DataFrame a guardar.
    :param file_path: Dirección y nombre del fichero csv en donde guardar los datos
    """
    if file_path[-4:] != ".csv":
        raise Exception("El fichero no es de tipo .csv")
    if not isinstance(x, pd.DataFrame):
        raise Exception("Sólo se admiten pd.DataFrames para guardar")
    if 0 in x.shape:
        raise Exception("Alguna dimensión de x es 0")
    
    with open(file_path,"w",newline='') as f:
        csvwriter = csv.writer(f) 
        # Escribe los nombres de las columnas
        csvwriter.writerow([""]+list(x.columns))
        # Escribe cada fila de datos con su nombre
        for i in range(x.shape[0]):
            row = [x.index[i]] + list(x.iloc[i,:])
            csvwriter.writerow(row)

def csv2dataframe(file_path):
    """
    Esta función lee un dataset desde un fichero csv especificado.
  
    :param file_path: Dirección y nombre del fichero csv a leer
    :return: pd.DataFrame construido a partir de los datos leídos en el fichero. 
    """ 
    if file_path[-4:] != ".csv":
        raise Exception("El fichero no es de tipo .csv")
    
    with open(file_path,"r",newline='') as f:
        csvreader = csv.reader(f)
        # Lee la primera línea (los nombres de las columnas)
        colnames = next(csvreader)[1:]
        rownames = []
        data = []
        # Lee cada fila de datos
        for row in csvreader:
            rownames.append(row[0])
            data.append(row[1:])
        # Crea el pd.DataFrame a partir de los datos leídos
        result = pd.DataFrame(data)
        result.columns = colnames
        result.index = rownames

        return result

