import pandas as pd
import matplotlib.pyplot as plt

# BOXPLOT

def boxplot_dataset(x, together=True):
    """
    Esta función visualiza los valores numéricos de un dataframe en un gráfico de tipo boxplot. 

    :param x: pd.DataFrame a visualizar
    :param together: Valor boolear que especifica si se quieren ver todas las variables en un mismo boxplot o no
    """ 
    if not isinstance(x, pd.DataFrame):
        raise Exception("Sólo se admiten pd.DataFrames para visualizar")
    # Filtrar las columnas numéricas
    numeric_cols = x.select_dtypes(include='number')

    if together:
        # Mostrar todos los boxplots juntos
        plt.clf()
        numeric_cols.boxplot()
        plt.show()
    else:
        for i,name in enumerate(numeric_cols.columns):
            plt.clf()
            # Mostrar un boxplot para cada columna numérica por separado
            numeric_cols[[name]].boxplot()
            plt.title(name)
            plt.show()
            if i<len(numeric_cols.columns)-1:
              input("Presiona Enter para ver el siguiente gráfico")

# PIEPLOT

def pieplot(x, only_categorical=True):
    """
    Esta función visualiza las variables de un dataframe distintos gráficos tipo pieplot ("tarta"). 

    :param x: pd.DataFrame a visualizar
    :param only_categorical: Valor boolear que especifica si sólo se quieren visualizar sólo las variables categóricas o no
    """ 
    if not isinstance(x, pd.DataFrame):
        raise Exception("Sólo se admiten pd.DataFrames para visualizar")
    if only_categorical:
        # Filtrar las columnas categóricas
        categorical_cols = x.select_dtypes(exclude='number')
    else:
        categorical_cols = x

    for i,name in enumerate(categorical_cols.columns):
        # Crear un gráfico de pastel para cada columna categórica
        value_counts = categorical_cols[name].value_counts()
        labels = value_counts.index
        sizes = value_counts.values
        plt.clf()
        plt.pie(sizes, labels=labels, autopct='%1.1f%%')
        plt.title(f'{name} variable distribution')
        plt.axis('equal')  # Asegura que el gráfico de pastel sea circular

        plt.show()
        
        if i<len(categorical_cols.columns)-1:
             input("Presiona Enter para ver el siguiente gráfico")
