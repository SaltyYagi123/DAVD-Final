"""
CHULETA DESARROLLO APLICACIONES VISUALES - PYTHON - GRAFICAS CON PLOTLY 
YAGO F*CKIN TOBIO - 2023

* Estructura de la chuleta: 
    - Librerias (Linea 11)
        ? - Librerias estandar (Linea 15)
        ? - Librerias de Dash (Linea 27)
        ? - Librerias Machine Learning (Linea 34)
    - Cosas genericas: Funciones, Clases, Diccionarios, Listas (Linea 64)
        ? - Funciones (Linea 67)
        ? - Clases (Linea 77)
        ? - Diccionarios (Linea 97)
        ? - Listas (Linea 105)
    - Pandas - Cargar el dataset, Pre-procesamiento, Manipulación (Linea 113)
        ? - Cargar el dataset (Linea 129)
        ? - Convertirlo a un dataframe de pandas (Linea 136)
        ? - Pre-visualización del dataframe (Linea 141)
        ? - Información del dataframe (Linea 147)
        ? - Acceso a los datos (Linea 160)
        ? - Manipulación de los datos (Linea 171)
        ? - Guardar un dataframe en varios formatos (Linea 202)
    - Figuras Plotly (Linea 200)
        ? - Scatter Plot (Linea 231)
        ? - Line Plot (Linea 236)
        ? - Bar Plot (Linea 240)
        ? - Histograma (Linea 244)
        ? - Box Plot (Linea 253)
        ? - Violin Plot (Linea 263)
        ? - Heatmap (Linea 267)
        ? - Mapamundi (Linea 278)
        ? - Correlation Matrix/Mapa de Calor (Linea 284)
        ? - Hacer subplots (Linea 300)
    - Modelos Machine Learning (Linea 327)
        ? - Preparación de los datos (Linea 329)
        ? - Regresión Lineal (Linea 339)
        ? - Regresión Logistica (Linea 385)
        ? - Random Forest (Linea 432)
        ? - K-Means Clustering (Linea 496)
            ? - PCA (Linea 626)
        ? - SVM (Linea 682)
        ? - Stochastic Gradient Descendant (Al final de todo)
    - Dash (Linea 700)

PAGADME UNA CERVEZA POR ESTO PORFAVOR

"""

#!#################################################################
# ! - LIBRERIAS                                                  ##                 
#!#################################################################

# * Librerias estandar: 
import pandas as pd                 # Para la manipulación de los datasets
import numpy as np                  # Para operaciones matematicas
import sklearn.datasets             # Para los datasets

import plotly.express as px         # Plotly
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pickle                       # Esto sirve para el K-Means Clustering y poder guardar el modelo en el formator correcto. 
from plotly.subplots import make_subplots

# * Librerias de Dash
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State

# * Librerías Machine Learning
# * Modelos Existentes (Regresión Lineal, Logistica, Random Forest, K-Means Clustering, SVM)

from sklearn.model_selection import train_test_split 
# ? - Regresión Lineal + Logisticas
from sklearn.linear_model import LinearRegression, SGDClassifier, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
# T-test
from scipy.stats import ttest_ind

# ? - Random Forest (+ Escalamiento necesario)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ? - Metricas de Precisión
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    classification_report,
    confusion_matrix,
    silhouette_score
)

from sklearn.pipeline import Pipeline # ? - Para generar modelos de machine learning con escalamiento automatico

#!#################################################################
# ! - COSAS GENERICAS: FUNCIONES, CLASES, DICCIONARIOS, LISTAS   ##
#!#################################################################
# * Funciones
def function(param1, param2):
    """
    Descripción de la función
    """
    return param1 + param2

function(1, 2) #? - Llamada a la función. Resultado: 3

# * Clases
class Clase:
    """
    Descripción de la clase
    """
    # ? - Constructor - Inicializa la clase y sus parametros
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
    # ? - Método - Función dentro de la clase
    def method(self):
        """
        Descripción del método
        """
        return self.param1 + self.param2

clase = Clase(1, 2)
clase.method() #? - Resultado: 3

# * Diccionarios
diccionario = {
    "key1": "value1",
    "key2": "value2"
}
diccionario["key1"] #? - Resultado: value1

# * Listas
lista = ["value1", "value2"]
# 5 Países Europeos
paises_europeos = ['Spain', 'Italy', 'France', 'Germany', 'United Kingdom']
# 5 Paises Americanos
paises_americanos = ['United States', 'Canada', 'Argentins', "Mexico", "Brasil"]

lista[0]                                                                  #? - Hacer referencia al primer elemento de la lista
lista[0:1]                                                                #? - Hacer referencia a los primeros 2 elementos de la lista
lista.append("value3")                                                    #? - Añadir un elemento al final de la lista
lista.pop(0)                                                              #? - Eliminar el primer elemento de la lista
lista.remove("value3")                                                    #? - Eliminar el elemento "value3" de la lista
paises = paises_americanos + paises_europeos                              #? - Concatenar listas
df_global_group = [df_europe, df_america, df_africa, df_asia, df_oceania] #? - Lista de dataframes
global_df = pd.concat(df_global_group)                                    #? - Concatenar dataframes de una lista


#!#################################################################
# ! - PANDAS - CARGAR EL DATASET, PRE-PROCESAMIENTO, MANIPULACION #
#!#################################################################
# * Cargar un dataset en varios formatos 
dataset = sklearn.datasets.load_boston()        #? - Formato sklearn
dataset = pd.read_csv("dataset.csv")            #? - Formato CSV
dataset = pd.read_excel("dataset.xlsx")         #? - Formato Excel
dataset = pd.read_json("dataset.json")          #? - Formato JSON

# * Convertirlo a un dataframe de pandas
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)  #? - Para el dataset de sklearn
df = pd.DataFrame(dataset)                                      #? - Para los otros
df = pd.DataFrame(columns=["columna1", "columna2"])             #? - Crear un dataframe vacio

# * Pre-visualización del dataframe
df.head()       #? - Visualización de los primeros 5 elementos del dataframe
df.tail()       #? - Visualización de los últimos 5 elementos del dataframe
df.sample(5)    #? - Visualización de 5 elementos aleatorios del dataframe

# * Información del dataframe
df.info()       #? - Información del dataframe
df.shape        #? - Dimensiones del dataframe
df.isna().sum() #? - Suma de valores nulos por columna del dataframe
df.dtypes       #? - Tipos de datos de las columnas del dataframe

df.columns      #? - Las columnas del dataframe
df.index        #? - Los indices del dataframe
df.values       #? - Valores del dataframe
df.describe()   #? - Estadisticas del dataframe

# * Acceso a los datos
df["columna"]                                           #? - Acceder a una columna del dataframe
df[["columna1", "columna2"]]                            #? - Acceder a varias columnas del dataframe
# TODO: - Acceso con iloc es con indices numericos solo, con loc es con indices numericos y strings
df.iloc[0:5, 0:3]                                       #? - Acceder a las primeras 5 filas y las columnas de la 1 a la 3 del dataframe
df.loc[0:5, "columna1":"columna3"]                      #? - Acceder a las primeras 5 filas y las columnas de la 1 a la 3 del dataframe

df_europe = df.loc[df['country'].isin(europe)]          #? - Filtrar el dataframe por una lista de paises
df_europe = df_europe.reset_index(drop=True)            #? - Resetear los indices del dataframe

# * Manipulación de los datos
df[df["columna"] == "value"]                            #? - Filtrar el dataframe por una condición
df.sort_values(by="columna", ascending=False)           #? - Ordenar el dataframe por una columna de forma descendente
df.drop("columna", axis=1)                              #? - Eliminar una columna del dataframe
df.dropna()                                             #? - Eliminar filas con valores nulos
df.drop_duplicates()                                    #? - Eliminar filas duplicadas
df.fillna("value")                                      #? - Rellenar valores nulos con un valor
df.rename(columns={"columna": "columna2"})              #? - Renombrar una columna del dataframe
df["columna"].astype("int")                             #? - Cambiar el tipo de dato de una columna del dataframe
df["columna"].apply(lambda x: x + 1)                    #? - Aplicar una función a una columna del dataframe
df["columna"].map({"value1": 1, "value2": 2})           #? - Mapear valores de una columna del dataframe
df["columna"] = df['columna1'].sub(df['columna2'])      #? - Restar dos columnas del dataframe
df['hit'] = df["streams"] > df["streams"].quantile(0.5) #? - Crear una columna con una condición booleana y con quantiles 

df_filtered = df.nlargest(10, 'columna')                #? - Filtrar el dataframe por los 10 paises con mayor valor en la columna
df_sorted = df_filtered.sort_values(by='gdpp', ascending = False)

# ? - Añadir filas a un dataframe con una funcion en un bucle for 
for i in range(10):
    df = df.append({"columna1": i, "columna2": i + 1}, ignore_index=True) #? - ignore_index=True para que no se repitan los indices

# ? -  Modificar filas de un dataframe con una funcion en un bucle for
for i in range(10):
    df.loc[i, "columna1"] = 1
    # OR: 
    df.loc[i] = [i, funcion(i), funcion(i + 1)]

df_paises = df.copy()[df["country"].isin(paises)]   #? - Filtrar el dataframe por una lista de paises
df_paises = df_paises.reset_index(drop=True)        #? - Resetear los indices del dataframe

# * Guardar un dataframe en varios formatos
df.to_csv("dataset.csv") #? - Formato CSV
df.to_excel("dataset.xlsx") #? - Formato Excel
df.to_json("dataset.json") #? - Formato JSON

#!#################################################################
#! - FIGURAS PLOTLY                                               #
#!#################################################################

# * Scatter Plot - Para ver la relación entre dos variables
fig = px.scatter(df, x="columna1", y="columna2", color="columna3", size="columna4", hover_data=["columna5"])
fig.update_layout(title="Countries with the Highest GDPP",) #? - Titulo de la figura
fig.show()

# * Line Plot - Para ver la evolución de una variable en el tiempo
fig = px.line(df, x="columna1", y="columna2", color="columna3", hover_data=["columna4"])
fig.show()

# * Bar Plot - Para ver la relación entre dos variables categoricas
fig = px.bar(df, x="columna1", y="columna2", color="columna3", hover_data=["columna4"])
fig.show()

# * Histograma - Para ver la distribución de una variable
fig = px.histogram(df, x="streams",
                   labels = {"streams": "Reproducciones"},
                   histnorm='probability density',
                   nbins = 100,
                   opacity=0.6)
fig.update_traces(marker_color = "darkorange")
fig.show()

# * Box Plot - Para ver la distribución de una variable categorica
fig = px.box(df, x="columna1", y="columna2", color="columna3", hover_data=["columna4"])
fig.show()

# ? - T-test -> Test de hipotesis para ver si dos grupos tienen una diferencia significativa
group1 = df[df['more_two_artists'] == True]['streams']
group2 = df[df['more_two_artists'] == False]['streams']

ttest_ind(group1.dropna(), group2.dropna(), trim=.2)

# * Violin Plot - Para ver la distribución de una variable categorica ( Combinación Box Plot + Histograma)
fig = px.violin(df, x="columna1", y="columna2", color="columna3", hover_data=["columna4"])
fig.show()

# * Heatmap - Para ver la correlación entre variables
# ? - Debemos de eliminar las columnas que no sean numericas o pasarlo a factores numericos
df = df.drop(["columna1", "columna2"], axis=1)          #? - Eliminar columnas
df["columna3"] = pd.Categorical(df["columna3"]).codes   #? - Pasar a factores numericos aquellas columnas

df['key'] = df['key'].astype('category')
df['mode'] = df['mode'].astype('category')

fig = px.imshow(df.corr())
fig.show()

# * Mapamundi
# ? - Debe de tener una columna con los nombres de los paises que encajen el codigo ISO de 3 letras
fig = px.choropleth(df, locations="columna1", color="columna2", hover_data=["columna3"])
fig.show()


# * Correlation Matrix/Mapa de Calor - Para ver la correlación entre varias variables
# Calculate the correlation matrix
corr = df.corr(numeric_only=True)

# Create a heatmap using Plotly's heatmap function
figure = ff.create_annotated_heatmap(
        x = list(corr.columns),
        y = list(corr.index),
        z = np.array(corr),
        annotation_text = np.round(np.array(corr)*100),
        hoverinfo='z',
        colorscale=px.colors.sequential.GnBu
)

figure.show()

# * Hacer subplots 
fig = make_subplots(rows = 1,
                    cols = 2,
                    subplot_titles=("Distribución reproducciones", "Diferencia #artistas"))

fig.add_trace(
    go.Scatter(x=df["columna1"], #! - En caso de que quieras meter otra grafica aquí, la sustituyes con go.Scatter/go.Bar/go.Line/go.Histogram/etc...
               y=df["columna2"], 
               mode="markers"), 
    row=1, col=1
    )
fig.add_trace(
    go.Scatter(x=df["columna1"], #! - En caso de que quieras meter otra grafica aquí, la sustituyes con go.Scatter/go.Bar/go.Line/go.Histogram/etc...
               y=df["columna2"], 
               mode="markers"), 
    row=1, col=2
    )

fig.update_xaxes(title_text = "Nº Clusters", row = 1, col = 1)                                          #? - Titulo del eje x para row 1, col 1
fig.update_xaxes(title_text = "Nº Clusters", row = 1, col = 2)                                          #? - Titulo del eje x para row 1, col 2
fig.update_yaxes(title_text = "WCSS", row = 1, col = 1)                                                 #? - Titulo del eje y para row 1, col 1
fig.update_yaxes(title_text = "Silhouette Score", row = 1, col = 2)                                     #? - Titulo del eje y para row 1, col 2
fig.update_layout(height=700, width=1500, title_text="Optimum Number of Clusters", showlegend = False)  #? - Titulo + tamaño de la figura
fig.show()


#!#################################################################
# ! - MODELOS MACHINE LEARNING                                    #
#!#################################################################
# * Preparación de los datos
# ? - Separar los datos en X e y
# ! - Acuerdate de cambiarle el nomber a la columna que quieres predecir
X = df.drop("target", axis=1) #? - Quitamos la columna que queremos predecir. axis=1 para eliminar una columna, axis=0 para eliminar una fila.
y = df["target"]              #? - Nos quedamos con la columna que queremos predecir.

# ? - Separar los datos en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 123) #? - test_size=0.2 para que el 20% de los datos sean de test

#*########################################################################
# * Regresión Lineal -  Ideal para relaciones lineales entre variables. ##  
#*########################################################################

lm = Pipeline(steps=[
  ('scaler', StandardScaler()),
  ("lm", LinearRegression()),
])

lm = lm.fit(X_train, y_train)
print("Variance explanation R^2 = {}".format(round(lm.score(X, y),2)))


# *  - Visualizar los coeficientes mas importantes de la regresión lineal
lmc = lm.named_steps['lm'].coef_
print(lmc)

# Plot for LM
objects = X.columns
y_pos = np.arange(len(objects))
coefficients = lmc

fig = go.Figure()
# Agrego las trazas necesarias
fig.add_trace(
    go.Bar(
        x = coefficients,
        y = objects,
        name = "Coeficientes",
        orientation='h'
    )
)

# Actualizo el diseño
fig.update_layout(title = "LM coefficients importance", xaxis_title = "Coeficientes", yaxis_title = "Variables")

# Muestro la figura
fig.show()

# ? - Predecir con el modelo
y_pred = lm.predict(X_test)
# ? - Evaluar el modelo
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_pred))
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

#*############################################################################
# * Regresión Logistica - Adecuada para problemas de clasificación binaria. ##
#*############################################################################

# ? - Crear el modelo
clf_log = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ("glm", LogisticRegression(max_iter=10000, tol=0.1)),
])


# clf_log = LogisticRegression(max_iter=10000, tol=0.1)
clf_log.fit(X_train, y_train)

predictions = clf_log.predict(X_test)

print("Classification report")
print(classification_report(y_test, predictions))

print("Confusion matrix")
print(confusion_matrix(y_test, predictions))

lmc = clf_log.named_steps['glm'].coef_

# Plot for LM
objects = X.columns
y_pos = np.arange(len(objects))
coefficients = lmc[0]

fig = go.Figure()

# Agrego las trazas necesarias
fig.add_trace(
    go.Bar(
        x = coefficients,
        y = objects,
        name = "Coeficientes",
        orientation='h'
    )
)

# Actualizo el diseño
fig.update_layout(title = "GLM coefficients importance", xaxis_title = "Coeficientes normalizados", yaxis_title = "Variables")

# Muestro la figura
fig.show()

#*###############################################################################################################################
# * Random Forest - Excelente para Reconocimiento de patrones con un gran número de características y evitar el sobreajuste.   ##    
#*###############################################################################################################################

# ? - Asegurate de haber hecho el split. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #? - test_size=0.2 para que el 20% de los datos sean de test

rf = RandomForestRegressor().fit(X_train, y_train)

rf = Pipeline(steps=[
  ('scaler', StandardScaler()),
  ("rf", RandomForestRegressor()),
])

rf = rf.fit(X_train, y_train)

print("Variance explanation R^2 = {}".format(round(rf.score(X, y),2)))

# * Alternativa al Random Forest - Escalar los datos (Solo para Random Forest)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ? - Crear el modelo
model = RandomForestRegressor()
# ? - Entrenar el modelo
model.fit(X_train, y_train)
# ? - Predecir con el modelo
y_pred = model.predict(X_test)

# * Visualizar la importancia de las variables
rfc = rf.named_steps['rf'].feature_importances_
print(rfc)

# Plot for RF
objects = X.columns
y_pos = np.arange(len(objects))
coefficients = rfc

fig = go.Figure()

# Agrego las trazas necesarias
fig.add_trace(
    go.Bar(
        x = coefficients,
        y = objects,
        name = "Coeficientes",
        orientation='h'
    )
)

# Actualizo el diseño
fig.update_layout(title = "RF coefficients importance", xaxis_title = "Coeficientes", yaxis_title = "Variables")
# Muestro la figura
fig.show()

# ? - Predecir con el modelo
y_pred = rf.predict(X_test)

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_pred))
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))


#*#####################################################################################################
# * K-Means Clustering  - K-Means para agrupación                                                    ##           
#*#####################################################################################################
# ? - Crear el modelo
model = KMeans(n_clusters=3)
# ? - Entrenar el modelo
model.fit(X_train)
# ? - Predecir con el modelo
y_pred = model.predict(X_test)

# ! - Cuidado con esta función que viene copiada de una solución. 
def get_optimum_cluster_number(df_k_means):
  """
    This function calculates the optimum minimum of clusters and plots the results from the search.

    Parameters
    ----------
    df_k_means : Pandas Dataframe
      Processed data that we will introduce to our model.

    Returns
    -------
    """

  # ELBOW METHOD
  wcss=[]
  for i in range(2,20):
      km=KMeans(n_clusters=i)
      km.fit(df_k_means)
      wcss.append(km.inertia_)

  # SILHOUETTE SCORE
  sil = []
  for k in range(2, 30):
    kmeans = KMeans(n_clusters=k, random_state=123)
    kmeans.fit(df_k_means)
    labels = kmeans.labels_
    sil.append(silhouette_score(df_k_means, labels, metric='euclidean'))

  # PLOTS
  fig = make_subplots(rows=1, cols=2,
                      subplot_titles=('Elbow Method', 'Silhouette Method'))
  fig.add_trace(
      go.Scatter(x=list(range(1,20)), y=wcss),
      row=1, col=1
  )
  fig.add_trace(
      go.Scatter(x=list(range(2,30)), y=sil),
      row=1, col=2
  )
  fig.update_xaxes(title_text = "Nº Clusters", row = 1, col = 1)
  fig.update_xaxes(title_text = "Nº Clusters", row = 1, col = 2)
  fig.update_yaxes(title_text = "WCSS", row = 1, col = 1)
  fig.update_yaxes(title_text = "Silhouette Score", row = 1, col = 2)
  fig.update_layout(height=700, width=1500, title_text="Optimum Number of Clusters", showlegend = False)
  fig.show()


  # OPTIMAL NUMBER OF CLUSTERS - Max(Silhouette Score)
  max_value = max(sil)
  index = sil.index(max_value)
  k = index + 2                                                # Starts on 0 the index and clusters on 2
  print("Number of optimal clusters is: " + str(k))

# Datos a introducir al modelo
df_k_means = df.copy()                      #! - Modificar el dataset correspondiente 
df_k_means = df_k_means.set_index('column') #! - Establecer la columna que se va a usar como indice

# Se estandarizan los datos de entrada
scaler = StandardScaler()
df_scaled_k_means = scaler.fit_transform(df_k_means)

# ? - Calculamos el número óptimo de clusters en base a la función anterior
get_optimum_cluster_number(df_scaled_k_means)

# KMEANS - 4 Clusters
km = KMeans(n_clusters=5, random_state = 123)

# Fitting the input data
km.fit(df_scaled_k_means)

# Predicting the clusters of the input data
cluster_labels = km.predict(df_scaled_k_means)

# Adding the cluster labels to the original dataframe
df_results_k_means = df_k_means.copy()
df_results_k_means ["cluster"] = cluster_labels 
df_results_k_means.head(10)                 #? - Visualización de la tabla junto al cluster correspondiente. Modificar el numero de filas que se quieren ver

# ? - Visualizar las caracteristicas de los clusters 
# ! - No se como se supone que esto lo tenemos que hacer en un examen pero bueno.
# Estandarizamos los resultados del modelo obtennido
X_std = scaler.fit_transform(df_results_k_means)
X_std = pd.DataFrame(X_std, columns=df_results_k_means.columns)

# Normalizamos las variables del modelo obtenido
df_variables_model = df_results_k_means.copy().drop('cluster', axis=1)
df_variables_model =(df_variables_model-df_variables_model.min())/(df_variables_model.max()-df_variables_model.min())
df_variables_model["cluster"] = df_results_k_means['cluster']

# Calculamos la desviación de cada cluster respecto de la media
X_mean = pd.concat([pd.DataFrame(df_variables_model.mean().drop('cluster'), columns=['mean']), 
                   df_variables_model.groupby('cluster').mean().T], axis=1)

X_dev_rel = X_mean.apply(lambda x: round((x-x['mean'])/x['mean'],2)*100, axis = 1)
X_dev_rel.drop(columns=['mean'], inplace=True)
X_mean.drop(columns=['mean'], inplace=True)
X_std_mean = pd.concat([pd.DataFrame(X_std.mean(), columns=['mean']), 
                   X_std.mean().T], axis=1)

X_std_dev_rel = X_std_mean.apply(lambda x: round((x-x['mean'])/x['mean'],2)*100, axis = 1)
X_std_dev_rel.drop(columns=['mean'], inplace=True)
X_std_mean.drop(columns=['mean'], inplace=True)

# Dataframe para la visualización
df_vis = pd.DataFrame(X_dev_rel.T)

# Visualizamos la desviación de cada cluster respecto de la media
fig = px.bar(df_vis, 
             x = df_vis.index, 
             y = df_vis.columns, 
             barmode='group')
fig.update_layout(title = "Características de los diferentes clusters obtenidos", 
                  yaxis_title = "Desviación en % con respecto a la media", 
                  xaxis_title = "Número de cluster")

fig.show()

# ! - Tras esto explicar como se interpretan los clusters y cuales son sus caracteristicas distinctivas.

#*############################################################################
# * PCA  (Con origen de K-Means)  - y PCA para reducción de dimensionalidad ##           
#*############################################################################

"""
PCA (Análisis de Componentes Principales) es una técnica estadística utilizada para reducir 
la dimensionalidad de los conjuntos de datos, aumentando la interpretabilidad pero al mismo 
tiempo minimizando la pérdida de información. <------------------------------------------->
"""
# ! Cuidado con esta función que viene de un examen 
def plotPCA(df_results_k_means):
    """
    This function reduces the dimesionality of the clusters and returns a figure

    Parameters
    ----------
    df_result_k_means : Pandas Dataframe
        Results of the KMeans model.
    cluster_colors:
        Predefined colors for each cluster

    Returns
    -------
    figure : Plotly figure object
        Figure object containing the cluster PCA.
    """
    # PCA
    pca = PCA(n_components=2) # * Puedes modificar el numero de dimensiones maximas que quieres que tenga el PCA, pero yo creo que con 2 es suficiente. 
    df_results_k_means_r=pca.fit_transform(df_results_k_means)
    df_pca = pd.DataFrame(df_results_k_means_r, columns=["pc1","pc2"]) # ! - Cuidado con las dimensiones pero mantenlo a 2.
    df_pca['cluster'] = df_results_k_means['cluster'].values

    # Representamos el PCA
    data = []
    y_categories=np.sort(df_pca["cluster"].unique())
    color_dict={0:"blue",1:"red",2:"yellow", 3:"green"}
    # Representamos el PCA

    for cat in y_categories:
        data.append(
            go.Scatter(
                x=df_pca[df_pca["cluster"]==cat]["pc1"],
                y=df_pca[df_pca["cluster"]==cat]["pc2"], #! - Cuidado con las dimensiones pero mantenlo a 2. 
                mode= "markers", 
                name=str(cat),
                marker_color=color_dict[cat]
            ) 
        )
    layout= go.Layout(title="PCA Iris", xaxis_title="PC1", yaxis_title="PC2")
    fig = go.Figure(data = data, layout = layout)

    return fig

pca_plot = plotPCA(df_results_k_means)
pca_plot.show()

#*##################################################################################################################################
# * SVM - Efectivas en espacios de alta dimensión y en casos donde el número de dimensiones es mayor que el número de muestras.   ##  
#*##################################################################################################################################

# ? - Crear el modelo
model = SVC()
# ? - Entrenar el modelo
model.fit(X_train, y_train)
# ? - Predecir con el modelo
y_pred = model.predict(X_test)

#* Stochastic Gradient Descendant - Este es bastante xd pero tenia que incluirlo
clf_sgd = SGDClassifier(
    loss='hinge', penalty='l2',
    alpha=1e-3, random_state=42,
    max_iter=5, tol=None)

clf_sgd.fit(X_train, y_train)

predictions = clf_sgd.predict(X_test)

print("Classification report")
print(classification_report(y_test, predictions, target_names=dataset["target_names"]))

print("Confusion matrix")
print(confusion_matrix(y_test, predictions))
