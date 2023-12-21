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
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
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
# * 2. CREAR APP


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# * 3. Pre-procesamiento de los datos
dataset = pd.read_csv("bank-full.csv", delimiter=";")
df = pd.DataFrame(dataset)

df_deposito = df[df['y'] == 'yes']
df_no_deposito = df[df['y'] == 'no']

df_deposito['y'] = df_deposito['y'].apply(lambda x: 1)
df_no_deposito['y'] = df_no_deposito['y'].apply(lambda x: 0)


# Vamos a incluir 4 graficas
# * La de distribución de miembros de depositos por educación y trabajo
# * La de percentage of housing loan
fig4 = make_subplots(rows = 1, cols = 2)
fig4.add_trace(go.Scatter(x = df_deposito['age'], y = df_deposito['balance'], mode = 'markers', name = 'Deposito'), row = 1, col = 1)
fig4.add_trace(go.Scatter(x = df_no_deposito['age'], y = df_no_deposito['balance'], mode = 'markers', name = 'No Deposito'), row = 1, col = 2)
fig4.update_layout(title = 'Balance vs Age')





# Fig 2 -> Sunburst chart pero filtrado
fig2 = px.sunburst(df_deposito, path=['education', 'job'], values='y')
fig2.update_layout(
    title_text='Distribución de miembros de depositos por educación y trabajo',
)

# Fig 3 -> Stacked bard chart 
df_edu = df_deposito.groupby('education').count()['marital'].reset_index()
df_edu_n = df_no_deposito.groupby('education').count()['marital'].reset_index()   

fig3 = make_subplots(rows = 1, cols = 2)

fig3.add_trace(go.Bar(x = df_edu['education'], y = df_edu['marital'], name = 'Deposito'), row = 1, col = 1)
fig3.add_trace(go.Bar(x = df_edu_n['education'], y = df_edu_n['marital'], name = 'No Deposito'), row = 1, col = 2)
fig3.update_layout(title = 'Distribución de Estado Civil')


#Fig1 
def func_corrmatrix(): 

    df['y'] = pd.Categorical(df['y']).codes
    df['default'] = pd.Categorical(df['default']).codes
    df['housing'] = pd.Categorical(df['housing']).codes
    df['loan'] = pd.Categorical(df['loan']).codes
    df['job'] = pd.Categorical(df['job']).codes
    df['education'] = pd.Categorical(df['education']).codes
    df['marital'] = pd.Categorical(df['marital']).codes
    df['contact'] = pd.Categorical(df['contact']).codes
    df['month'] = pd.Categorical(df['month']).codes
    df['poutcome'] = pd.Categorical(df['poutcome']).codes

    corr = df.corr(numeric_only=True)
    fig1 = ff.create_annotated_heatmap(
            x = list(corr.columns),
            y = list(corr.index),
            z = np.array(corr),
            annotation_text = np.round(np.array(corr)*100),
            hoverinfo='z',
            colorscale=px.colors.sequential.GnBu
    )

    return fig1




# * 4. Estructura de la aplicación
app.layout = dbc.Container(
    [
        dbc.Row(
            html.H1(
                "Analisis Exploratorio de los clientes que aceptan depositos en un banco",
                style={"text-align": "center"},
            )
        ),
        dbc.Row(
            [
                html.H3("Selectors"),
                dbc.Col(
                    [
                        dbc.Label("Select education level"),
                        dbc.Checklist(
                            id="radio-indicator",
                            options=[
                                {"label": i, "value": i}
                                for i in df["education"].unique()
                            ],
                            value=list(df["education"].unique())[0],
                            input_class_name="me-2",
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        dbc.Label("Select job"),
                        dcc.Dropdown(
                            id = "my-dropdown", 
                            multi = True, 
                            options = [{"label":x, 'value':x} for x in sorted(df["job"].unique())], 
                            value = ["management"]
                        )
                    ]
                ),
                dbc.Col(
                    [
                        dbc.Label("Select Age Range"), 
                        dcc.RangeSlider(
                            id="year-range",
                            min=df["age"].min(),
                            max=df["age"].max(),
                            step=5,
                            value=[20, 30],
                            marks={
                                year: str(year)
                                if year == df["age"].min() or year == df["age"].max()
                                else f"{str(year)[-2:]}"
                                for year in range(df["age"].min(), df["age"].max(), 5)
                            },
                            allowCross=True,
                        ),
                    ]
                ),
            ], 
        ),
        # * Aqui comenzamos con las gráficas 
        dbc.Row(
            [
                dbc.Col([
                    dbc.Label("Matriz de Correlación"), 
                    dcc.Graph(id='age-balance-histogram', figure = func_corrmatrix())
                ]), 
                dbc.Col([
                    dbc.Label("Pie chart de posibilidad de aceptar deposito dependiendo del trabajo"),
                    dcc.Graph(id='job-pie-chart', figure = fig2),
                ])
            ]
        ),
        dbc.Row([

            dbc.Col([
                dbc.Label("Proporción de estado civil + educación"), 
                dcc.Graph(id = 'marital-status-bar-chart', figure = fig3)
            ]), 
            dbc.Col([
                dbc.Label("Edad vs. Balance Precio"),
                dcc.Graph(id = 'age-balance-scatter', figure = fig4)
            ])
        ]),
    ]
)


# * 5. Call-backs
@app.callback(
    Output(component_id="job-pie-chart", component_property="figure"),
    [Input(component_id='my-dropdown', component_property='value'), 
     Input(component_id='radio-indicator', component_property='value'), 
     Input(component_id='year-range', component_property='value')]
)

def update_sunburst(chosen_job, chosen_education, age_range):
    print(f"Job chosen by user: {chosen_job}")
    print(f"Education chosen by user: {chosen_education}")

    if len(chosen_job) == 0: 
        return {}
    else: 
        df_deposito_edad = df_deposito[df_deposito['age'].between(age_range[0], age_range[1])]
        df_filtered_edu = df_deposito_edad[df_deposito_edad['education'].isin(chosen_education)]
        df_filtered_job = df_filtered_edu[df_filtered_edu['job'].isin(chosen_job)]

        # Plot de la figura 2 
        fig2 = px.sunburst(df_filtered_job, path=['education', 'job'], values='y')
        fig2.update_layout(
            title_text='Distribución de miembros de depositos por educación y trabajo',
        )
        return fig2 


@app.callback(
    Output(component_id="age-balance-scatter", component_property="figure"),
    [Input(component_id="year-range", component_property="value")]
)

def update_scatter(chosen_value): 

    if len(chosen_value) == 0: 
        return {}
    else: 
        df_age_filtered_deposit = df_deposito[df_deposito['age'].between(chosen_value[0],chosen_value[1])]
        df_age_filtered_no_deposit = df_no_deposito[df_no_deposito['age'].between(chosen_value[0], chosen_value[1])]
        fig4 = make_subplots(rows = 1, cols = 2)
        fig4.add_trace(go.Scatter(x = df_age_filtered_deposit['age'], y = df_age_filtered_deposit['balance'], mode = 'markers', name = 'Deposito'), row = 1, col = 1)
        fig4.add_trace(go.Scatter(x = df_age_filtered_no_deposit['age'], y = df_age_filtered_no_deposit['balance'], mode = 'markers', name = 'No Deposito'), row = 1, col = 2)
        fig4.update_layout(title = 'Balance vs Age')

        return fig4

# * 6. El arranque
# ! - QUE NO SE TE OLVIDE
if __name__ == "__main__":
    app.run_server(debug=True)
