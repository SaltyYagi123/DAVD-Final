#*  YAGO TOBIO 2023
#*  PLANTILLA PARA CREAR UN DASHBOARD CON DASH

# * 1. IMPORTAR LIBRERIAS
from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px
import dash_bootstrap_components as dbc # Easier to manage the layout of the app. 
import pandas as pd
from pandas_datareader import wb # Helps to obtain data from API's and create DF's, in this case, world bank

# * 2. CREAR APP
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

indicators = {
    "IT.NET.USER.ZS": "Individuals using the Internet (% of population)",
    "SG.GEN.PARL.ZS": "Proportion of seats held by women in national parliaments (%)",
    "EN.ATM.CO2E.KT": "CO2 emissions (kt)",
}

# * 3. CREAR PRE-PROCESAMIENTO DE DATOS (Lo de abajo es codigo placeholder)
# get country name and ISO id for mapping on choropleth
countries = wb.get_countries()
countries["capitalCity"].replace({"": None}, inplace=True)
countries.dropna(subset=["capitalCity"], inplace=True)
countries = countries[["name", "iso3c"]]
countries = countries[countries["name"] != "Kosovo"]
countries = countries.rename(columns={"name": "country"})

def update_wb_data():
    # Retrieve specific world bank data from API
    df = wb.download(
        indicator=(list(indicators)), country=countries["iso3c"], start=2005, end=2016
    )
    df = df.reset_index()
    df.year = df.year.astype(int)

    # Add country ISO3 id to main df
    df = pd.merge(df, countries, on="country")
    df = df.rename(columns=indicators)
    return df

# * 4. CREAR APP LAYOUT - Con estructura de Bootstrap
    # * - dbc.Container -> Contenedor principal
    # * - dbc.Row -> Filas
    # * - dbc.Col -> Columnas
        # * Y aqui adentro metes las figuras, botones, etc. aquellas que vayan a necesitar callbacks, van a necesitar un ID

    # * Simple app place holder

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [
                    html.H1(
                        "Comparison of World Bank Country Data",
                        style={"textAlign": "center"},
                    ),
                    dcc.Graph(id="my-choropleth", figure={}),
                ],
                width=12,
            )
        ),
        dbc.Row(
            dbc.Col(
                [
                    dbc.Label(
                        "Select Data Set:",
                        className="fw-bold",
                        style={"textDecoration": "underline", "fontSize": 20},
                    ),
                    dcc.RadioItems(
                        id="radio-indicator",
                        options=[{"label": i, "value": i} for i in indicators.values()],
                        value=list(indicators.values())[0],
                        inputClassName="me-2",
                    ),
                ],
                width=4,
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label(
                            "Select Years:",
                            className="fw-bold",
                            style={"textDecoration": "underline", "fontSize": 20},
                        ),
                        dcc.RangeSlider(
                            id="years-range",
                            min=2005,
                            max=2016,
                            step=1,
                            value=[2005, 2006],
                            marks={
                                2005: "2005",
                                2006: "'06",
                                2007: "'07",
                                2008: "'08",
                                2009: "'09",
                                2010: "'10",
                                2011: "'11",
                                2012: "'12",
                                2013: "'13",
                                2014: "'14",
                                2015: "'15",
                                2016: "2016",
                            },
                        ),
                        dbc.Button(
                            id="my-button",
                            children="Submit",
                            n_clicks=0,
                            color="primary",
                            className="mt-4",
                        ),
                    ],
                    width=6,
                ),
            ]
        ),
        dcc.Store(id="storage", storage_type="session", data={}),
        dcc.Interval(id="timer", interval=1000 * 60, n_intervals=0),
    ]
)


# * 5. CREAR CALLBACKS
    # * - Output -> Componente que va a ser actualizado, mediante el ID
    # * - Input -> Componente que va a ser actualizado pero los valores que va a recibir (mediante el ID), 
    # * - Cada vez que surja un cambio en el componente del Input se actualiza el output
@app.callback(Output("storage", "data"), Input("timer", "n_intervals"))
def store_data(n_time):
    dataframe = update_wb_data()
    return dataframe.to_dict("records")

# * - Tambien existen los States, pero no os preocupeis por ahora.
@app.callback(
    Output("my-choropleth", "figure"),
    Input("my-button", "n_clicks"),
    Input("storage", "data"),
    State("years-range", "value"),
    State("radio-indicator", "value"),
)
def update_graph(n_clicks, stored_dataframe, years_chosen, indct_chosen):
    dff = pd.DataFrame.from_records(stored_dataframe)
    print(years_chosen)

    if years_chosen[0] != years_chosen[1]:
        dff = dff[dff.year.between(years_chosen[0], years_chosen[1])]
        dff = dff.groupby(["iso3c", "country"])[indct_chosen].mean()
        dff = dff.reset_index()

        fig = px.choropleth(
            data_frame=dff,
            locations="iso3c",
            color=indct_chosen,
            scope="world",
            hover_data={"iso3c": False, "country": True},
            labels={
                indicators["SG.GEN.PARL.ZS"]: "% parliament women",
                indicators["IT.NET.USER.ZS"]: "pop % using internet",
            },
        )
        fig.update_layout(
            geo={"projection": {"type": "natural earth"}},
            margin=dict(l=50, r=50, t=50, b=50),
        )
        return fig

    if years_chosen[0] == years_chosen[1]:
        dff = dff[dff["year"].isin(years_chosen)]
        fig = px.choropleth(
            data_frame=dff,
            locations="iso3c",
            color=indct_chosen,
            scope="world",
            hover_data={"iso3c": False, "country": True},
            labels={
                indicators["SG.GEN.PARL.ZS"]: "% parliament women",
                indicators["IT.NET.USER.ZS"]: "pop % using internet",
            },
        )
        fig.update_layout(
            geo={"projection": {"type": "natural earth"}},
            margin=dict(l=50, r=50, t=50, b=50),
        )
        return fig


if __name__ == "__main__":
    app.run_server(debug=True)


#*  YAGO TOBIO 2023
#*  PLANTILLA PARA CREAR UN DASHBOARD CON DASH

# * 1. IMPORTAR LIBRERIAS
from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px
import dash_bootstrap_components as dbc # Easier to manage the layout of the app. 
import pandas as pd
from pandas_datareader import wb # Helps to obtain data from API's and create DF's, in this case, world bank

# * 2. Pre-procesamiento de datos (Tipicamente depuración y agrupación de columnas que aparecen en las gráficas)
# * Código placeholder de referencia y relleno, no hacer mucho caso. 
df = pd.read_csv("tweets.csv") #GLOBAL VARIABLES SHOULD NEVER BE ALTERED. 

df["name"] = pd.Series(df["name"]).str.lower()

# Specify the correct date format
df["date_time"] = pd.to_datetime(
    df["date_time"], format="%d/%m/%Y %H:%M", errors="coerce"
)

df = (
    df.groupby(
        [df["date_time"].dt.date, "name"]
    )[  # Group by day, without the hour, and by name.
        ["number_of_likes", "number_of_shares"]
    ]  # This basically means that we want to group the number of likes and the number of shares by date and by user.
    .mean()
    .astype(int)
)

df = df.reset_index()  # Sort

fig = px.line(data_frame=df, x="date_time", y="number_of_likes",
 color="name", log_y=True, height=300)


stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]


# * 3. CREAR APP
app = Dash(__name__, external_stylesheets=stylesheets)


# * 4. CREAR APP LAYOUT - Con estructura de Bootstrap
    # JERARQUÍA DE ESTRUCTURA A SEGUIR
    # * - dbc.Container -> Contenedor principal
    # * - dbc.Div -> Divs
    # * - dbc.Row -> Filas
    # * - dbc.Col -> Columnas
        # * Y aqui adentro metes las figuras, botones, etc. aquellas que vayan a necesitar callbacks, van a necesitar un ID

    # * Simple app place holder - Para que tengas de referencia pero es relleno
app.layout = html.Div(
    [
        html.Div(html.H1("Twitter Likes Analysis of Famous People", 
                         style={"textAlign":"center"}),
                         className="row"),
        html.Div(
            dcc.Graph(id="line-chart", figure=fig, className="row")
        ),
        html.Div(
            dcc.Dropdown(
                id="my-dropdown",
                multi=True,
                options=[{"label": x, "value": x} 
                         for x in sorted(df["name"].unique())],
                value=["taylorswift13", "cristiano", "jtimberlake"],
                style={"color":"green"}
            ),
            className="three columns",
        ), 
        html.Div(
            html.A(
                id="my-link", 
                children="Click here to Visit Twitter", 
                href="https://twitter.com/explore", 
                target="_blank", #Si quieres que se abra en una nueva pestaña, esto debería de ser _blank, en vez de _self
                style={"color":"red", "backgroundColor": "yellow", "fontSize":"40px"}
            ), 
            className="four columns", 
        ),
    ],
    className="row"
)



# App Callbacks ----------------------------------------------------------------------------------------------------------------
# * 5. CREAR CALLBACKS
    # * ESTRUCTURA CALLBACK: 
        # * - Output -> Componente que va a ser actualizado, mediante el ID
        # * - Input -> Componente que va a ser actualizado pero los valores que va a recibir (mediante el ID), 
        # * - Cada vez que surja un cambio en el componente del Input se actualiza el output
    # * FUNCION CALLBACK: 
        # * COMO PARAMETRO COGE EL NUMERO DE INPUTS QUE LE PASES EN EL ORDEN DEFINIDO EN EL @APP CALLBACK
        # * EN BASE A ESO EL OUTPUT A DEVOLVER SIEMPRE DEBE DE SER LA GRAFICA ACTUALIZADA. 

@app.callback(
    Output(component_id="line-chart", component_property="figure"),
    [Input(component_id="my-dropdown", component_property="value")],
)

def update_graph(chosen_value): #Chosen value se refiere a los valores del dropdown list, que se pasan por el 
                                #Component id del callback. Cada vez que se eliga otra opción, se activa. 
    print(f"Values chosen by user: {chosen_value}")

    # Aqui miramos la cantidad de valores que se le pasa para verificar. 
    if len(chosen_value) == 0: 
        return {}
    else: 
        df_filtered = df[df["name"].isin(chosen_value)]
        fig = px.line(
            data_frame = df_filtered, 
            x="date_time", 
            y= "number_of_likes", 
            color="name", 
            log_y=True, 
            labels={
                "number_of_likes":"Likes", 
                "date_time":"Date", 
                "name": "Celebrity",
            },
        )
        return fig

# ! - QUE NO SE TE OLVIDE
if __name__ == "__main__":
    app.run_server(debug=True)
