from LinearRegression import Normal
import numpy as np
import pandas as pd

import dash
from dash import Dash, html, callback, Output, Input, State, dcc
import dash_bootstrap_components as dbc
from dash import register_page

import mlflow
import os

#MLFLOW connection
mlflow.set_tracking_uri("http://mlflow.ml.brain.cs.ait.ac.th/")
os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'password'
os.environ["LOGNAME"] = "st125171_Sachin"
mlflow.set_experiment(experiment_name="st125171-a3")

# Initialize the app - incorporate a Dash Bootstrap theme
#app = Dash(__name__, use_pages=True,  external_stylesheets=[dbc.themes.BOOTSTRAP])
dash.register_page(__name__, path='/predictvthree')

#Form Elements
year = html.Div(
    [   
        dbc.Label("*Year (this is a required field)", html_for="year-input", width=4, className="form-label"),
        dbc.Col(dcc.Input(id="year-input", type="number", placeholder="e.g., 2015", className="form-control", required=True, style={'outline':'none'}) ,width=8),
    ],
    className="mb-3"
)

km_driven = html.Div(
    [   
        dbc.Label("Kilometers Driven", html_for="km_driven-input", width=4, className="form-label"),
        dbc.Col(dcc.Input(id="km_driven-input", type="number", placeholder="e.g., 45000", className="form-control"), width=8),
    ],
    className="mb-3"
)

mileage = html.Div(
    [   
        dbc.Label("Mileage (kmpl)", html_for="mileage-input", width=4, className="form-label"),
        dbc.Col(dcc.Input(id="mileage-input", type="number", placeholder="e.g., 15.5", className="form-control"), width=8),
    ],
    className="mb-3"
)

engine = html.Div(
    [   
        dbc.Label("Engine (cc)", html_for="engine-input", width=4, className="form-label"),
        dbc.Col(dcc.Input(id="engine-input", type="number", placeholder="e.g., 1197", className="form-control"), width=8),
    ],
    className="mb-3"
)

submit_model = html.Div([
    dbc.Button(id="submit_model", children="Submit", color="primary", className="me-1", style={'borderRadius': '5px'}),
], style={'marginTop': '20px'})

# Layout with Side-by-Side Form and Prediction
layout = dbc.Container([
    html.H3("Enter Car Details for Price Prediction Ver.3", style={'padding': '20px 0', 'color': '#004d00', 'font-weight': 'bold', 'textAlign': 'center'}),
    
    dbc.Row([
        dbc.Col(
            dbc.Form([
                year,
                km_driven,
                mileage,
                engine,
                submit_model,
            ], className="g-3"),
            width=6,  # Equal width for both columns
        ),
        dbc.Col(
            html.Div([
                html.H3("Prediction Classfication", style={'color': '#004d00', 'font-weight': 'bold', 'textAlign': 'center'}),
                html.Div(id="prediction-output-vthree", className="mt-3", style={
                    'fontSize': '1.5rem', 'color': '#004d00', 'border': '2px solid #004d00', 
                    'borderRadius': '10px', 'padding': '20px', 'textAlign': 'center',
                    'background': 'linear-gradient(135deg, #e0f7fa 0%, #c8e6c9 100%)',  # Gradient background
                    'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)',  # Subtle shadow for depth
                }),
            ], style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'height': '100%'}),
            width=5,  # Equal width for both columns
        ),
    ], style={'alignItems': 'center', 'gap': '20px'}),  # Center-align with gap for spacing

    html.Hr(),
    
], fluid=True, className="p-4")

default_values = {'year' : 2015, 'km_driven' : 70000, 'mileage': 23.4, 'engine': 1200}   
num_cols = ['year', 'km_driven', 'mileage', 'engine']

def get_X(year, km_driven, mileage, engine):
    features = {
        'year': year,
        'km_driven': km_driven,
        'mileage': mileage,
        'engine': engine
    }

    for feature in features:
        # Checking for null
        if not features[feature]:
            features[feature]=default_values[feature]
            # Checking for negative values
        elif feature in num_cols:
            if features[feature]<0:
                features[feature]= default_values[feature]
    X= pd.DataFrame(features, index=[0])

    print(X)
    return X.to_numpy(), features

@callback(
    Output(component_id="prediction-output-vthree", component_property="children"),
    Input(component_id="submit_model", component_property='n_clicks'),
    State(component_id="year-input", component_property="value"),
    State(component_id="km_driven-input", component_property="value"),
    State(component_id="mileage-input", component_property="value"),
    State(component_id="engine-input", component_property="value"),
    prevent_initial_call=True
)

def predict_model(self, input1, input2, input3, input4):
    from utils import load
    import pandas as pd
    import numpy as np
    import mlflow

    df = pd.read_csv('./data/Cars.csv')

    df['engine'] = df['engine'].str.split().str[0].astype('Int64')
    df['mileage'] = df['mileage'].str.split().str[0].astype(float)

    X_impute = df[['year', 'km_driven', 'mileage', 'engine']]
    X_cleaned = X_impute.dropna(subset=['mileage', 'engine'])

#   median_input2 = X_cleaned['km_driven'].fillna(X_cleaned['km_driven'].median(), inplace=True)
#   median_input3 = X_cleaned['mileage'].fillna(X_cleaned['mileage'].mean(), inplace=True)
#   median_input4 = X_cleaned['engine'].fillna(X_cleaned['engine'].median(), inplace=True)

    dict_cat = {
         0 : 'Cheap',
         1 : 'Affordable', 
         2 : 'Expensive', 
         3 : 'Highly Expensive'
    }  

    median_km_driven = X_cleaned['km_driven'].median()
    mean_mileage = X_cleaned['mileage'].mean()
    median_engine = X_cleaned['engine'].median()

    input2 = input2 if input2 is not None else median_km_driven
    input3 = input3 if input3 is not None else mean_mileage
    input4 = input4 if input4 is not None else median_engine

    #model = load('./model/carpriceprediction_vtwo.pickle')
    model_name = "st125171-a3-model"
    model_version = 1
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}") 
    
    X = np.array([[input1, input2, input3, input4]])
    pred = model.predict(X)

    return dict_cat[float(pred)]