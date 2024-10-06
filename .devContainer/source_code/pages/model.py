import numpy as np
import pandas as pd

import dash
from dash import Dash, html, callback, Output, Input, State, dcc
import dash_bootstrap_components as dbc
dash.register_page(__name__, path='/predict')


# Form Elements
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
    html.H3("Enter Car Details for Price Prediction Ver.1", style={'padding': '20px 0', 'color': '#004d00', 'font-weight': 'bold', 'textAlign': 'center'}),
    
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
                html.H3("Predicted Price", style={'color': '#004d00', 'font-weight': 'bold', 'textAlign': 'center'}),
                html.Div(id="prediction-output", className="mt-3", style={
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


@callback(
    Output(component_id="prediction-output", component_property="children"),
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

    df = pd.read_csv('./data/Cars.csv')

    df['engine'] = df['engine'].str.split().str[0].astype('Int64')
    df['mileage'] = df['mileage'].str.split().str[0].astype(float)

    X_impute = df[['year', 'km_driven', 'mileage', 'engine']]
    X_cleaned = X_impute.dropna(subset=['mileage', 'engine'])

    median_input2 = X_cleaned['km_driven'].fillna(X_cleaned['km_driven'].median(), inplace=True)
    median_input3 = X_cleaned['mileage'].fillna(X_cleaned['mileage'].mean(), inplace=True)
    median_input4 = X_cleaned['engine'].fillna(X_cleaned['engine'].median(), inplace=True)

    input2 = input2 if input2 is not None else median_input2
    input3 = input3 if input3 is not None else median_input3
    input4 = input4 if input4 is not None else median_input4

    model = load('./model/carpriceprediction.pickle')
    X = np.array([[input1, input2, input3, input4]])
    pred = np.exp(model.predict(X))
    return round(float(pred), 2)
