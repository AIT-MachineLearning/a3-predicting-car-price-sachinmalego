import dash
from dash import Dash, html, callback, Output, Input, State, dcc
import dash_bootstrap_components as dbc
dash.register_page(__name__, path='/')


layout = html.Div([
    html.Div([
        html.H1('Welcome to Car Price Prediction 2024 - Ver.3', style={'padding-top':'20px', 'color':'green', 'font-style':'bold', 'text-decoration':'underline', 'text-align':'center', 'padding-bottom':'10px'}),
        html.Div([
            html.P('Instructions for using this Web Application.', style={'font-weight':'bold'}),
            html.P("Step 1: Navigate to the Prediction Section from the Navigation bar at the top. Click on the tab named PredictV1: Old Model or PredictV2: New Model or PredictV3: New Version Model with Classification to open the prediction form."),
            html.P("Step 2: The form will ask for the following details about the car. Enter the information in the respective fields:"),
            html.Ul([
                html.Li("Year: Enter the manufacturing year of the car."),
                html.Li("Kilometers Driven: Enter the total distance the car has been driven (in kilometers)."),
                html.Li("Mileage: Enter the mileage of the car in kmpl (kilometers per liter)."),
                html.Li("Engine: Enter the engine capacity in cc (cubic centimeters)."),
            ]),
            html.P("Step 3: Once all fields are filled, click the Submit button at the bottom of the form."),
            html.P("View the Prediction Classification:"),
            html.Ul([
                html.Li("After submission, the application will process the data."),
                html.Li("The predicted car price will be displayed on the screen."),
                html.Li("Review the prediction to get an estimate of your car's price based on the provided details."),
            ])
        ]),
    ], className='container')  # Use the container class here
])

