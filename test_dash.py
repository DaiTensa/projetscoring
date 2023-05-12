from dash import Dash,html, dcc, callback, Input, Output

app = Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='This is our Analytics page'),
	html.Div([
        "Select a city: ",
        dcc.RadioItems(['New York City', 'Montreal','San Francisco'],
        id='analytics-input')
    ]),
	html.Br(),
    html.Div(id='analytics-output'),
])
@callback(
    Output(component_id='analytics-output', component_property='children'),
    Input(component_id='analytics-input', component_property='value')
)
def update_city_selected(int_value):
    return f'You selected: {int_value}'
# @app.callback(
#     Output(component_id='out-put-data', component_property='children'),
#     Input(component_id='ID_CLIENT', component_property='value')
# )
# def update_output_div(value):
#     return f'Output: {value}'


if __name__ == '__main__':
    app.run_server(debug=True)