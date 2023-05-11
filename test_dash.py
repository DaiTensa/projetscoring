from dash import Dash, dcc, html, Input, Output

app = Dash(__name__)

app.layout = html.Div([
    html.H6("Change the value in the text box to see callbacks in action!"),
    html.Div([
        "Input: ",
        dcc.Input(id='ID_CLIENT', value=1, type='text')
    ]),
    html.Br(),
    html.Div(id='ID_CLIENT'),

])


@app.callback(
    Output(component_id='ID_CLIENT', component_property='children'),
    Input(component_id='Input', component_property='value')
)
def update_output_div(input_value):
    return f'Output: {input_value}'


if __name__ == '__main__':
    app.run_server(debug=True)