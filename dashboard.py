from dash import Dash, html, dcc, callback, Output, Input, State
import requests
import time as t

url_server = 'http://localhost:5000'

app_dash = Dash(__name__)

app_dash.layout = html.Div([
    html.H1('Dashboard Client'),
    dcc.Input(
        id='id_client',
        type='text',
        value=''
    ),
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='output-data'),
])

@app_dash.callback(
    Output(component_id='output-data', component_property='children'),
    [Input(component_id='submit-button', component_property='n_clicks')],
    [State(component_id='id_client', component_property='value')]
)
def get_data_client(n_clicks, id_client):
    if n_clicks:
        time_0 = t.time()
        # response_data_client = requests.get(f'{url_server}/data/{id_client}')
        response_pred_client = requests.get(f'{url_server}/prediction/{id_client}')
        # data_client = json.loads(response_data_client.text)
        # df = pd.DataFrame(data_client[0])
        time_1 = t.time() - time_0
        return (
            response_pred_client.text, 
            time_1
            # response_data_client.text,
            # print(response_pred_client.text),
            # print(type(data_client[0])),
            # print(data_client[0].keys()),
            # print(data_client[0].items()) 
            # print(type(df))
        )



   
if __name__ == '__main__':
    app_dash.run_server(debug=True)
    
    
    

# data_client = None
# app_dash.layout = dash_table.DataTable(data_client.to_dict('records'), [{"name": i, "id": i} for i in data_client.columns])


# @app_dash.callback(
#      Output(component_id='table', component_property='data'),
#      [Input(component_id='submit-button', component_property='n_clicks')],
#      [State('ID_CLIENT', 'value')]
#      )
# def update_data(n_clicks, id_client):
#     if n_clicks:
#         response_data_client = requests.get(f'http://localhost:5000/data/{id_client}')
#         data_client = json.loads(response_data_client.text)
#         return data_client

 
     
# app_dash.layout = html.Div(children=[
#     html.H1('Dashboard Client'),
#     html.Label('Enter Customer ID'),
#     dcc.Input(id='ID_CLIENT', type='number', value=1),
#     html.Button('Submit', id='submit-button', n_clicks=0),
#     html.H3('Dataframe'),
#     html.Div(id='output-prediction'),
#     table
    # dash_table.DataTable(id='table',columns=[{"name": i, "id": i} for i in data_client.columns], data= data_client.to_dict('records'))
    # dropdown,
# ])


# response_data_client = requests.get('http://localhost:5000/data')
# response_pred_client = requests.get('http://localhost:5000/prediction')
# data_client = json.loads(response_data_client.text)
# pred_client = response_pred_client.json()
# print(response_data_client.text)
# print(response_pred_client.text)
# options = [{'label': k, 'value': k} for k in data.keys()]
# df = pd.DataFrame.from_dict(data, orient='index', columns=list(data.keys()))
# df = pd.DataFrame(data_client)
# dropdown = dcc.Dropdown(
#     options=options,  # options à partir des clés du dictionnaire
#     value=list(data.keys())[0]  # valeur par défaut est la première clé du dictionnaire
# )   
    
# def update_data(n_clicks, ID_ClIENT):
#     if n_clicks > 0:
#         response_data_client = requests.get(f'http://localhost:5000/data/{id_client}')
#         data_client = json.loads(response_data_client.text)
#         df = pd.DataFrame(data_client) 
#         
#         return table
    
    # app.layout = html.Div([
#     html.H1(children='Ma fonction mon dash', style={'textAlign':'center'}),
#     dcc.Dropdown(df.country.unique(), 'Canada', id='dropdown-selection'),
#     dcc.Graph(id='graph-content')
  
# ])

# @callback(
#     Output('graph-content', 'figure'),
#     Input('dropdown-selection', 'value')
# )

# def update_graph(value):
#     dff = df[df.country==value]
#     return px.line(dff, x='year', y='pop')