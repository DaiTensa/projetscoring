from flask import Flask, request, render_template
import numpy as numpy
import pandas as pd 

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData

application= Flask(__name__)

app= application

## Route for a home page


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data_client=CustomData(
            
            
            SK_ID_CURR = request.form.get("client_id")
            
            
            )

        df = data_client.get_data_as_data_frame()
        pred = data_client.predict(df)
        
        return render_template('home.html', results=pred[0])

if __name__ == "__main__":
    app.run(debug=True)