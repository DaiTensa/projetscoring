import sys
import pandas as pd
from src.exception import CustomException
from src.utilis import load_object

from src.components.data_config import AppConfig

class DataClient:
    def __init__(self):
        self.data_preprocessor_model_path = AppConfig()
        self.df_clients= pd.read_csv(self.data_preprocessor_model_path.clients_data__path)
        self.preprocessor= load_object(file_path=self.data_preprocessor_model_path.preprocessor_ob_file__path)
        self.model= load_object(file_path=self.data_preprocessor_model_path.trained_model_file__path)
        
    def get_data_as_df(self, ID_client):
        
        try:
            ID_client = int(ID_client)
            df_client= (self.df_clients.loc[self.df_clients["SK_ID_CURR"] == ID_client]).copy()
            df_client = df_client.drop(["SK_ID_CURR"], axis=1)
            return df_client

        except Exception as e:
            raise CustomException(e, sys)

    def predict_function(self, df):
        try:

            data_scaled= self.preprocessor.transform(df)
            data_scaled = pd.DataFrame(data_scaled, columns= list(df.columns))
            pred= self.model.predict(data_scaled)
            return pred

        except Exception as e:
            raise CustomException(e, sys)