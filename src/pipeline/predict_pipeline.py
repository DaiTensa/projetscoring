import sys
import pandas as pd
from src.exception import CustomException
from src.utilis import load_object
from src.components.data_config import AppConfig


class CustomData:
    def __init__(self,SK_ID_CURR):
        self.SK_ID_CURR = int(SK_ID_CURR)
        self.base_data_path = AppConfig()
        self.data_config= AppConfig()
        self.model_path= AppConfig()

    def get_data_as_data_frame(self):

        try:

            df_clients__path = self.base_data_path.clients_data__path
            df=pd.read_csv(df_clients__path)
            df_client = df.copy()
            df_client= df_client.loc[df_client["SK_ID_CURR"] == self.SK_ID_CURR, :]
            return df_client


        except Exception as e:
            raise CustomException(e, sys)


    def predict(self, df):
        try:

            # df_clients_path = self.base_data_path.clients_data_path
            # df=pd.read_csv(df_clients_path)

            model_path = self.model_path.trained_model_file__path
            preprocessor_path = self.data_config.preprocessor_ob_file__path

            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(df)
            preds=model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


# class PredictPipeline:
#     def __init__(self):
#         self.base_data_path = DataIngestionConfig()
#         self.data_config= DataTransformationConfig()
#         self.model_path= ModelTrainerConfig()

#     def predict(self, features):
#         try:

#             df_clients_path = self.base_data_path.clients_data_path
#             df=pd.read_csv(df_clients_path)

#             model_path = self.model_path.trained_model_file_path
#             preprocessor_path = self.data_config.preprocessor_ob_file_path

#             model=load_object(file_path=model_path)
#             preprocessor=load_object(file_path=preprocessor_path)
#             data_scaled=preprocessor.transform(features)
#             preds=model.predict(data_scaled)
#             return preds

#         except Exception as e:
#             raise CustomException(e, sys)



