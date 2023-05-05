# Imports

import os
import sys
from dataclasses import dataclass



###################################### Data path config ######################################################

@dataclass
class DataIngestionConfig:
    # data_base_path : 
    data_base_path : str=os.path.join('C:/Users/Lenovo/Documents/DSPython/data_projet_7/')
    train_data_path: str=os.path.join('artifacts', "train.csv")
    test_data_path: str=os.path.join('artifacts', "test.csv")
    

    # Si le train et test existe déjà alors : changer le chemin et le nom du dossier
    train_path: str=os.path.join('C:/Users/Lenovo/Documents/DSPython/data_projet_7/', "application_train.csv")
    test_path: str=os.path.join('C:/Users/Lenovo/Documents/DSPython/data_projet_7/', "application_test.csv")
        
#########################################Data Transformation Config #############################################

@dataclass
class DataTransformationConfig():
    data_path:str=os.path.join('artifacts', "data.csv")
    X_train_path:str=os.path.join('artifacts', "X_train.csv")
    y_train_path:str=os.path.join('artifacts', "y_train.csv")
    X_test_path:str=os.path.join('artifacts', "X_test.csv")
    y_test_path:str=os.path.join('artifacts', "y_test.csv")
    preprocessor_ob_file_path=os.path.join('artifacts', "preprocessor.pkl")


######################################### Models trainer config #########################################################
# file_path:str=os.path.join('C:/Users/Lenovo/Documents/DSPython/data_projet_7/', "df_final.csv")
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    