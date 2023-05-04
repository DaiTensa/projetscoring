# Imports

import os
import sys
from dataclasses import dataclass



###################################### Data path config ######################################################

@dataclass
class DataIngestionConfig:
    # data_base_path : 
    data_base_path : str=os.path.join('C:/Users/Lenovo/Documents/DSPython/data_projet_7/')

    
    # Création des chemin pour la création des train test set split
    file_path : str=os.path.join('C:/Users/Lenovo/Documents/DSPython/data_projet_7/', "application_train.csv")
    

    train_data_path: str=os.path.join('artifacts', "train.csv")
    test_data_path: str=os.path.join('artifacts', "test.csv")
    raw_data_path: str=os.path.join('artifacts', "data.csv")

    # Si le train et test existe déjà alors : changer le chemin et le nom du dossier
    train_path: str=os.path.join('C:/Users/Lenovo/Documents/DSPython/data_projet_7/', "application_train.csv")
    test_path: str=os.path.join('C:/Users/Lenovo/Documents/DSPython/data_projet_7/', "application_test.csv")
        
#########################################Data Transformation Config #############################################

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifacts', "preprocessor.pkl")


######################################### Models trainer config #########################################################
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    