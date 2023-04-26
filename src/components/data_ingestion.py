import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass




@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', "train.csv")
    test_data_path: str=os.path.join('artifacts', "test.csv")
    raw_data_path: str=os.path.join('artifacts', "data.csv")



class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    # Fonction pour créer les set : tain, test, raw est les enregistré dans un directory : artifacts
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Ne pas oublier de changer le nom du fichier csv : dans le chemin suivant ---> NOM_FICHIER_DATA
            df= pd.read_csv("C:/Users/Lenovo/Documents/DSPython/data_projet_7/NOM_FICHIER_DATA.csv")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated")
            train_set,test_set= train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomException(e, sys) 
        
@dataclass   
class DataPath:
    train_path: str=os.path.join('C:/Users/Lenovo/Documents/DSPython/data_projet_7/', "application_train.csv")
    test_path: str=os.path.join('C:/Users/Lenovo/Documents/DSPython/data_projet_7/', "application_test.csv")
    datapath: str=os.path.join('C:/Users/Lenovo/Documents/DSPython/data_projet_7/')
    train_data_path_: str=os.path.join('artifacts', "train.csv")
    test_data_path_: str=os.path.join('artifacts', "test.csv")


class ImportData:
    def __init__(self):
        self.data_path=DataPath()
    def train_test_set(self):
        logging.info("Importation du train et du test set")
        try:
            train_data = pd.read_csv(self.data_path.train_path)
            os.makedirs(os.path.dirname(self.data_path.train_data_path_), exist_ok=True)
            train_data.to_csv(self.data_path.train_data_path_, index=False, header=True)
            logging.info("Train set split initiated")

            test_data = pd.read_csv(self.data_path.test_path)
            os.makedirs(os.path.dirname(self.data_path.test_data_path_), exist_ok=True)
            test_data.to_csv(self.data_path.test_data_path_, index=False, header=True)
            logging.info("Test set split initiated")
            return(train_data, test_data)
        

        except Exception as e:
            raise CustomException(e,sys)

class DirectoryDataPath:

    def __init__(self):
        self.data_directory_path = DataPath()
    
    def get_files_names(self):
        logging.info("Extraction des noms des fichiers")
        try:
            files_liste_name = os.listdir(self.data_directory_path.datapath)
            sub1 = ""
            sub2 = ".csv"
            idx1 = 0
            idx2 = 0
            liste_name = []
            for name in files_liste_name: 
                name = str(name)
                idx1 = name.index(sub1)
                idx2 = name.index(sub2)
                res = ''
                for idx in range(idx1 + len(sub1), idx2):
                    res = res + name[idx]
                name_= str(res)
                liste_name.append(name_)
            return(liste_name, files_liste_name)

        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    obj= ImportData()
    obj.train_test_set()
        
