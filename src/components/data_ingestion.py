import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.utilis import DataIngestionConfig

from src.logger import logging
from src.exception import CustomException


class DataIngestion:

    def __init__(self, method = None):
        
        """ DataIngestion Class : importer les données via le chemin path
        initiate_data_ingestion : split des données en Train et Test Sets
        enregitrement dans le dossier spécifié dans : src.utilis

        get_files_names : récupérer les noms de fichiers avec l'extension, et extraction des noms sans l'extension. 

        Args:
            path (str): chemin pour accèder aux données
            file_name (str, optional): fichier de données. Defaults to "NOM_FICHIER_DATA.csv".
        """
        self.ingestion_config = DataIngestionConfig()
        self.method = method

    

    def initiate_data_ingestion(self):

        logging.info("Entered the data ingestion method or component")

        if self.method == "split":

            try:
                # Ouvrir le fichier de données
                df= pd.read_csv(f"{self.ingestion_config.file_path}")
                logging.info("Lecture du fichier de données")

                # Création du dossier artifacts pour sauvegrder les données
                os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

                # Sauvegarde du fichier de données
                df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

                
                train_set,test_set= train_test_split(df, test_size=0.2, random_state=42)
                logging.info("Train test split initiated")

                # Sauvergarde des deux fichiers Train et Test
                train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
                test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
                logging.info("Ingestion of the data is completed")

                return(
                    self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
                    )

            except Exception as e:
                raise CustomException(e, sys)
            

                
        else:
            
            try:
                
                train_set = pd.read_csv(f"{self.ingestion_config.train_path}")
                test_set = pd.read_csv(f"{self.ingestion_config.test_path}")
                logging.info("Train Test Set  initiated")
                
                train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
                test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
                logging.info("Ingestion of the data is completed")

                return(self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)
                
            except Exception as e:
                raise CustomException(e, sys)
            
            
                

        


    def get_files_names(self):

        logging.info("Extraction des noms des fichiers")
        try:
            files_liste_name = os.listdir(self.ingestion_config.data_base_path)
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
    

# Exemple pour importer le fichier et faire un train test split 
# if __name__=="__main__":
#     obj= DataIngestion(method="split")
#     obj.initiate_data_ingestion()


        
