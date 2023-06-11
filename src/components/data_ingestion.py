import pandas as pd
import numpy as np
import os
import sys
from src.components.data_config import DataIngestionConfig
from src.utilis import reduce_memory_usage
from src.logger import logging
from src.exception import CustomException


class DataIngestion:

    def __init__(self):
        
        """ DataIngestion Class : importer les données via le chemin path
        initiate_data_ingestion : split des données en Train et Test Sets
        enregitrement dans le dossier spécifié dans : src.utilis

        get_files_names : récupérer les noms de fichiers avec l'extension, et extraction des noms sans l'extension. 

        Args:
            path (str): chemin pour accèder aux données
            file_name (str, optional): fichier de données. Defaults to "NOM_FICHIER_DATA.csv".
        """
        self.ingestion_config = DataIngestionConfig()

    def get_files_names(self):

        logging.info("Extraction des noms des fichiers avec l'extension")
        logging.info("Extraction des noms des fichiers sans l'extension")
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
    

    def import_file(self, file_name, reduce_memory_usage = False, number_of_rows=None):
        logging.info(f"Importation du dataset raw : {file_name}")
        try:
            
            path_to_data_base = self.ingestion_config.data_base_path
            print("Importation du dataset...")
            if reduce_memory_usage:
                df = reduce_memory_usage(pd.read_csv(path_to_data_base + file_name, nrows= number_of_rows))
                print("Importation du dataset réussie !")
                logging.info(f"Importation du dataset raw : {file_name} OK")
                
            else:
                df = pd.read_csv(path_to_data_base + file_name, nrows= number_of_rows)
                print("Importation du dataset réussie !")
                logging.info(f"Importation du dataset raw : {file_name} OK")

            
                
            return df
        
        except Exception as e:
            CustomException(e,sys)
        

    

class RapportDataFrame:
    def __init__(self, df, target_column:str, ID_Columns: list[str]):
        self.df = df
        self.target_col = target_column
        self.ID_Columns = ID_Columns
    
    def columns_missing_values(self):

        # Total NaN/Features:
        total = self.df.isnull().sum().sort_values(ascending = False)

        # Pourcentage NaN/Features:
        percent = (self.df.isnull().sum()/self.df.isnull().count()*100).sort_values(ascending = False)

        # Sortie sous forme d'un data frame: 
        df_missing  = pd.concat([total, percent], axis=1, keys=['Total_NAN', 'Percent'])
        return df_missing
    

    def rapport(self, nan_threshold, return_column_to_keep=False, print_rapport = False):

        missing_data= self.columns_missing_values()
        liste_features_vides = list((missing_data[missing_data.Percent == 100 ]).index)
        nombre_col_vides = len(liste_features_vides)
        len_colum_20_percent_nan = len(list(missing_data.loc[missing_data['Percent'] <= nan_threshold].index))
        
        if return_column_to_keep:
            columns_to_keep = list(missing_data.loc[missing_data['Percent'] <= nan_threshold].index)
            return columns_to_keep
            
        if print_rapport:
            (rows, col) = self.df.shape
            
            # calcul du taux de valeurs manquantes : 
            taux_remplissage = (self.df.notnull().sum().sum()/np.product(self.df.shape)) * 100
            columns_nan_sup_nan_threshold = list(missing_data.loc[missing_data['Percent'] > nan_threshold].index)
            
            print(f"\033[1mNombre de ligne :\033[0m {rows} --- \033[1mNombre de colonnes :\033[0m {col}")
            print(20 * "--")
            print('Le Taux de remplissage total est égal à :', round(taux_remplissage,2), "%")  
            print(f"Nombre de colonnes ayant moins de {nan_threshold}% de valeurs manquantes : {len_colum_20_percent_nan}")
            print('Le Nombre de features vides est égal à :', nombre_col_vides, "Features")
            print(20 * "--")
            print('Les Features vides sont :', liste_features_vides)
            print(f'Les Features Ayant plus  {nan_threshold}%  de valeurs manquantes sont:')
            print(columns_nan_sup_nan_threshold)
            print(20 * "--")
            print("*****Nombre de catégorie features catégorielles******\n")
            # Nombre de catégories par varaible qualitatives
            for col in self.df.select_dtypes('object'):
                  print(f'{col :-<50} {self.df[col].unique().size}')
        
        

    def get_df_columns(self):
        
        cols_to_keep = [col for col in self.df.columns if col not in self.ID_Columns + [self.target_col]]
        input_feature_df = self.df[cols_to_keep].copy()
        original_columns = [col for col in self.df.columns]
        categorical_columns = [col for col in self.df.columns if self.df[col].dtype == 'object']
        binary_columns = [col for col in input_feature_df.columns if (len(input_feature_df[col].unique()) == 2) ]
        numerical_columns = list(input_feature_df.select_dtypes(exclude='O').columns)

        return(
            original_columns,
            categorical_columns,
            numerical_columns,
            binary_columns)
    
    def recap_columns_info(self):

        data= []
        level=""
        role= ""
        for col in self.df.columns:
            if col == "TARGET":
                role = 'target'
            elif col == "SK_ID_CURR":
                role = 'id'
            else:
                role = 'input'

            if self.df[col].dtype == "float64":
                level = 'ordinal'
            elif self.df[col].dtype == "int64":
                level = 'ordinal'
            elif self.df[col].dtype == "object":
                level = 'categorical'

            column_dic = {
                'NomColonne' : col,
                'role': role,
                'level' : level,
                'dtype' : self.df[col].dtype,
                'response_rate': 100 * self.df[col].notnull().sum() / self.df.shape[0]
                }

            data.append(column_dic)
        
        recap = pd.DataFrame(data, columns=['NomColonne', 'role', 'level', 'dtype', 'response_rate'])
        # recap.set_index('NomColonne', inplace=True)

        return recap

    
# Exemple pour importer le fichier
# if __name__=="__main__":
    
#     # Base de données
#     obj= DataIngestion()
#     # Liste des fichier et noms
#     liste_name, files_liste_name = obj.get_files_names()
#     # Importer le fichier application_train_.csv
#     application_train = obj.import_file(file_name='application_train.csv', reduce_memory_usage = False, number_of_rows=None)
#     application_train.head()
#     # Générer le rapport
#     rapport_df_train = RapportDataFrame(application_train, target_column="TARGET", ID_Columns=["SK_ID_CURR", "SK_ID_BUREAU"])
#     rapport_df_train.rapport(nan_threshold = 20, return_column_to_keep=False, print_rapport = True)
    


        
