import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.utilis import DataTransformationConfig
from src.utilis import save_object
from src.logger import logging
import os


class DataPreprocessing():
    def __init__(self):
        return
        
    def one_hot_encoder(slef, df, nan_as_category=True):
        original_columns = list(df.columns)
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        df = pd.get_dummies(df, columns= categorical_columns, dummy_na=nan_as_category)
        new_columns = [col for col in df.columns if col not in original_columns]
        return df, new_columns
    
    
class DataTransformation:
    def __init__(self, numerical_columns, categorical_columns, target="" , train_path ="", test_path =""):
        self.data_transformation_config=DataTransformationConfig()
        self.col_num = numerical_columns
        self.col_cat = categorical_columns
        self.target_column = target
        self.train_path = train_path
        self.test_path = test_path
        
 
    def get_data_transformer_object(self):
        """
        Cette Fonction permet de faire la transformation des données 
        """
        try:
            
            num_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(missing_values=np.nan, strategy='mean')),
#                 ("imputer", IterativeImputer(max_iter=10, random_state=0)),
#                 ("imputer", KNNImputer(n_neighbors=2, weights="uniform")),
                ("scaled", StandardScaler())
                ])
            
            cat_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value='missing')),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                ("sclaer", StandardScaler(with_mean=False))
                ])

            logging.info("Numerical columns standar scaling  completed")
        
            logging.info("Categorical columns encoding completed")

            preprocessor=ColumnTransformer(
                
                [
                ("num_pipeline", num_pipeline, self.col_num),
                ("cat_pipeline", cat_pipeline, self.col_cat)
                
                ])
            
            return preprocessor

        
        except Exception as e:
            raise CustomException(e, sys)
    
    def  initiate_data_transformation(self):
        
        try:
            train_df = pd.read_csv(self.train_path)
            test_df = pd.read_csv(self.test_path)
            
            logging.info("Read Train and Test data completed")
            
            logging.info("Obtaining preprocessin object")
            
            preprocessing_obj = self.get_data_transformer_object()
            
            input_feature_train_df = train_df.drop(columns= [self.target_column], axis= 1)
            target_feature_train_df= train_df[self.target_column]
            
            input_feature_test_df= test_df.drop(columns= [self.target_column], axis= 1)
            target_feature_test_df= test_df[self.target_column]
            
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
            
            preprocessing_obj.fit(input_feature_train_df)
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info(f"Saved preprocessing object.")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessing_obj
            
            
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
              
            
            )
             
        except Exception as e:
                raise CustomException(e, sys)

# try:
#             # Ouvrir le fichier de données
#             df= pd.read_csv(f"{self.ingestion_config.file_path}")
#             logging.info("Lecture du fichier de données")

#             # Création du dossier artifacts pour sauvegrder les données
#             os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

#             # Sauvegarde du fichier de données
#             df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
#             y = df[target]
#             X = df.drop(target, axis= 1)
            
#             X_train, y_train, X_test, y_test= train_test_split(X, y, test_size=0.2, random_state=42, startify= y)
#             logging.info("Train test split initiated")
            
#             train_set = 
#             test_set = 
            
#             # Sauvergarde des deux fichiers Train et Test
#             train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
#             test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
#             logging.info("Ingestion of the data is completed")

#             return(
#                 self.ingestion_config.train_data_path,
#                 self.ingestion_config.test_data_path
#                 )

#         except Exception as e:
#             raise CustomException(e, sys)
            

                