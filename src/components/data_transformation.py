import sys
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from src.exception import CustomException
from src.components.data_config import DataTransformationConfig
from src.components.data_ingestion import RapportDataFrame
from src.utilis import save_object
from src.logger import logging
import os

  
    
class DataTransformation:
    def __init__(self, df:Optional[pd.DataFrame] = None):
        
        self.data_transformation_config= DataTransformationConfig()
        self.df = df

    def initiate_train_test_split(self,target, test_size, stratification=False):

        logging.info(f"Debut de initiation du train_test_split")

        try:

            os.makedirs(os.path.dirname(self.data_transformation_config.data_path), exist_ok=True)
            logging.info("Dossier artifacts : OK")

            self.df.copy().to_csv(self.data_transformation_config.data_path, index=False, header=True)
            logging.info("Dossier artifacts : OK")
            
            y = self.df[target]
            X = self.df.drop(columns=[target])
            logging.info("X et y : OK")
            


            if stratification:
                logging.info("Initiation du train_test_split avec stratification")
                X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=test_size, random_state=0, shuffle=True, stratify=y)
                logging.info("Train et Test split avec stratification : OK")
            else:
                logging.info(f"Initiation du train_test_split sans stratification")
                X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=test_size, random_state=0)
                logging.info("Train et Test split sans stratification : OK")
            
          
            
                
            # Sauvegarde des data sets: 
            X_train.to_csv(self.data_transformation_config.X_train_path, index=False, header=True)
            print(f"X_train : done 1/4 ---- Nombre de lignes :{X_train.shape[0]}, Nombre de colonnes:{X_train.shape[1]}")
            y_train.to_csv(self.data_transformation_config.y_train_path, index=False, header=True)
            print(f"y_train : done 2/4 ---- Nombre de lignes :{y_train.shape[0]}")
            X_test.to_csv(self.data_transformation_config.X_test_path, index=False, header=True)
            print(f"X_test : done 3/4 ---- Nombre de lignes :{X_test.shape[0]}, Nombre de colonnes:{X_test.shape[1]}")
            y_test.to_csv(self.data_transformation_config.y_test_path, index=False, header=True)
            print(f"y_test : done 4/4 ---- Nombre de lignes :{y_test.shape[0]}")
            print(20 * "#")
            print("Les données sont enregistrées dans le dossier artifacts")

            logging.info("X_train X_test y_train y_test : OK")
            logging.info("Save X_train X_test y_train y_test dans artifacts: OK")
               
       

            

            return(
                X_train,
                X_test,
                y_train,
                y_test
            )

        except Exception as e:
            raise CustomException(e, sys)
            
    def get_data_frames(self):
        logging.info("Importation des donnes de train et de test")
        X_train = pd.read_csv(self.data_transformation_config.X_train_path)
        X_test = pd.read_csv(self.data_transformation_config.X_test_path)
        y_train = pd.read_csv(self.data_transformation_config.y_train_path)
        y_test = pd.read_csv(self.data_transformation_config.y_test_path)
        logging.info("Importation des donnes de train et de test: OK")
        
        return(
        X_train,
        X_test,
        y_train,
        y_test)
    
    
    
 
    def initiate_data_transformation(self,X_train, X_test, y_train,y_test, undersampling= False, return_train_test_array=False):
        
        """
        Cette Fonction permet de faire la transformation des données 
        """
        try:
            if undersampling:
                logging.info("Initiate Undersampling")
                print('TARGET distribution avant Undersampling')
                print(y_train.value_counts())
                under_sampler_object = RandomUnderSampler(random_state=0) # revoir le partitionnement des target ????? 
                X_train_res, y_train_res = under_sampler_object.fit_resample(X_train, y_train)
                print()
                print('TARGET distribution après Undersampling')
                print(y_train_res.value_counts())
                logging.info("Undersampling :  OK")
                


            
            X_train_rapport = RapportDataFrame(X_train_res, target_column="TARGET", ID_Columns=["SK_ID_CURR", "SK_ID_BUREAU"])
            _, categ, numerical_columns_train, _= X_train_rapport.get_df_columns()
            logging.info("X_train : Extraction features numeriques :  OK")
            logging.info("X_train : Extraction features categorielles :  OK")
            
            X_test_rapport = RapportDataFrame(X_test, target_column="TARGET", ID_Columns=["SK_ID_CURR", "SK_ID_BUREAU"])
            _, _, numerical_columns_test, _  = X_test_rapport.get_df_columns()
            

            
            logging.info("Obtaining preprocessor object")
            
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='mean')),
#                 ("imputer", IterativeImputer(max_iter=10, random_state=0)),
#                 ("imputer", KNNImputer(n_neighbors=2, weights="uniform")),
                ("scaled", StandardScaler())
            ])

            logging.info("Pipeline numerique : OK")
            
            cat_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value='missing')),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                ("sclaer", StandardScaler(with_mean=False))
                ])

            logging.info("Pipeline categorielle : OK")
        
            logging.info("Categorical columns encoding completed")

            preprocessor=ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns_train),
                ("cat_pipeline", cat_pipeline, categ)
                
            ])

            logging.info("Preprocessor object : OK")
    
            print()
            print("Les étapes de transformation :")
            print(preprocessor)
    
            X_train_res = X_train_res.replace((np.inf, -np.inf), np.nan).reset_index(drop=True)
            X_test = X_test.replace((np.inf, -np.inf), np.nan).reset_index(drop=True)
            
            preprocessor.fit(X_train_res)
            logging.info("Fit Preprocessor object sur les donnes X_train : OK")

            X_train_arr = preprocessor.transform(X_train_res)
            

            X_train_input_features = np.concatenate((preprocessor.named_transformers_["num_pipeline"].get_feature_names_out(numerical_columns_train),
                                          preprocessor.named_transformers_["cat_pipeline"].named_steps["one_hot_encoder"].fit(X_train_res[categ]).get_feature_names(categ)))

            X_test_arr = preprocessor.transform(X_test)
            logging.info("Fit Transform Preprocessor object sur les donnes X_test : OK")

            X_test_input_features = np.concatenate((preprocessor.named_transformers_["num_pipeline"].get_feature_names_out(numerical_columns_train),
                                                      preprocessor.named_transformers_["cat_pipeline"].named_steps["one_hot_encoder"].fit(X_test[categ]).get_feature_names(categ)))
            

            X_train = pd.DataFrame(X_train_arr, columns=X_train_input_features)
            X_test = pd.DataFrame(X_test_arr, columns=X_test_input_features)
            
            save_object(
                
                file_path = self.data_transformation_config.preprocessor_ob_file_path, 
                obj = preprocessor
            
            
            )

            logging.info("Save du preprocessor object format pkl : OK")

            if return_train_test_array:
                train_arr = np.c_[np.array(X_train), np.array(y_train_res)]
                test_arr = np.c_[np.array(X_test), np.array(y_test)]
                logging.info("Fin du preprocessing : OK")
                return(train_arr, test_arr)
            
            else:
                logging.info("Fin du preprocessing : OK")
                return (        
                    X_train,
                    X_test,
                    y_train_res,
                    y_test
                )

        
        except Exception as e:
            raise CustomException(e, sys)
            
            
# Ancienne version de initiate_data_transformation
#     def  initiate_data_transformation(self):
        
#         try:

            
#             logging.info("Read Train and Test data completed")
            
#             logging.info("Obtaining preprocessin object")
            
#             preprocessing_obj = self.get_data_transformer_object()
            
#             input_feature_train_df = train_df.drop(columns= [self.target_column], axis= 1)
#             target_feature_train_df= train_df[self.target_column]
            
#             input_feature_test_df= test_df.drop(columns= [self.target_column], axis= 1)
#             target_feature_test_df= test_df[self.target_column]
            
#             logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
            
#             preprocessing_obj.fit(input_feature_train_df)
#             input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
#             input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            
            
#             train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            
#             test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
#             logging.info(f"Saved preprocessing object.")
            
#             save_object(
#                 file_path = self.data_transformation_config.preprocessor_ob_file_path,
#                 obj = preprocessing_obj
            
            
#             )
            
#             return (
#                 train_arr,
#                 test_arr,
#                 self.data_transformation_config.preprocessor_ob_file_path
              
            
#             )
             
#         except Exception as e:
#                 raise CustomException(e, sys)


#             logging.info("Concatenate X_train y_train et X_test y_test")
#             train_df = pd.concat([X_train, y_train], axis = 1)
#             test = pd.concat([X_test, y_test], axis = 1)
#             logging.info("Concatenate : Done")


            

                