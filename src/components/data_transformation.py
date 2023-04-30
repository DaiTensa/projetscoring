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
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
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
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)
            
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