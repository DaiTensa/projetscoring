import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init_(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self, numerical_columns, categorical_columns):
        """
        Cette Fonction permet de faire la transformation des données 
        """
        try:
           # numerical_columns = ['','']
           # categorical_columns = ['','']

            num_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy= "median")),
                ("scaled", StandardScaler())
                ])
            
            cat_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("sclaer", StandardScaler())
                ])

            logging.info("Numerical columns standar scaling  completed")
        
            logging.info("Categorical columns encoding completed")

            preprocessor=ColumnTransformer(
                
                [
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
                
                ])
            
            return preprocessor

        
        except Exception as e:
            raise CustomException(e, sys)