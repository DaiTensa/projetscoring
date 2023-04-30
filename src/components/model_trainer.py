import os
import sys
from dataclasses import dataclass

# Importer les models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Importer les metrics 
from sklearn.metrics import accuracy_score, precision_score, recall_score
# Exception et logging
from src.logger import logging
from src.exception import CustomException
from src.utilis import save_object, evaluate_models
from src.utilis import ModelTrainerConfig



class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Spliting training and test input data")
            X_train, y_train, X_test, y_test= (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )
            
            models = {
                "LogRegression" : LogisticRegression(),
                "Naive Bayes" : GaussianNB(),
                "Nearest Neighbors" : KNeighborsClassifier(3),
#                 "model_4" : model_4(),
#                 "model_5" : model_5(),
#                 "model_6" : model_6()
            }
            
            model_report:dict= evaluate_models(X_train=X_train, y_train=y_train,X_test= X_test ,y_test= y_test, 
                                              models=models)
            # Pour avoir le meilleur score 
            best_model_score= max(sorted(model_report.values()))
            
            # Pour avoir le nom du meilleur model
            best_model_name= list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            # Best model
            best_model = models[best_model_name]
            
            if best_model_score < 0.5:
                raise CustomException("No best model found", sys)

            logging.info(f"Best found model on both training and testing dataset")
            
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            
            
            )
            
            predicted= best_model.predict(X_test)
            
            metric = accuracy_score(y_test, predicted)
            
            return metric
      
        except Exception as e:
            raise CustomException(e, sys)