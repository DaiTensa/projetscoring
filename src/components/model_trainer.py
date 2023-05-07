import os
import sys
from dataclasses import dataclass

# Importer les models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Importer les metrics 
from sklearn.metrics import accuracy_score, precision_score, recall_score
# Exception et logging
from src.logger import logging
from src.exception import CustomException
from src.utilis import save_object, evaluate_model_
from src.components.data_config import ModelTrainerConfig


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()
        
    def initiate_model_trainer(self, X_train, y_train, X_test, y_test, model, params, save_best_model=False):
        try:
            logging.info("Evaluation modele : Debut")
            logging.info("Evaluation du modele et fine tuning des hyperparametres")
            
            test_model_accuracy, train_model_accuracy, best_model = evaluate_model_(X_train, y_train, X_test ,y_test, 
                                              model, params)

            
            if test_model_accuracy < 0.5:
                raise CustomException("No best model found", sys)

            logging.info(f"Evaluation modele : Fin")

            if save_best_model:
                save_object(file_path= self.model_trainer_config.trained_model_file_path,
                            obj= best_model
                            )
                logging.info(f"Save du  best modele format pkl : OK")
            
            print(f"{type(best_model).__name__} -- test_accuracy: {test_model_accuracy} -- train_accuracy: {train_model_accuracy}")
             
      
        except Exception as e:
            raise CustomException(e, sys)



# Ancienne Version : pour comparer plusierus modèles
# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config= ModelTrainerConfig()
        
#     def initiate_model_trainer(self, X_tr, y_tr, X_te, y_te, models, param):
#         try:
#             logging.info("Evaluation modele : Debut")
#             logging.info("Evaluation du modele et fine tuning des hyperparametres")
            
#             model_report:dict= evaluate_models(X_train=X_tr, y_train=y_tr, X_test= X_te ,y_test= y_te, 
#                                               models=models, params=param)
#             # Pour avoir le meilleur score 
#             best_model_score= max(sorted(model_report.values()))
            
#             # Pour avoir le nom du meilleur model
#             best_model_name= list(model_report.keys())[
#                 list(model_report.values()).index(best_model_score)
#             ]
            
#             # Best model
#             best_model = models[best_model_name]
            
#             if best_model_score < 0.5:
#                 raise CustomException("No best model found", sys)

#             logging.info(f"Evaluation modele : Fin")
            
#             save_object(
#                 file_path= self.model_trainer_config.trained_model_file_path,
#                 obj= best_model
            
            
#             )

#             logging.info(f"Save du  best modele format pkl : OK")
            
#             predicted= best_model.predict(X_te)
            
#             metric = accuracy_score(y_te, predicted)
            
#             return model_report
      
#         except Exception as e:
#             raise CustomException(e, sys)