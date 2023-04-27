import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', "train.csv")
    test_data_path: str=os.path.join('artifacts', "test.csv")
    raw_data_path: str=os.path.join('artifacts', "data.csv")
    train_path: str=os.path.join('C:/Users/Lenovo/Documents/DSPython/data_projet_7/', "application_train.csv")
    test_path: str=os.path.join('C:/Users/Lenovo/Documents/DSPython/data_projet_7/', "application_test.csv")


############################ EXPLORATION DATA FRAMES#####################################

class RapportDataFrame:
    def __init__(self, df):
        self.df = df
    
    def chek_missing(self):

        # Total NaN/Features:
        total = self.df.isnull().sum().sort_values(ascending = False)

        # Pourcentage NaN/Features:
        percent = (self.df.isnull().sum()/self.df.isnull().count()*100).sort_values(ascending = False)

        # Sortie sous forme d'un data frame: 
        df_missing  = pd.concat([total, percent], axis=1, keys=['Total_NAN', 'Percent'])
        return df_missing
    
    def rapport(self):

        missing_data= self.chek_missing()
        liste_features_vides = list((missing_data[missing_data.Percent == 100 ]).index)
        nombre_col_vides = len(liste_features_vides)

        # calcul du taux de valeurs manquantes : 
        taux_remplissage = (self.df.notnull().sum().sum()/np.product(self.df.shape)) * 100

        print('Le Taux de remplissage total est égal à :', taux_remplissage, "%")   
        print('Le Nombre de features vides est égal à :', nombre_col_vides, "Features")
        print('Les Features vides sont :', liste_features_vides)
        # print('Le Nombre de Valeurs en double est égal à :', valeurDoubl, "valeurs")
        print()
        print("*****Nombre de catégorie features catégorielles******\n")
        # Nombre de catégories par varaible qualitatives
        for col in self.df.select_dtypes('object'):
              print(f'{col :-<50} {self.df[col].unique().size}')


    def get_df_columns(self):

        original_columns = [col for col in self.df.columns]
        categorical_columns = [col for col in self.df.columns if self.df[col].dtype == 'object']
        binary_columns = [col for col in self.df.columns if (self.df[col].dtype == 'object' and len(self.df[col].unique()) == 2) ]
        numerical_columns = list(self.df.select_dtypes(exclude='O').columns)

        return(
            original_columns,
            categorical_columns,
            numerical_columns,
            binary_columns)
################################################# PLOTS #################################################

def bar_plot(df, feature="", bar_title=""):
    
    temp = df[feature].value_counts()
    data = pd.DataFrame(
        {'labels': temp.index,
        'values': temp.values
        })
    plt.figure(figsize = (6,6))
    plt.title(bar_title)
    sns.set_color_codes("pastel")
    sns.barplot(x = 'labels', y="values", data=data)
    locs, labels = plt.xticks()
    plt.show()




############################# Memory Usage Reduction ####################################################
def reduce_memory_usage(df):
    """
    Cette Fonction permet d'importer un data frame avec une gestion de la mémoire 
    Copié dans : https://www.kaggle.com/code/rinnqd/reduce-memory-usage/notebook

    Args:
        df (dataframe): dataframe à importer

    Returns:
        df: dataframe après optimisation de la mémoire
    """
  
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df 