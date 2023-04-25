import pandas as pd
import numpy as np

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
        numerical_columns = list(self.df.select_dtypes(exclude='O').columns)

        return(
            original_columns,
            categorical_columns,
            numerical_columns


        )


