import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os
import sys
from dataclasses import dataclass

#######################################Config Data Path ####################################

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


            
################################################# PLOTS #################################################


# Fonction pour générer les couleurs pour les cluster
def generate_colors(num_color, palette='bright'):
    
    """
    Selon le nombre de Cluster : num_cluster, on vas générer des couleurs au format hex.
    
    deep, muted, bright, pastel, dark, colorblind
    """
    
    colors = color = sns.color_palette(palette= palette, n_colors= num_color, desat=None, as_cmap=False).as_hex()
    return colors


def count_plot_for_object(df, temp_col, target = "TARGET", label_rotation=False, palette='bright', nan_as_categ=None):
    
    # Seaborn config : 
    sns.set_style("whitegrid")

    if(nan_as_categ):
        df_filled_na = df[[temp_col]].fillna('null')
        df_filled_na['TARGET'] = df['TARGET']
        
        df_0_filled = df_filled_na.loc[df_filled_na['TARGET'] == 0,:]
        df_1_filled = df_filled_na.loc[df_filled_na['TARGET'] == 1,:]
        
        num_categories_df = len(df_filled_na[temp_col].unique())
        num_categories_df_0 = len(df_0_filled[temp_col].unique())
        num_categories_df_1 = len(df_1_filled[temp_col].unique())
        
        colors_df = generate_colors(num_categories_df, palette = palette)
        colors_df_0 = generate_colors(num_categories_df_0, palette = palette)
        colors_df_1 = generate_colors(num_categories_df_1, palette = palette)
        
        fig, axs = plt.subplots(ncols=3, figsize=(10,5))
        s = sns.countplot(x= temp_col, data=df_filled_na, ax=axs[0], palette=colors_df, saturation=0.75)
        if(label_rotation):
            s.set_xticklabels(s.get_xticklabels(),rotation=90)
        axs[0].set_title("Total")
        
        s = sns.countplot(x= temp_col, data=df_1_filled, ax=axs[1], palette=colors_df_1, saturation=0.75)
        if(label_rotation):
            s.set_xticklabels(s.get_xticklabels(),rotation=90)
        axs[1].set_title("Count : TARGET = 1 ")
        
        s = sns.countplot(x=temp_col, data=df_0_filled, ax=axs[2], palette=colors_df_0, saturation=0.75)
        if(label_rotation):
            s.set_xticklabels(s.get_xticklabels(),rotation=90)
        axs[2].set_title("Count : TARGET = 0 ")
        
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        plt.show()
        
    # Affichage sans fill valeurs manquantes   
    else:
        
        df_0 = df.loc[df[target] == 0,:]
        df_1 = df.loc[df[target] == 1,:]

        num_categories_df = len(df[temp_col].unique())
        num_categories_df_0 = len(df_0[temp_col].unique())
        num_categories_df_1 = len(df_1[temp_col].unique())

        colors_df = generate_colors(num_categories_df, palette = palette)
        colors_df_0 = generate_colors(num_categories_df_0, palette = palette)
        colors_df_1 = generate_colors(num_categories_df_1, palette = palette)

        fig, axs = plt.subplots(ncols=3, figsize=(10,5))
        s = sns.countplot(x= temp_col, data=df, ax=axs[0], palette=colors_df, saturation=0.75)
        if(label_rotation):
            s.set_xticklabels(s.get_xticklabels(),rotation=90)
        axs[0].set_title("Total")

        s = sns.countplot(x= temp_col, data=df_1, ax=axs[1], palette=colors_df_1, saturation=0.75)
        if(label_rotation):
            s.set_xticklabels(s.get_xticklabels(),rotation=90)
        axs[1].set_title("Count : TARGET = 1 ")

        s = sns.countplot(x=temp_col, data=df_0, ax=axs[2], palette=colors_df_0, saturation=0.75)
        if(label_rotation):
            s.set_xticklabels(s.get_xticklabels(),rotation=90)
        axs[2].set_title("Count : TARGET = 0 ")

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        plt.show()



def plot_categ_vs_numeric_columns(df, categ_col, numeric_col, nan_as_categ=None, palette="bright"):
    
    fig, axs = plt.subplots(ncols=2, figsize=(10,5))


    if(nan_as_categ):
        df_filled_na = df.copy()
        df_filled_na.loc[:, categ_col] = df_filled_na[categ_col].fillna('null')
        

        df_0_filled = df_filled_na.loc[df_filled_na['TARGET'] == 0,:]
        df_1_filled = df_filled_na.loc[df_filled_na['TARGET'] == 1,:]


        num_categories_df_0 = len(df_0_filled[categ_col].unique())
        num_categories_df_1 = len(df_1_filled[categ_col].unique())

        colors_df_0 = generate_colors(num_categories_df_0, palette = palette)
        colors_df_1 = generate_colors(num_categories_df_1, palette = palette)

        



        s = sns.boxplot(data=df_0_filled,  x= df_0_filled[numeric_col],  y= df_0_filled[categ_col],
                ax=axs[0], 
                palette=colors_df_0, 
                saturation=0.75, 
                showfliers=False
                 ).set_title(f"{numeric_col} : TARGET = 0 ", fontsize=16)


        s = sns.boxplot(data=df_1_filled,  x= df_1_filled[numeric_col],  y= df_1_filled[categ_col],
                ax=axs[1], 
                palette=colors_df_1, 
                saturation=0.75, 
                showfliers=False
                 ).set_title(f"{numeric_col} : TARGET = 1 ", fontsize=16)

        axs[0].tick_params(axis='y', labelsize=15)
        axs[1].set(yticks=[])
        axs[1].set(ylabel=None)
        plt.tight_layout(pad=0.5)
        plt.show()


    else:
        
        df_0 = df.loc[df["TARGET"] == 0,:]
        df_1 = df.loc[df["TARGET"] == 1,:]

        num_categories_df_0 = len(df_0[categ_col].unique())
        num_categories_df_1 = len(df_1[categ_col].unique())

        colors_df_0 = generate_colors(num_categories_df_0, palette = palette)
        colors_df_1 = generate_colors(num_categories_df_1, palette = palette)


        s = sns.boxplot(data=df_0,  x= df_0[numeric_col],  y= df_0[categ_col],
                ax=axs[0], 
                palette=colors_df_0, 
                saturation=0.75, 
                showfliers=False
                 ).set_title(f"{numeric_col} : TARGET = 0 ", fontsize=16)

        s = sns.boxplot(data=df_1,  x= df_1[numeric_col],  y= df_1[categ_col],
                ax=axs[1], 
                palette=colors_df_1, 
                saturation=0.75, 
                showfliers=False
                 ).set_title(f"{numeric_col} : TARGET = 1 ", fontsize=16)


        axs[0].tick_params(axis='y', labelsize=15)
        axs[1].set(yticks=[])
        axs[1].set(ylabel=None)
        plt.tight_layout(pad=0.5)
        plt.show()





# Code Copié : https://github.com/samirhinojosa/OC-P4-consumption-needs-of-buildings/blob/master/analysis_notebook.ipynb

# Distribution initiale et transformées des colonnes : columns

def plot_distribution(df, columns, hue_col=None):
    # Copie du df
    df_ = df.copy()
    
    # Transformation Log, Log2, et Log10
    for col in columns:
        df_[col + "_log"] = np.log(df_[col] + 1)
        df_[col + "_log2"] = np.log2(df_[col] + 1)
        df_[col + "_log10"] = np.log10(df_[col] + 1)

    for var in columns:

        # We are going to work only with the rows without missing-values for the features
        df_subset = pd.DataFrame(df_[df_[[col]].notnull().all(axis=1)]).reset_index(drop=True)

        var_cols = [var + "", var + "_log", var + "_log2", var + "_log10"]

        fig = plt.figure(constrained_layout=True, figsize=[15,10])
        fig.suptitle(var, size=25, fontweight="bold", y=1.05)
        spec = fig.add_gridspec(ncols=2, nrows=4, width_ratios=[1,1], height_ratios=[0.3,2,0.3,2])

        # to cycle through the columns 
        col_boxplot, col_histplot = 0, 0

        for i in range(4):

            for j in range(2):

                if i % 2 == 0:

                    if col_boxplot > len(var_cols) - 1:
                        break

                    ax_box = fig.add_subplot(spec[i, j])
                    boxplot = sns.boxplot(data=df_subset, x=var_cols[col_boxplot], ax=ax_box)

                    # Remove x axis name for the boxplot
                    ax_box.set(xlabel="", xticks=[])
                    ax_box.set(yticks=[])

                    boxplot.set_title(var_cols[col_boxplot], fontdict={ "fontsize": 15, "fontweight": "bold" })

                    col_boxplot += 1

                elif i % 2 != 0:

                    if col_histplot > len(columns) - 1:
                        break

                    ax_hist = fig.add_subplot(spec[i, j])
                    sns.histplot(data=df_subset, x=var_cols[col_histplot], bins=100,  kde=True,  ax=ax_hist, hue=hue_col)
                    ax_hist.set(xlabel=var_cols[col_histplot])

                    col_histplot += 1

    #     plt.savefig("figures/transformation-" + var + ".png", transparent=True, bbox_inches='tight', dpi=200)
        sns.despine(fig)  
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