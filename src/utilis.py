import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
import dill
from sklearn.metrics import accuracy_score, precision_score, recall_score
#from sklearn.metrics import metrics /!\ /!\ /!\ /!\ /!\



    
#########################################Save file pickl model #################################################

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok = True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
        
def save_transformed_df(df, path, filename):
    
    # Fonction pour Save un df, dans le dossier : path avec le nom : filename, 
    # avec sélection des colonnes : columns_to_save
    
    df.to_csv(path + filename, index=False)

            
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


def bar_plots_count(df, features=[]):
    num_plots = len(features)
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(20, 4*num_plots))
    fig.suptitle('Barplots of Categorical Features')
    for i, feature in enumerate(features):
        temp = df[feature].value_counts()
        data = pd.DataFrame({'labels': temp.index, 'values': temp.values})
        sns.set_color_codes("pastel")
        bar_plot = sns.barplot(y='labels', x='values', data=data, ax=axes[i], orient='horizontal')
        bar_plot.set_title(feature)
        axes[i].set_xlabel('')
        
    plt.show()

def barplot_groupby_categ_features_numeric(df, categ_col, numeric_col, palette="bright"):
    
    num_categories = len(df[categ_col].unique())
    colors = generate_colors(num_categories, palette = palette)
    sns.barplot(data=df, x=numeric_col, y=categ_col, 
       hue=None, 
       order=None,
       hue_order=None, 
       estimator='mean', 
       errorbar=('ci', 95), 
       n_boot=1000, 
       units=None, 
       seed=None, 
       orient=None, 
       color=None, 
       palette=colors, 
       saturation=0.75, 
       width=0.75, 
       errcolor=None, 
       errwidth=None, 
       ax=None).set_title(f"{categ_col} - {numeric_col}")
    plt.xlabel(numeric_col)
    plt.ylabel(categ_col)
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

        
###################################################Models Evaluation ############################################

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        accuracy={}
        
        for i in range(len(list(models))):
            model= list(models.values())[i]
            model.fit(X_train, y_train) # Train model
            
            y_train_pred= model.predict(X_train)
            y_test_pred= model.predict(X_test)
            
            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)
            
            accuracy[list(models.keys())[i]] = test_model_score
         
        return accuracy
    
    except Exception as e:
        raise CustomException(e, sys)

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