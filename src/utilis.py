import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
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
        recap.set_index('NomColonne', inplace=True)

        return recap
            
################################################# PLOTS #################################################

def plot_stats(df, feature, label_rotation=False, horizontal_layout=True, traget="TARGET"):
    
    
    temp = df[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index, feature: temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = df[[feature, traget]].groupby([feature],as_index=False).mean()
    cat_perc.sort_values(by=traget, ascending=False, inplace=True)
    
    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12,14))
    sns.set_color_codes("pastel")


    s = sns.barplot(ax=ax1, x = feature, y= feature ,data=df1)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    
    s = sns.barplot(ax=ax2, x = feature, y= traget, order=cat_perc[feature], data=cat_perc)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.ylabel('Pourcentage avec la TARGET = 1 [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.show()



def count_plot_for_object(df, temp_col, target = "TARGET", label_rotation=False):

    df_0 = df.loc[df[target] == 0,:]
    df_1 = df.loc[df[target] == 1,:]

    fig, axs = plt.subplots(ncols=3, figsize=(10,5))
    s = sns.countplot(x= temp_col, data=df, ax=axs[0])
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    axs[0].set_title("Total")

    s = sns.countplot(x= temp_col, data=df_1, ax=axs[1])
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    axs[1].set_title("Percent : TARGET = 1 ")

    s = sns.countplot(x=temp_col, data=df_0, ax=axs[2])
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    axs[2].set_title("Percent : TARGET = 0 ")

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.show()


def plot_count_percent_for_object(df, temp_col):

    df_0 = df.loc[df['TARGET'] == 0,:]
    df_0 = df.loc[df['TARGET'] == 1,:]

    fig, axs = plt.subplots(ncols=3, figsize=(10,5))

    # Count plot
    sns.countplot(x= temp_col, data=df, ax=axs[0])
    axs[0].set_title("Count")

    # Bar plot
    sns.barplot(x= temp_col, y='TARGET', data=df, ax=axs[1])
    axs[1].set_title("Percent : TARGET = 1 ")
    axs[1].set_ylim([0,1])

    sns.barplot(x=temp_col, y='TARGET', data=df, ax=axs[2])
    axs[2].set_title("Percent : TARGET = 0 ")
    axs[2].set_ylim([0,1])

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

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


def bar_plots(df, features=[],num_cols=3 ):
    
    num_plots = len(features)
    num_rows = math.ceil(num_plots / num_cols)
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(18, 6*num_rows))
    fig.suptitle('Barplots of Categorical Features')
    for i, feature in enumerate(features):
        row = i // num_cols
        col = i % num_cols
        temp = df[feature].value_counts()
        data = pd.DataFrame({'labels': temp.index, 'values': temp.values})
        sns.set_color_codes("pastel")
        bar_plot = sns.barplot(x='labels', y='values', data=data, ax=axes[row][col])
        bar_plot.set_title(feature)
        axes[row][col].set_xlabel('')
        axes[row][col].set_xticklabels([])
        for j, label in enumerate(data['labels']):
            bar_plot.text(j, data['values'][j], f"{data['values'][j]:,}", ha='center', va='bottom')
            handles, labels = bar_plot.get_legend_handles_labels()
            axes[row][col].legend(handles=handles, labels=labels)
            fig.legend(handles, labels, loc='lower right')
    plt.show()



def plot_category(data, col_name="", num_categories = 4, labels = None, plot_name ="",save_fig=False):
    
    """
    plot_category() : Création de catégories de la colonne col_name. 
    
    Paramètres: 
    ***********
    data : données à visualiser 
    col_name : le nom de la colonne. 
    num_categories : Nombre de catégories. par défaut : 4.
    labels : Résultat labels des clusters. 
    plot_name : Nom du graphique pour l'enregistrement. Spécifier le chemin du dossier dans lequel enregistrer le graphique. 
    save_fig : True : sauvagrder de la figure au format PNG. 
    
    
    return:
    *******
    graphique : distribution des catégories en pourcentage (%) pour chaque cluster selon la variable col_name.
    
    """
    # Copie du data original. 
    data_plot = data.copy()
    
    # Création de la colonne cluster. 
    data_plot['cluster'] = labels
    
    # Utilisez la fonction pd.cut() pour créer les catégories basées sur la colonne col_name.
    data_plot[f'{col_name}_category'] = pd.cut(data_plot[col_name], bins= num_categories, include_lowest=True , precision = 0)
    
    # groupby by = 'cluser' et col_name_category et renomage de la colonne par Total
    df_to_plot = data_plot.groupby(['cluster', f'{col_name}_category']).size().reset_index(name='Total')
    
    # Calcul du pourcentage pour chaque catégorie. 
    df_to_plot['Percentage'] = round(100* df_to_plot['Total'] / df_to_plot.groupby(['cluster'])['Total'].transform('sum'), 2)
    
    # Pivot pour traçer le graphique
    df_to_plot = df_to_plot.pivot_table('Percentage', 'cluster', f'{col_name}_category')
    
    # --- Visualisation ---
    #--Config figure------: 
    xy_label = dict(fontweight='bold', fontsize=9)
    colors = generate_colors(num_categories)
    suptitle = dict(fontweight='heavy', x=0.124, y=0.98, ha='left', fontsize=12)
    title = dict(style='italic', fontsize=8, loc='left')
    tick_params = dict(length=3, width=1, color='#CAC9CD')
    
    #----Catégories----:
    categories_ = list(df_to_plot)[:]
    
    
    #----Plot---------:
    ax = df_to_plot.plot(kind='barh', 
                         stacked=True, 
                         figsize=(9, 5), 
                         edgecolor='black', 
                         color=colors, 
                         linewidth=0.5, 
                         alpha=0.85, 
                         zorder=3)
    
    plt.ylabel('Cluster\n', **xy_label)
    for rect in ax.patches:
        width, height = rect.get_width(), rect.get_height()
        x, y = rect.get_xy()
        if width > 5:
            ax.text(x+width/2, y+height/2, '{:.1f}%'.format(width), fontsize=6, horizontalalignment='center', verticalalignment='center')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.grid(axis='y', alpha=0, zorder=2)
    plt.grid(axis='x', which='major', alpha=0.3, color='#9B9A9C', linestyle='dotted', zorder=1)
    plt.legend(categories_, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, borderpad=2, frameon=False, fontsize=8, columnspacing=3)
    plt.suptitle(f'Distribution des Cluster selon : {col_name}') 
    plt.tick_params(bottom='on', **tick_params)
    for spine in ax.spines.values():
        spine.set_color('None')
    for spine in ['bottom']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('#CAC9CD')
        
        
    # Save Figure:
    if save_fig:
        plt.savefig(f"figures/EDA/plot_{plot_name}.png", transparent=True, bbox_inches='tight', dpi=200)
    plt.show()

# Fonction pour générer les couleurs pour les cluster
def generate_colors(num_clusters, palette='colorblind'):
    
    """
    Selon le nombre de Cluster : num_cluster, on vas générer des couleurs au format hex.
    
    deep, muted, bright, pastel, dark, colorblind
    """
    
    colors = color = sns.color_palette(palette= palette, n_colors= num_clusters, desat=None, as_cmap=False).as_hex()
    return colors


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