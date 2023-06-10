# Projet 7 : Conception d'un modèle de Scoring et déploiement.

# Introduction

- **Objectif du projet** : L'objectif de notre projet est de mettre en œuvre un outil de **scoring crédit** au service des chargés de relation client d'une société financière. 

- **Contexte** : La société **Prêt à dépenser** propose des crédit à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt. 



## Architecture globale

Dans le tableau suivant nous donnons la liste des répertoires ainsi que le contenu de chaque dossier : 

|Dossier|Fichiers|Utilisation|Details|
|:---|:---|:---|:---|
|src/components |`data_config.py` `data_ingestion.py` `data_transformation.py` `model_trainer.py`|configuration des liens pour accéder aux données, configuration des sous dossiers |acquisition des données + transformation + training|
|src/pipeline|`predict_pipeline.py` `predict_pipeline.py`|Récupération des données d'un client, et calcul de la probabilité qu’un client rembourse son crédit.|Sélectionner les données du client avec son identifient unique + Preprocessing + Predict|
|notebook|`01_EDA` `02_Feature_Engineering` `03_Test_import_transformation_data_train_model` `04_Explainer` `05_Data_Drift`|Analyse exploratoire, préparation des données et features engineering, transformation des données et modélisation | Adaptation d'un kernel pour les besoins de notre mission, il s'agit de toutes les étapes de la construction du modèle du prétraitement des données au calcul de la probabilité de solvabilité en terminant pour l'analyse du data drift | 

## Collecte et préparation des données
Afin de pouvoir développer un **algorithme de classification** en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.) nous allons procéder comme suite : 

**Acquisition des données** : Les données sont accessible via le lien suivant : [Télécharger les données](https://www.kaggle.com/c/home-credit-default-risk/data)

Une fois la configuration des chemins d'accès à la base de donnée effectué **[**`voir data_config.py`**]** vous pourrez passer aux étapes suivantes. 

**1- EDA**
```python
from  src.components.data_ingestion  import  *

# 01 - Data Base : pour importer la base de donnée
data_base  =  DataIngestion()

# 02 - Extraction nom de fichiers avec et sans extensions
liste_name, files_liste_name  =  data_base.get_files_names()

# 03 - Pour importer un data set 
application_train  =  data_base.import_file(file_name='application_train.csv', reduce_memory_usage  =  False, number_of_rows=None)

# 04 - Pour affciher le rapport d'un data frame exécutez les lignes suivantes
apport_df_train  =  RapportDataFrame(application_train, target_column="TARGET", ID_Columns=["SK_ID_CURR", "SK_ID_BUREAU"])
rapport_df_train.rapport(nan_threshold  =  20, return_column_to_keep=False, print_rapport  =  True)
```
N'hésitez pas à regarder les autres méthodes disponibles dans le script `data_ingestion.py` 

**2- Feature_Engineering**

Exemple des transformation du dataset `application_train` : 

```python
def  application_train_preprocessing(df_):
	df  =  df_.copy()
	df['DAYS_EMPLOYED_PERC'] =  df['DAYS_EMPLOYED'] /  df['DAYS_BIRTH']
	df['INCOME_CREDIT_PERC'] =  df['AMT_INCOME_TOTAL'] /  df['AMT_CREDIT']
	df['INCOME_PER_PERSON'] =  df['AMT_INCOME_TOTAL'] /df['CNT_FAM_MEMBERS']
	df['ANNUITY_INCOME_PERC'] =  df['AMT_ANNUITY'] /  df['AMT_INCOME_TOTAL']
	df['PAYMENT_RATE'] =  df['AMT_ANNUITY'] /  df['AMT_CREDIT']
	df['ANNUITY_LENGTH'] =  df['AMT_CREDIT'] /  df['AMT_ANNUITY']
	df['ANN_LENGTH_EMPLOYED_RATIO'] =  df['ANNUITY_LENGTH'] /  df['DAYS_EMPLOYED']
	df['CHILDREN_RATIO'] =  df['CNT_CHILDREN'] /  df['CNT_FAM_MEMBERS']
	df['credit_div_goods'] =  df['AMT_CREDIT'] /  df['AMT_GOODS_PRICE']
	df['credit_minus_goods'] =  df['AMT_CREDIT'] -  df['AMT_GOODS_PRICE']
	df['reg_div_publish'] =  df['DAYS_REGISTRATION'] /  df['DAYS_ID_PUBLISH']
	df['birth_div_reg'] =  df['DAYS_BIRTH'] /  df['DAYS_REGISTRATION']
	df['document_sum'] =  df['FLAG_DOCUMENT_2'] +  df['FLAG_DOCUMENT_3'] +  df['FLAG_DOCUMENT_4'] +  df['FLAG_DOCUMENT_5'] +  df['FLAG_DOCUMENT_6'] +  df['FLAG_DOCUMENT_7'] +  df['FLAG_DOCUMENT_8'] +  df['FLAG_DOCUMENT_9'] +  df['FLAG_DOCUMENT_10'] +  df['FLAG_DOCUMENT_11'] +  df['FLAG_DOCUMENT_12'] +  df['FLAG_DOCUMENT_13'] +  df['FLAG_DOCUMENT_14'] + df['FLAG_DOCUMENT_15'] +  df['FLAG_DOCUMENT_16'] +  df['FLAG_DOCUMENT_17'] + df['FLAG_DOCUMENT_18'] + df['FLAG_DOCUMENT_19'] +  df['FLAG_DOCUMENT_20'] + df['FLAG_DOCUMENT_21']
	df['is_na_amt_annuity'] =  1.0*np.isnan(df['AMT_ANNUITY'])
	df['age_finish'] =  df['DAYS_BIRTH']*(-1.0/365) + (df['AMT_CREDIT']/df['AMT_ANNUITY']) *(1.0/12) #how old when finish

	return  df
```


## Entraînement et évaluation du modèle

Pour entrainer un modèle de Machine Learning, nous avons mis en place un pipeline qui comporte plusieurs étapes : 

 - Gestion du déséquilibre entre le nombre de bons et de moins bons clients : `undersampling`.
 - Initiation d'un `train_test_split` avec une stratégie de stratification.
 - Application d'un Preprocessing et enregistrement du Preprocessor sous un format réutilisable.  
 - Construction d'un `score métier` (voir `utilis.py` : *fonction_cout_metier()* et *evaluate_model_()*).
 - Entrainement du modèle, avec une recherche `GridSearchCV` afin d’optimiser les hyperparamètres et comparer les modèles.

**Exemple d'application**

 **Importer les données**
 ```python
 data_base  =  DataIngestion()
 row_data  =  data_base.import_file(file_name='Final_df.csv', reduce_memory_usage  =  False, number_of_rows=None)
 ```
  **Transformation des données**
```python
# Dans un prmier temps initiation d'un objet transformer_rox_data
transformer_row_data=  DataTransformation(row_data)
```  
La classe `DataTransformation()` comporte les méthodes suivantes : 

 - *initiate_train_test_split* : initiation d'un train test split.
**`target`** :str spécifier la colonne Target. 
**`test_size`** : la proportion de l'ensemble de données à inclure dans la répartition du test.
**`stratification`** : True ou False pour effectuer le split avec stratification ou pas. 

```python
X_train, X_test, y_train, y_test  = transformer_row_data.initiate_train_test_split( 
									target="TARGET", 
									test_size=0.30,
									stratification=True)
``` 

 - *get_data_frames* : Importer X_train X_test, y_train, y_test si existants. 
 - *initiate_data_transformation* : Transformation des données. 
 
```python
# Pour initier la transformation des données
# X_tra, X_test, y_train, y_test retournés : sont transformés avec une pipeline définie dans le fichier data_transformation
# avec la méthode initiate_data_transformation
# Les étape de la pipeline sont : 
# 1- application RandomUnderSampler(random_state=0)
# 2- num_pipeline
# 3- categ_piepline
# 4- fillna des valeur .inf -.inf
# 5- fit du preprocessor sur les données X_train
# 6- transform des données X_train et X_test
# 7- Save du preprocessor dans le dossier artifacts
# 8- Return X_train, X_test, y_train, y_test preprocssed
# transformer_row_data= DataTransformation()

X_train, X_test,y_train,y_test=  transformer_row_data.initiate_data_transformation(
								X_train,
								X_test,
								y_train,
								y_test,
								undersampling=  True,
								return_train_test_array=False,
								save_preprocessor=  True
								)


## Déploiement du modèle et de l'API



## Conception et développement du dashboard



## Résultats et discussions