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
Les données sont accessible via le lien suivant : [Télécharger les données](https://www.kaggle.com/c/home-credit-default-risk/data)

Une fois la configuration des chemins d'accès à la base de donnée effectué **[**`voir `data_config.py`**]** vous pourrez passer aux étapes suivantes. 

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


## Entraînement et évaluation du modèle



## Déploiement du modèle et de l'API



## Conception et développement du dashboard



## Résultats et discussions