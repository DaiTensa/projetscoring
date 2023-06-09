## Projet 7 : Conception d'un modèle de Scoring et déploiement  

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



## Entraînement et évaluation du modèle



## Déploiement du modèle et de l'API



## Conception et développement du dashboard



## Résultats et discussions