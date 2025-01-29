# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:27:21 2024

@author: Sandie
"""

# library import

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import make_scorer, recall_score, balanced_accuracy_score
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from helping_functions import infer_column_types, specificity_score, generate_performance_figure, identify_best_param
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import seaborn as sns

###################### Settings : DO NOT MODIFY ###############################

print("############### CHARGEMENT DES DONNÉES #####################")

# close older figures
plt.close("all")

# load heart failure dataset
dataset = pd.read_csv("heart_failure_dataset.csv")

# apply good type to dataframe (custom function)
dataset = infer_column_types(dataset)

# Définition des scorers
scorers = {
    'balanced_accuracy': make_scorer(balanced_accuracy_score),
    'sensibilité': make_scorer(recall_score),
    'specificité': make_scorer(specificity_score)
}

scorer = 'balanced_accuracy'

random_state = 12
np.random.seed(12)

###################### Settings : DO NOT MODIFY ###############################

################################# INSTRUCTIONS ################################

# le code ne s'executera pas tant que tous les champs ne seront pas complets.
# plusieurs solutions : 
    # dans Spyder : selectionner les instructions à executer puis clic droit ('executer la selection ou la ligne courante")
    # réaliser les tests dans la console
    # reprendre les codes d'intérêt et creer un script "brouillon"  


# récupération des variables en fonction de leurs types
target_feat = ...
num_feat_names =  list(dataset.select_dtypes(include=np.number).columns)
cat_feat_names = list(dataset.select_dtypes(include='category').columns)
cat_feat_names.remove(target_feat)

print("\n")
print("############ III. DEVELOPPEMENT MODELE 0 ############")
print("\n # Preparation des données #")
print("\n --> Gestion des données manquantes")

# completer pour récupérer le jeu de données sans les observations avec une valeur manquantes
# aide : utiliser la fonction dropna()
clean_dataset = ...

# Nombre d'observations et de variables présentes dans le jeu de données
# aide : utiliser la fonction len()
n_obs_new = ...
n_var_new = ...

print(f"\nLe jeu de données contient {n_obs_new} observations et {n_var_new} variables.")

# Séparation des variables prédictives et de la variable cible
clean_dataset_feat = clean_dataset.drop(target_feat, axis=1)
clean_dataset_target = clean_dataset[target_feat]


# Compléter le pipeline permettant de mettre à l'echelle les variables numériques avec un RobustScaler()
# les variables binaire/catégorielle restent inchangées.
# aide 1 : https://machinelearningmastery.com/columntransformer-for-numerical-and-categorical-data/
# aide 2 : https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html

preprocessor0 = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('scaler', ... ),
        ]), num_feat_names),
        ('cat', ... , cat_feat_names)
    ])


# On applique le pipeline de préparation des données sur le jeu en s'assurant de garder les noms des variables soit retenus pour la suite
# On utilise la fonction fit_transform() du preprocessor0
feat_names = list(clean_dataset_feat.columns)
X_scaled_train = preprocessor0.fit_transform(clean_dataset_feat)
clean_feat_train_scaled = pd.DataFrame(X_scaled_train, columns=feat_names)


print("\n # Identification des meilleurs hyperparamètres #")
# on crée un RandomForestClassifier() avec random_state=random_state (utiliser un random_state fixe permet la reprocuctibilité des résultats)
clf = ...

# aide : tiliser la fonction identify_best_param(clf, feat, target, scorer) proposée par helping_functions.py
best_params = ...
best_model = RandomForestClassifier(**best_params)
    

print("\n # Cross-validation en cours #")
# Calcul des performances en stratified cross validation en utilisant les meilleurs hyperparametres

# Configuration de cross-validation pour qu'elle soit stratifiée
# aide : utiliser StratifiedKFold(), avec 5 découpages, shuffle=True et random_state=random_state
stratified_cv = ...

# Cross-validation
cv_results_m0 = cross_validate(best_model,
                               clean_feat_train_scaled,
                               clean_dataset_target,
                               scoring=scorers,
                               cv=stratified_cv,
                               return_train_score=False)


# Affichage des resultats
print("\n # Performances Modèle 0 # \n")
generate_performance_figure(cv_results_m0, scorers, title="Modèle 0")


################ IV. Développement d’un modèle en excluant les observations ayant des données manquantes
# Piste 1 : imputation des valeurs manquantes par la valeur médiane de la variable ou par la valeur la plus fréquente.

print("\n")
print("############ III. DEVELOPPEMENT MODELE 1 ############")
print("\n # Preparation des données #")
print("\n --> Gestion des données manquantes")

# Nombre d'observations et de variables présentes dans le jeu de données 
n_obs_new = len(dataset)
n_var_new = len(dataset.columns)

print(f"\nLe jeu de données contient {n_obs_new} observations et {n_var_new} variables.")

# Séparation des variables prédictives et de la variable cible

dataset_feat = dataset.drop(target_feat, axis=1)
clean_dataset_target = dataset[target_feat]


# Compléter le pipeline permettant de mettre à l'échelle les variables numériques avec un RobustScaler() 
# puis d'imputer les données manquantes grâce à un SimpleImputer remplçant les valeurs manquantes par la valeur médiane de la variable
# les variables binaire/catégorielle sont imputées en utilisant un SimpleImputer permettant de remplacer par la valeur la plus fréquente.
# aide 1 : https://machinelearningmastery.com/columntransformer-for-numerical-and-categorical-data/
# aide 2 : https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
# aide 3 : https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html

preprocessor1 = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('scaler', ...),
            ('imputer', ...)
        ]), num_feat_names),
        ('cat', ..., cat_feat_names)
    ])

feat_names = list(dataset_feat.columns)
X_scaled_train = preprocessor1.fit_transform(dataset_feat)
clean_feat_train_scaled = pd.DataFrame(X_scaled_train, columns=feat_names)


print("\n # Identification des meilleurs hyperparamètres #")
# on crée un RandomForestClassifier() avec random_state=random_state (utiliser un random_state fixe permet la reprocuctibilité des résultats)
clf = ...

# aide : tiliser la fonction identify_best_param(clf, feat, target, scorer) proposée par helping_functions.py
best_params = ...
best_model1 = RandomForestClassifier(**best_params)


print("\n # Cross-validation en cours #")
# Calcul des performances en stratified cross validation en utilisant les meilleurs hyperparametres

# Configuration de cross-validation pour qu'elle soit stratifiée
# aide : utiliser StratifiedKFold(), avec 5 découpages, shuffle=True et random_state=random_state
stratified_cv = ...


# Cross-validation
cv_results_m1 = cross_validate(best_model1,
                               clean_feat_train_scaled,
                               clean_dataset_target,
                               scoring=scorers,
                               cv=stratified_cv,
                               return_train_score=False)


# Affichage des resultats
print("\n # Performances Modèle 1 # \n")
generate_performance_figure(cv_results_m1, scorers, title="Modèle 1")


################ IV. Développement d’un modèle en excluant les observations ayant des données manquantes
# Piste 2 : imputation des valeurs manquantes en utilisant un K-plus proche voisin basé sur l’âge et sur le sexe du patient (k=10).
print("\n")
print("############ III. DEVELOPPEMENT MODELE 2 ############")
print("\n # Preparation des données #")
print("\n --> Gestion des données manquantes")

# Nombre d'observations et de variables présentes dans le jeu de données nettoyé
n_obs_new = len(dataset)
n_var_new = len(dataset.columns)

print(f"\nLe jeu de données contient {n_obs_new} observations et {n_var_new} variables.")

# Séparation des variables prédictives et de la variable cible

dataset_feat = dataset.drop(target_feat, axis=1)
clean_dataset_target = dataset[target_feat]


# Compléter le pipeline permettant de mettre à l'échelle les variables numériques avec un RobustScaler() 
# puis d'imputer les données manquantes grâce à un KNNImputer avec k=10 voisins remplaçant les valeurs manquantes par la valeur moyenne obtenus pour les k voisins les plus proches
# les variables binaire/catégorielle sont imputées de la même manière.
# aide 1 : https://machinelearningmastery.com/columntransformer-for-numerical-and-categorical-data/
# aide 2 : https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
# aide 3 : https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html

preprocessor2 = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('scaler', ...),
            ('imputer', ...
             )
        ]), num_feat_names),
        ('cat', ..., cat_feat_names)
    ])


feat_names = list(dataset_feat.columns)
X_scaled_train = preprocessor2.fit_transform(dataset_feat)
clean_feat_train_scaled = pd.DataFrame(X_scaled_train, columns=feat_names)

print("\n # Identification des meilleurs hyperparamètres #")
# on crée un RandomForestClassifier() avec random_state=random_state (utiliser un random_state fixe permet la reprocuctibilité des résultats)
clf = ...

# aide : tiliser la fonction identify_best_param(clf, feat, target, scorer) proposée par helping_functions.py
best_params = ...
best_model2 = RandomForestClassifier(**best_params)


print("\n # Cross-validation en cours #")
# Configuration de cross-validation pour qu'elle soit stratifiée
# aide : utiliser StratifiedKFold(), avec 5 découpages, shuffle=True et random_state=random_state
stratified_cv = ...

# Cross-validation
cv_results_m2 = cross_validate(best_model2,
                               clean_feat_train_scaled,
                               clean_dataset_target,
                               scoring=scorers,
                               cv=stratified_cv,
                               return_train_score=False)


# function to plot results
print("\n # Performances Modèle 2 # \n")
generate_performance_figure(cv_results_m2, scorers, title="Modèle 2")
