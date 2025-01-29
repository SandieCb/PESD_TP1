# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:04:26 2024

@author: Sandie Cabon
"""
# library import

import pandas as pd
from helping_functions_et import infer_column_types
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

###################### Configuration : NE PAS MODIFIER ########################

print("############### CHARGEMENT DES DONNÉES #####################")

# close older figures
plt.close("all")

# load heart failure dataset
dataset = pd.read_csv("heart_failure_dataset.csv")

# apply good type to dataframe (custom function)
dataset = infer_column_types(dataset)

random_state = 12
np.random.seed(12)

###################### Configuration : NE PAS MODIFIER ########################

################################# INSTRUCTIONS ################################

# le code ne s'executera pas tant que tous les champs ne seront pas complets.
# plusieurs solutions : 
    # dans Spyder : selectionner les instructions à executer puis clic droit ('executer la selection ou la ligne courante")
    # réaliser les tests dans la console
    # reprendre les codes d'intérêt et creer un script "brouillon"  


################ II. Prise en main du jeu de données ##########################

# 1.	Combien d’observations contient le jeu de données ?
# 2.	Préciser les variables du jeu de données ainsi que leurs types.
# 3.	Quelle est la variable cible ? Combien de patients sont décédés ? Combien de patients ont survécu ?
# 4.	Commenter le contenu de chacune des variables pour les patients décédés et pour les survivants en rapportant les graphes de visualisation (boxplot, histplot…) ou les valeurs qui vous ont permis de répondre. 
#         a.	Est-ce que les proportions sont balancées ? 
#         b.	Que peut-t ‘on dire des distributions (dispersion, symétrie, variabilité, outliers, balancement…) ? Est-ce que les distributions sont similaires pour les deux groupes ?
#         c.	Est-ce que les valeurs observées vous semblent cohérentes avec vos connaissances sur ces variables ?
#         d.	Le jeu de données est-t ‘il complet ? Combien d’observations sont complètes ? Quel est le pourcentage de données manquante par variable ?
#         e.	Les variables sont-t ’elles corrélées entre elles ou avec la cible ?
        

print("############ II.	PRISE EN MAIN JEU DE DONNEES ############")

print("\n # Contenu #")
# Nombre d'observations et de variables présentes dans le jeu de données
# aide : utiliser la fonction len()
n_obs = ... 
n_var = ...

print(
    f"\nLe jeu de données contient {n_obs} observations et {n_var} variables.")

print("\n # Variable et types # \n")
# Affichage des variables avec leurs types dans la console
# aide : utiliser l'attribut "dtypes" du dataframe
for idx, feat_name in enumerate(dataset.columns):
    var_type = ...
    print(f"{feat_name}: {var_type}")


# compléter avec le nom de la variable cible
target_feat = ...
print("\n # Variable cible #")
print(f"\nLa variable cible est : {target_feat}")

# compléter avec le label associé aux  décédés
n_decedes = len(dataset[dataset[target_feat] == ... ])
n_survivants = n_obs - n_decedes

print(f"{n_decedes} patients sont décédés.")
print(f"{n_survivants} patients ont survécu.")

print("\n # Affichage des repartitions, distributions... #")
# Generer des graphes de visualisation (density, boxplot, violinplot, barplot...)
# aide : https://python-graph-gallery.com/ propose un ensemble de façons de visualiser des données ainsi que des exemples, choisissez ceux qui vous paraissent pertinents



















print("\n # Données manquantes #")

print('\n --> sur le jeu entier')
# compléter pour obtenir le nombre d'observation avec au moins une valeur manquante
# sur l'ensemble du dataset
# aide : utiliser la fonction isna(), any(axis=1) et sum() du dataframe

n_with_one_missing = ...

print("%s observations sur %s ont au moins une valeur manquante (%.2f%%)" %
      (n_with_one_missing, n_obs, n_with_one_missing/n_obs*100))

print('\n --> par variable')
# compléter pour obtenir le nombre de données manquantes par variable
# aide : utiliser les fonctions isna() et sum() du dataframe
for feat_name in list(dataset.columns):
    n_missing_number = ... 
    percent_missing = round(n_missing_number/n_obs*100, 2)
    print(f"{feat_name}: {percent_missing} %")
    
    
    
print("\n # Correlation #")
# aide : utiliser la fonction corr() de DataFrame

corr_matrix = ...

plt.figure()
sns.heatmap(corr_matrix, annot=True)
plt.show()