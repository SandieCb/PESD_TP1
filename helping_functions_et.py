# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 15:03:15 2023

@author: Sandie

Collection of python function to handle data type
"""
import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold
np.random.seed(12)


def specificity_score(y, y_pred):
    '''
    function to compute specificity_score.
    
    Parameters
    ----------    
    y : 1D np.array
       true labels
    y_pred : 1D np.array
       predicted labels
   
    Returns
    -------
    specificity_score : float
        specificity_score between 0 and 1
    '''
    
    cm = confusion_matrix(y, y_pred)
    [TN, FP, FN, TP] = cm.ravel()
    specificity_score = TN/(TN+FP)
    return specificity_score



def is_date(str_to_test):
    """ This function automatically infer if a string can be infer as a date
    It includes the following list of date formatting :
        "%Y-%m-%d %H:%M:%S.%f"
        "%Y-%m-%d %H:%M:%S"
        "%Y-%m-%d %H:%M"
        "%Y-%m-%d"
        "%d/%m/%Y %H:%M:%S.%f"
        "%d/%m/%Y %H:%M:%S"
        "%d/%m/%Y %H:%M"
        "%d/%m/%Y"
        "%d %B, %Y %I:%M:%S %p"
        "%d %B, %Y %H:%M:%S"
        "%d %B, %Y %I:%M %p"
        "%d %B, %Y %H:%M",
        "%B %d, %Y %I:%M:%S %p"
        "%B %d, %Y %H:%M:%S"
        "%B %d, %Y %I:%M %p"
        "%B %d, %Y %H:%M"
    Parameters
    ----------
    str_to_test : string to be tested
    
    Returns
    -------
    True/False
    """

    formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M:%S.%f",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%d %B, %Y %I:%M:%S %p",
        "%d %B, %Y %H:%M:%S",
        "%d %B, %Y %I:%M %p",
        "%d %B, %Y %H:%M",
        "%B %d, %Y %I:%M:%S %p",
        "%B %d, %Y %H:%M:%S",
        "%B %d, %Y %I:%M %p",
        "%B %d, %Y %H:%M"
    ]

    for format in formats:
        try:
            datetime.strptime(str_to_test, format)
            return True
        except ValueError:
            pass

    return False


def infer_column_types(df):
    """ This function automatically infer and set datatype of each column of a dataset
    It considers :
     - 1 ("1", "true"), 0 ("0", "false") and missing values [np.nan, "", "None", "NaN", "NA", "ND", None] --> bool
     - str : conversion to datetime or to category
     - int to int
     - float to float
     
    Parameters
    ----------
    df : DataFrame()
    
    Returns
    -------
    df : DataFrame() with a data type for each column
    """
    
    # generic list of value to be considered as missing
    missing_data_generic_list = ["nan", "", "None", "NaN", "NA", "ND", None]


    for column in df.columns:
        values = df[column].values
        column_type = df[column].dtype
        
        # category transformation
        if all(value in ['0', '1'] + missing_data_generic_list or np.isnan(value) for value in values):
            df[column] = df[column].replace({'0': 0, '1': 1})
            df[column] = df[column].astype('category')
        elif all((value in [0, 1] + missing_data_generic_list) or (np.isnan(value)) for value in values):
            df[column] = df[column].astype('category')
        elif all(value in [False, True]  + missing_data_generic_list or (np.isnan(value)) for value in values):
            df[column] = df[column].replace({False:0, True:1})
            df[column] = df[column].astype('category')
        elif all(isinstance(value, str) and (value.lower() == 'true' or value.lower() == 'false' or value.isdigit() or value in missing_data_generic_list or np.isnan(values[0])) for value in values):
            [column] = df[column].astype('category')
            
        # str to date or category transformation
        elif all((value in missing_data_generic_list) or isinstance(value, str) for value in values):
            if all((value) in missing_data_generic_list or is_date(value) for value in values):
                df[column] = pd.to_datetime(df[column])
                df[column] = df[column].astype('datetime64[ns]')
            else:
                df[column] = df[column].astype(str)
                if len(df[column]) != len(set(df[column])):
                    df[column] = df[column].astype('category')
        # int to int transformation
        elif column_type == int: 
            df[column] = df[column].astype(int)
        # float to float transformation
        elif column_type == float:
            df[column] = df[column].astype(float)                

    return df


def get_basic_RF_grid(n_features):
    '''
    function to get a basic random grid for hyper-parameter tuning
    
    Parameters
    ----------    
    n feature : int
    
    Returns
    -------
    random_grid : dict
        random_grid grid of RF hyperparameter tuning
    '''
    random_grid = {}
    
    ## RANDOM FOREST RANDOM SEARCH GRID
    # Number of trees in random forest
    n_estimators = [10, 20]
   
    # Number of features to consider at every split
    max_features = ['sqrt']
    # Maximum number of levels in tree
    max_depth = [3, 5]

    # Minimum number of samples required to split a node
    min_samples_split = [5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [10, 20]
    # Method of selecting samples for training each tree
    bootstrap = [False]
    # Method of class balancing samples for training each tree
    class_weight = ["balanced"]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'class_weight': class_weight}

    return random_grid



def generate_performance_figure(cv_results, scorers, title=''):
    '''
    function to generate results and figure with performance
    
    Parameters
    ----------    
    cv_results : object returned by cross_validate
    test_df : dataframe that contains performance computed on test
    scorers : a dict containing scorers 
        e.g. scorers = {
            'balanced_accuracy': make_scorer(balanced_accuracy_score),
            'sensibilité': make_scorer(recall_score),
            'specificité': make_scorer(specificity_score)
        }
    Returns
    -------
        None 
    '''

    
    # Affichage des résultats dans la console pour chaque métrique
    print("--> Cross-validation performances")
    for metric, values in cv_results.items():
        name_metric = metric[5:] # enlever le mot test qui porte à confusion
        if metric != "fit_time" and metric != "score_time":
            mean_value = round(values.mean(), 2)
            std_value = round(values.std(), 2)
            print(f"{name_metric}: {mean_value} +/- {std_value}")
            
    
    
    # Récupération des valeurs pour la cross-validation et le test
    cv_means = [cv_results['test_balanced_accuracy'].mean(), cv_results['test_sensibilité'].mean(), cv_results['test_specificité'].mean()]
    cv_stds = [cv_results['test_balanced_accuracy'].std(), cv_results['test_sensibilité'].std(), cv_results['test_specificité'].std()]

    # Positions des barres
    bar_positions = np.arange(len(scorers))

    # Largeur des barres
    bar_width = 0.15

    # Création du graphique
    fig, ax = plt.subplots(figsize=(6, 3))

    # Barres pour les performances en cross-validation
    ax.bar(bar_positions, cv_means, yerr=cv_stds, capsize=5, color='skyblue', alpha=0.7, width=bar_width, label='Cross-Validation')

    # Ajout des valeurs en annotation pour les performances sur le jeu de test
    for bar_cv, bar_test in zip(bar_positions, bar_positions + bar_width):
        yval_cv = cv_means[bar_cv]
        plt.text(bar_cv, yval_cv + 0.01, round(yval_cv, 2), ha='center', va='bottom')
        
    # Ajout de labels et de titres
    ax.set_ylabel('Valeur')
    ax.set_title('Cross-validation for %s' % title)
    ax.set_xticks(bar_positions + bar_width / 2)
    ax.set_xticklabels(scorers.keys())
    ax.legend(loc='lower right')

    # Affichage du graphique
    plt.show()
        

    
def identify_best_param(clf, feat, target, scorer):
    '''
    function to identify best hyper parameter of a Random Forest model based on
    Stratified Cross Validation and GridSearch
    
    Parameters
    ----------    
    clf : clf to be used

    feat_train : DataFrame
        contains predictives
    target : list
        contains associated target
    scorer : str
        indicates the metric to be optimized
    Returns
    -------
    best_params : dict
        dictionnary with best param
    '''
    
    np.random.seed(12)
    # récupération d'un dictionnaire avec les hyperparamètres à tester
    grid = get_basic_RF_grid(len(feat.columns))
    
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)

    # idenfitication des meilleurs hyperparamètres
    grid_search = GridSearchCV(estimator=clf, param_grid=grid,
                               cv=stratified_cv, verbose=0,
                               n_jobs=-1, scoring=scorer,
                               refit=True,
                               return_train_score=True)

    grid_search.fit(feat, target)

    # récupération des meilleurs hyperparamètres
    best_params = grid_search.best_params_
    print("\n Meilleurs paramètres :\n", grid_search.best_params_)


    return best_params
