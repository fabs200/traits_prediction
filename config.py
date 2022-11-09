from features_sets import *

project_path = "/Users/fabiannemeczek/Library/CloudStorage/OneDrive-Persönlich/Projekt/Crypto_Ownership/"
graphs_path = project_path + "out/graphs/08_prediction/"
tables_path = project_path + "out/tables/07_prediction/"
external_data_path = '/Users/fabiannemeczek/Dropbox/MPC_BIG5/4_data/databricks'

model = {
    'method': 'randomforest',  # randomforest OR logistic

    'test_size': 0.35,  # test size

    'random_state': 123,

    'cv': 5,  # cross validations

    ### logistic classification
    'Cs': 1,  # inverse regularization
    'solver': 'liblinear',  # liblinear solver for small sample sizes
    'penalty': 'l2',  # penalty to reduce overfitting, default l2 (l1 shrinks down coef to zero if not important)

    ### random forest
    'do_grid_seary': False,  # True if do a grid search and select best_params_, otherwise below params are chosen

    'min_samples_split': 0.1,  # e.g. 0.1 = 10% or 10 = 10 obs, default 2
    'max_depth': 10,  # max depth of tree, # nodes, default None
    'min_samples_leaf': 1,  # The minimum number of samples required to be at a leaf node, default 1
    'max_features': 10,  # The number of features to consider when looking for the best split, default "sqrt"
    'n_estimators': 500,  # number of trees
    'class_weight': "balanced"  # default: None, class weights e.g. y={0, 1} -> weights={1:10}
}

random_forest_param_grid = {
    'bootstrap': [False],
    'max_depth': [5, 8, 10, 12, 15],
    'max_features': [8, 10, 12, 15],
    'min_samples_leaf': [1, 2, 5],
    'min_samples_split': [0.05, 0.1, 0.15, 0.2],
    'n_estimators': [200, 300, 500]
}


feature_sets = {
    'feature_set_1': consumption_features,
    'feature_set_2': consumption_features + demographics_features,
    'feature_set_3': financial_account_features,
    'feature_set_4': financial_account_features + demographics_features,
    'feature_set_5': financial_account_features + consumption_features,
    'feature_set_6': financial_account_features + consumption_features + demographics_features,
    'feature_set_7': financial_account_features + consumption_features + demographics_features + selected_features,
    'feature_set_8': financial_account_features + consumption_features + selected_features,
    'feature_set_9': consumption_features + demographics_features + selected_features,
    'feature_set_10': consumption_features + selected_features,
    'feature_set_11': selected_features,
    'feature_set_12': consumption_features + behavioral_traits_features,
}

