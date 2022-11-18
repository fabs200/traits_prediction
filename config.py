from features_sets import *

project_path = "/Users/fabiannemeczek/Library/CloudStorage/OneDrive-Persönlich/Projekt/Traits_PSD2_Prediction/"
graphs_path = project_path + "out/graphs/prediction/"
tables_path = project_path + "out/tables/prediction/"
external_data_path = '/Users/fabiannemeczek/Dropbox/MPC_BIG5/4_data/databricks'

model = {
    'method': 'randomforest',  # randomforest OR logistic

    'test_size': 0.3,  # test size

    'random_state': 123,

    'cv': 5,  # cross validations

    ### logistic classification
    'Cs': 1,  # inverse regularization
    'solver': 'liblinear',  # liblinear solver for small sample sizes
    'penalty': 'l2',  # penalty to reduce overfitting, default l2 (l1 shrinks down coef to zero if not important)

    ### random forest
    'do_grid_search': False,  # True if do a grid search and select best_params_, otherwise below params are chosen

    'min_samples_split': 5,  # e.g. 0.1 = 10% or 10 = 10 obs, default 2
    'max_depth': 15,  # max depth of tree, # nodes, default None
    'min_samples_leaf': 10,  # The minimum number of samples required to be at a leaf node, default 1
    'max_features': "sqrt",  # The number of features to consider when looking for the best split, default "sqrt"
    'n_estimators': 300,  # number of trees
    'min_impurity_decrease': .05  # A node will be split if this split induces a decrease of the impurity greater
                                  # than or equal to this value.
}

random_forest_param_grid = {
    'bootstrap': [False],
    'max_depth': [10],  #[5, 8, 10, 12, 15],
    'max_features': [12],  #[8, 10, 12, 15],
    'min_samples_leaf': [1],  #[1, 2, 5],
    'min_samples_split': [0.1],  #[0.05, 0.1, 0.15, 0.2],
    'n_estimators': [200],  #[200, 300, 500]
}

feature_sets = {
    'feature_set_1': consumption_features,
    'feature_set_2': consumption_features + financial_account_features,
    'feature_set_3': consumption_features + demographics_features,
    'feature_set_4': consumption_features + financial_account_features + demographics_features,
    'feature_set_5': consumption_subcategories + demographics_features
}
