import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from config import model


color_palette = ['dodgerblue', 'gold', 'coral', 'mediumslateblue', 'darkorange', 'darkgray']


def store_results(df=None, filepath=None, filename=None, model_specs=None):

    filename_ = f"{filename}.xlsx"
    sheet_name_ = f"{filename}"
    cols_ = list(set(list(df.columns) + ['model']))

    # if not exists, create empty Excel sheet
    if not os.path.exists(filepath+filename_):
        writer = pd.ExcelWriter(filepath+filename_, engine='xlsxwriter')
        writer.save()
        pd.DataFrame([], columns=cols_).to_excel(
            filepath+filename_, sheet_name=sheet_name_, index=False
        )
        print("new file created!")

    df_full = pd.read_excel(filepath+filename_, sheet_name=sheet_name_, engine='openpyxl')
    df_full = df_full.dropna()
    df_full = pd.DataFrame(df_full, columns=cols_)

    if not model_specs is None:
        model_ = model_specs[list(model_specs.keys())[0]].\
            replace(list(model_specs.keys())[0], '').replace('model__', '')
        # model_ = model_.split("_")
        # model_ = "_".join([j for i, j in enumerate(model_) if i not in [2, 3]])
        df['model'] = model_
    df_full = df_full.append(df, ignore_index=True)
    df_full = df_full.drop_duplicates().dropna(how='all')
    df_full.to_excel(filepath+filename_, header=1, sheet_name=sheet_name_, index=False)
    print(f"{filename_} stored to {filepath}")


def extract_model_specification(method='logistic', selected_feature_set=None):
    if method is None:
        return None
    elif method == 'logistic':
        # format: model_logistic_set<feature_set>_ts<ts>_cv<cv>_Cs<Cs>_<solver>_<penalty>
        # example: model_logistic_set7_ts03_cv5_Cs1_liblinear_l2
        model_ = f"model_{model['method']}_{selected_feature_set[8:].replace('_', '')}_ts{model['test_size']}_cv{model['cv']}_Cs{model['Cs']}_{model['solver']}_{model['penalty']}"
    else:
        # format: model_randomforest_set<feature_set>_ts<ts>_mss<min_samples_split>_md<max_depth>_msl<min_samples_leaf>_mf<max_features>_ne<n_estimators>
        # example: model_randomforest_set7_ts03_mss02_md8_msl1_mf10_ne300
        if model['class_weight'] is None:
            model_ = f"model_{model['method']}_{selected_feature_set[8:]}_ts{model['test_size']}_mss{model['min_samples_split']}_md{model['max_depth']}_msl{model['min_samples_leaf']}_mf{model['max_features']}_ne{model['n_estimators']}_GS{model['do_grid_search']}"
        elif model['class_weight'] == 'balanced':
            model_ = f"model_{model['method']}_{selected_feature_set[8:]}_ts{model['test_size']}_mss{model['min_samples_split']}_md{model['max_depth']}_msl{model['min_samples_leaf']}_mf{model['max_features']}_ne{model['n_estimators']}_GS{model['do_grid_search']}_wbalanced"
        elif type(model['class_weight']) == dict:
            model_ = f"model_{model['method']}_{selected_feature_set[8:]}_ts{model['test_size']}_mss{model['min_samples_split']}_md{model['max_depth']}_msl{model['min_samples_leaf']}_mf{model['max_features']}_ne{model['n_estimators']}_GS{model['do_grid_search']}_{str(model['class_weight']).replace(': ', '_').replace('{', '').replace('}', '')}"
    return model_
    # elif method == 'logistic':
    #     # format: model_logistic_set<feature_set>_ts<ts>_cv<cv>_Cs<Cs>_<solver>_<penalty>
    #     # example: model_logistic_set7_ts03_cv5_Cs1_liblinear_l2
    #     model_ = f"model_{target}_{model['method']}_{selected_feature_set[8:].replace('_', '')}_ts{model['test_size']}_cv{model['cv']}_Cs{model['Cs']}_{model['solver']}_{model['penalty']}"
    # else:
    #     # format: model_randomforest_set<feature_set>_ts<ts>_mss<min_samples_split>_md<max_depth>_msl<min_samples_leaf>_mf<max_features>_ne<n_estimators>_<mid>_GS<do_grid_search>
    #     # example: model_randomforest_set7_ts03_mss02_md8_msl1_mf10_ne300_GSTrue
    #     model_ = f"model_{target}_{model['method']}_{selected_feature_set[8:]}_ts{model['test_size']}_mss{model['min_samples_split']}_md{model['max_depth']}_msl{model['min_samples_leaf']}_mf{model['max_features']}_ne{model['n_estimators']}_mid{model['min_impurity_decrease']}_GS{model['do_grid_search']}"


def set_figsize(number_features=None):
    if number_features is None:
        return None
    elif number_features > 60:
        return 8, 15
    elif number_features >= 45 and number_features < 60:
        return 8, 12
    elif number_features >= 20 and number_features < 45:
        return 8, 10
    else:
        return 8, 8


def set_model_params(best_params: dict, model: dict):
    for param in best_params:
        model[param] = best_params[param]


def corrmat_with_pval(df=None):
    """
    returns df correlation matrix with p-values
    https://stackoverflow.com/questions/25571882/pandas-columns-correlation-with-statistical-significance
    :param df: specify pandas.DataFrame
    :return: df with p-vals
    """
    rho = df.corr()
    pval = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
    p = pval.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x <= t]))
    return rho.round(2).astype(str) + p


def store_grid_search_params(random_forest_param_grid=dict,
                             gridsearch_model_specs_collected=dict,
                             feature_set=None,
                             filepath=None, filename=None):

    # write grid search values to df
    df = pd.DataFrame(gridsearch_model_specs_collected)
    df['params'] = df.index
    df['feature_set'] = feature_set
    # put params, feature_set as first columns
    for i, var in enumerate(['params', 'feature_set']):
        column_ = df.pop(var)
        # insert column using insert(position,column_name, first_column) function
        df.insert(i, var, column_)

    for param in random_forest_param_grid:
        df[f"gridsearch_{param}"] = str(random_forest_param_grid[param])

    filename_ = f"{filename}.xlsx"
    sheet_name_ = f"{filename}"
    cols_ = df.columns

    # if not exists, create empty Excel sheet
    if not os.path.exists(filepath + filename_):
        writer = pd.ExcelWriter(filepath+filename_, engine='xlsxwriter')
        writer.save()
        pd.DataFrame([], columns=cols_).to_excel(
            filepath+filename_, sheet_name=sheet_name_, index=False
        )
        print("new file created!")

    df_full = pd.read_excel(filepath+filename_, sheet_name=sheet_name_)
    df_full = df_full.dropna()
    df_full = pd.DataFrame(df_full, columns=cols_)

    df_full = df_full.append(df, ignore_index=True)
    df_full = df_full.drop_duplicates().dropna(how='all')
    df_full.to_excel(filepath+filename_, header=1, sheet_name=sheet_name_, index=False)
    print(f"{filename_} stored to {filepath}")
