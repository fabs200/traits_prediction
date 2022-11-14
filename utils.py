import os
import pandas as pd
from config import model


def store_results(df=None, filepath=None, filename=None):

    filename_ = f"{filename}.xlsx"
    sheet_name_ = f"{filename}"
    cols_ = df.columns

    # if not exists, create empty excel sheet
    if not os.path.exists(filepath+filename_):
        writer = pd.ExcelWriter(filepath+filename_, engine='xlsxwriter')
        writer.save()
        pd.DataFrame([], columns=cols_).to_excel(
            filepath+filename_, sheet_name=sheet_name_, index=False
        )
        print("new file created!")

    df_full = pd.read_excel(filepath+filename_, sheet_name=sheet_name_)
    df_full = pd.DataFrame(df_full, columns=cols_)
    df_full = df_full.append(df, ignore_index=True)
    df_full = df_full.drop_duplicates().dropna(how='all')
    df_full.to_excel(filepath+filename_, header=1, sheet_name=sheet_name_, index=False)
    print(f"{filename_} stored to {filepath}")


def store_criterions_results(df=None, filepath=None, critereon_results=None):
    filename_ = "criterions.xlsx"


def extract_model_specification(method='logistic', selected_feature_set=None, depvar=None):
    if method is None:
        return None
    elif method == 'logistic':
        # format: model_logistic_set<feature_set>_ts<ts>_cv<cv>_Cs<Cs>_<solver>_<penalty>
        # example: model_logistic_set7_ts03_cv5_Cs1_liblinear_l2
        model_ = f"model_{depvar}_{model['method']}_{selected_feature_set[8:].replace('_', '')}_ts{model['test_size']}_cv{model['cv']}_Cs{model['Cs']}_{model['solver']}_{model['penalty']}"
    else:
        # format: model_randomforest_set<feature_set>_ts<ts>_mss<min_samples_split>_md<max_depth>_msl<min_samples_leaf>_mf<max_features>_ne<n_estimators>_<mid>
        # example: model_randomforest_set7_ts03_mss02_md8_msl1_mf10_ne300
        model_ = f"model_{depvar}_{model['method']}_{selected_feature_set[8:]}_ts{model['test_size']}_mss{model['min_samples_split']}_md{model['max_depth']}_msl{model['min_samples_leaf']}_mf{model['max_features']}_ne{model['n_estimators']}_mid{model['min_impurity_decrease']}"
    return model_


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
