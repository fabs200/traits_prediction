import os
import pandas as pd
from config import model

def store_feature_importance(filename = None,
                             df = None,
                             current_model = None):

    # if not exists, create empty excel sheet
    if not os.path.exists(filename):
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        writer.save()
        pd.DataFrame([], columns=["index", "importance", "model"]).to_excel(filename, sheet_name='feature_importance', index=False)
        print("new file created!")

    df['model'] = f"{current_model}"
    feat_importances = df.reset_index()
    full_model_evaluation = pd.read_excel(filename, sheet_name='feature_importance')
    full_model_evaluation = pd.DataFrame(full_model_evaluation, columns=["index", "importance", "model"])
    full_model_evaluation = full_model_evaluation.append(feat_importances)
    full_model_evaluation = full_model_evaluation.drop_duplicates().dropna(how='all')
    full_model_evaluation.to_excel(filename,
                                   header=1,
                                   sheet_name="feature_importance",
                                   index=False)
    print(f"feature importances stored to {filename}")


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
            model_ = f"model_{model['method']}_{selected_feature_set[8:]}_ts{model['test_size']}_mss{model['min_samples_split']}_md{model['max_depth']}_msl{model['min_samples_leaf']}_mf{model['max_features']}_ne{model['n_estimators']}"
        elif model['class_weight'] == 'balanced':
            model_ = f"model_{model['method']}_{selected_feature_set[8:]}_ts{model['test_size']}_mss{model['min_samples_split']}_md{model['max_depth']}_msl{model['min_samples_leaf']}_mf{model['max_features']}_ne{model['n_estimators']}_wbalanced"
        elif type(model['class_weight']) == dict:
            model_ = f"model_{model['method']}_{selected_feature_set[8:]}_ts{model['test_size']}_mss{model['min_samples_split']}_md{model['max_depth']}_msl{model['min_samples_leaf']}_mf{model['max_features']}_ne{model['n_estimators']}_{str(model['class_weight']).replace(': ', '_').replace('{', '').replace('}', '')}"
    return model_

def set_figsize(number_features=None):
    if number_features is None:
        return None
    elif number_features > 60:
        return (8, 15)
    elif number_features >= 45 and number_features<60:
        return (8, 12)
    elif number_features >= 20 and number_features<45:
        return (8, 10)
    else:
        return (8, 8)
