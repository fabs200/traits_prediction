import os
import pandas as pd

from load_data import df_prep
from targets import *
from features_sets import *
from config import tables_path

feature_sets_and_targets = {"targets": targets,
                            "behavioral_traits": behavioral_traits,
                            "consumption_features": consumption_features,
                            "demographics_features": demographics_features,
                            "financial_account_features": financial_account_features,
                            "consumption_subcategories": consumption_subcategories
                            }

cols_ = \
    [item for sublist in [feature_sets_and_targets[el] for el in feature_sets_and_targets.keys()] for item in sublist]

feature_sets_and_targets_nicely = {}
for key in feature_sets_and_targets.keys():
    list_ = []
    for item in feature_sets_and_targets[key]:
        if item in feature_names_nicely.keys():
            list_.append(feature_names_nicely[item])
        feature_sets_and_targets_nicely[key] = list_

df_prep = df_prep[cols_]
for dummy_var in ['i_male_recoded', 'has_depot_recoded']:
    df_prep[dummy_var] = pd.to_numeric(df_prep[dummy_var])

df_descriptives = df_prep[cols_].rename(columns=feature_names_nicely).describe().T
df_descriptives = df_descriptives.drop(columns=["25%", "75%"])
df_descriptives['variable'] = df_descriptives.index

# if not exists, create empty Excel file
if not os.path.exists(tables_path + "descriptives.xlsx"):
    writer = pd.ExcelWriter(tables_path + "descriptives.xlsx", engine='xlsxwriter')
    writer.save()

# loop over each correlation matrix and append to new sheet in Excel file
with pd.ExcelWriter(tables_path + "descriptives.xlsx") as writer:
    for res in feature_sets_and_targets_nicely.keys():
        print(res, feature_sets_and_targets_nicely[res])
        df_descriptives.loc[feature_sets_and_targets_nicely[res]].to_excel(writer, sheet_name=res)
