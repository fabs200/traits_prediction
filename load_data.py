import warnings

import numpy as np
import pandas as pd

from config import project_path, external_data_path
from targets import behavioral_traits

# capture warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

df_prep = pd.read_stata(project_path + "/data/03_prep_final_depots_by-token.dta")

# set dummies
df_prep['i_male_recoded'] = df_prep['i_male_imp'].map({'female': 0, 'male': 1})
df_prep['has_depot_recoded'] = df_prep['has_depot'].map({'not invested': 0, 'invested': 1})

# Exclude single obs where we partly missings among traits, and thus predicted y have not same lenghts
# df_prep = df_prep[df_prep["token"] != "X9464974DA0"]

for trait in behavioral_traits:
    df_prep[trait].fillna(df_prep[trait].median())

# make dummies of all behavioral traits as target vars
for trait in behavioral_traits:
    df_prep[f'i_{trait}'] = 0
    df_prep.loc[df_prep[trait] >= df_prep[trait].median(), f'i_{trait}'] = 1
    df_prep[f'i_{trait}'] = df_prep[f'i_{trait}'].astype(int)

# drop vars in consumption_subcategories if x% are zeros.
# Note: All in consumption_features are filled. Thus, no need for additional filtering here
# UPDATE: features that are missing are already exluded in feature_sets.py

# drop_zero_var_list = df_prep[consumption_features +
#                              consumption_subcategories +
#                              demographics_features +
#                              financial_account_features].quantile(q=model['drop_consumption_vars_at_pct']).reset_index()
# drop_zero_var_list = drop_zero_var_list[drop_zero_var_list[model['drop_consumption_vars_at_pct']] == 0.0]
# print("Following consumption vars mostly missing and will be excluded", drop_zero_var_list['index'].to_list())
# df_prep = df_prep.drop(drop_zero_var_list['index'], axis=1)
