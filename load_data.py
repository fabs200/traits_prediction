import warnings

import numpy as np
import pandas as pd

from config import project_path, external_data_path
from targets import behavioral_traits

# capture warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

df_transactions_daily_prep = pd.read_csv(external_data_path + "/df_prep_accounts_detailed_daily_2021_12_01.csv")
df_investorsonly_prep = pd.read_stata(project_path + "/data/04_prep_final_depots_by-token_onlyInvestors.dta")
df_portfolios = pd.read_stata(project_path + "/data/03_prep_final_depots_by-token.dta")
df_accounts_portfolios_full_l = pd.read_stata(project_path + "/data/02_prep_final_accounts_depots_full_l.dta")
df_transactions_raw = pd.read_csv(project_path + "/data/raw/account_transactions_2022_09_06.csv")
df_security_account_balances = pd.read_csv(project_path + "/data/raw/df_security_account_balances_prep_timeseries_2022-10-25.csv", low_memory=False)

# prepare sub-category aggregated means (0 set to missing to avoid excess zeros)
df_accounts_portfolios_full_l = df_accounts_portfolios_full_l.replace({'0': np.nan, 0: np.nan})
df_sc_means = df_accounts_portfolios_full_l[['token', 'monthn'] +
                                            list(df_accounts_portfolios_full_l.iloc[:, df_accounts_portfolios_full_l.columns.str.contains(r'^sc_*')].columns)] \
    .groupby('token').agg(['mean']).reset_index()
df_sc_means.columns = [col[0] for col in df_sc_means.columns.values]

# merge df_sc_means to df_crypto_prep
df_prep = df_portfolios.merge(df_sc_means,
                              on='token')

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
