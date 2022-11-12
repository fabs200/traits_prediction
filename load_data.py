import warnings

import numpy as np
import pandas as pd

from config import project_path, external_data_path

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
df_prep['i_young'] = 0
df_prep['i_young'] = (df_prep['age_imp'] < 35).astype(int)
df_prep['has_depot_recoded'] = df_prep['has_depot'].map({'not invested': 0, 'invested': 1})
df_prep['i_virtual_goods'] = 0
df_prep['i_virtual_goods'] = (df_prep['sc_virtuelle_gueter'] > df_prep['sc_virtuelle_gueter'].median()).astype(int)
# df_crypto_prep['i_virtual_goods'] = (df_crypto_prep['sc_virtuelle_gueter'] > 0).astype(int)
df_prep['i_electronics'] = 0
df_prep['i_electronics'] = (df_prep['sc_elektrohandel'] > df_prep['sc_elektrohandel'].median()).astype(int)
# df_crypto_prep['i_electronics'] = (df_crypto_prep['sc_elektrohandel'] > 0).astype(int)
df_prep['i_traintickets'] = 0
df_prep['i_traintickets'] = (df_prep['sc_bahntickets'] > df_prep['sc_bahntickets'].median()).astype(int)
# df_crypto_prep['i_traintickets'] = (df_crypto_prep['sc_bahntickets'] > 0).astype(int)
df_prep['i_newspaper_publisher'] = 0
df_prep['i_newspaper_publisher'] = (df_prep['sc_verlag_zeitung'] > df_prep['sc_verlag_zeitung'].median()).astype(int)
# df_crypto_prep['i_newspaper_publisher'] = (df_crypto_prep['sc_verlag_zeitung'] > 0).astype(int)
df_prep['i_device_insurance'] = 0
df_prep['i_device_insurance'] = (df_prep['sc_geraeteversicherung'] > df_prep['sc_geraeteversicherung'].median()).astype(int)
# df_crypto_prep['i_device_insurance'] = (df_crypto_prep['sc_geraeteversicherung'] > 0).astype(int)
df_prep['i_mobility'] = 0
df_prep['i_mobility'] = (df_prep['avg_mc_mobilitaetsausgaben'] > df_prep['avg_mc_mobilitaetsausgaben'].median()).astype(int)
# df_crypto_prep['i_mobility'] = (df_crypto_prep['avg_mc_mobilitaetsausgabe'] > 0).astype(int)
df_prep['i_internet_telephone_low'] = 0
df_prep['i_internet_telephone_low'] = (df_prep['sc_internet_telefon'] > df_prep['sc_internet_telefon'].median()).astype(int)
# df_crypto_prep['i_internet_telephone_low'] = (df_crypto_prep['sc_internet_telefon'] > 0).astype(int)

# dummy geek if consumes in any of the following cats
df_prep['geek_consumption'] = 0
for col in ['sc_virtuelle_gueter', 'sc_elektrohandel', 'sc_geraeteversicherung', 'sc_domain_hosting', 'sc_werkstatt_service']:
    df_prep['geek_consumption'] = df_prep['geek_consumption'] + df_prep[col]
df_prep['i_geek'] = 0
df_prep['i_geek'] = (df_prep['geek_consumption'] > 0).astype(int)

df_prep['life_style'] = 0
for col in ['sc_lieferservice', 'sc_prime_mitgliedschaft', 'sc_kino', 'sc_tickets', 'sc_verlag_zeitung', 'sc_hotel_urlaubswohnung', 'sc_versandhandel']:
    df_prep['life_style'] = df_prep['geek_consumption'] + df_prep[col]
df_prep['i_life_style'] = 0
df_prep['i_life_style'] = (df_prep['life_style'] > 0).astype(int)

# log
for col in ['sc_virtuelle_gueter',
            'sc_versandhandel',
            'sc_domain_hosting',
            'sc_sonstige_shopping',
            'sc_werkstatt_service',
            'sc_elektrohandel',
            'sc_geraeteversicherung',
            'sc_internet_telefon',
            'sc_verlag_zeitung',
            'sc_lieferservice',
            'sc_prime_mitgliedschaft',
            'sc_kino',
            'sc_tickets',
            'sc_hotel_urlaubswohnung',
            'sc_versandhandel',
            'sc_lieferservice',
            'geek_consumption',
            'life_style']:
    df_prep[f'log_{col}'] = np.log(df_prep[f'{col}'])
    df_prep[f'log_{col}'] = df_prep[f'log_{col}'].fillna(0)
    df_prep[f'log_{col}'] = df_prep[f'log_{col}'].replace([np.inf, -np.inf], 0)

# Exclude single obs where we partly missings among traits, and thus predicted y have not same lenghts
df_prep = df_prep[df_prep["token"] != "X9464974DA0"]
