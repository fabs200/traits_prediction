consumption_features = [
    'log_mc_barentnahmen',
    'log_mc_bildungberufsausga',
    'log_mc_einkommen',
    'log_mc_essen_trinken_ausg',
    'log_mc_finanzausgaben',
    'log_mc_finanzeinnahmen',
    'log_mc_freizeitausgaben',
    'log_mc_gesundheitsausgabe',
    'log_mc_haustierausgaben',
    'log_mc_kinderausgaben',
    'log_mc_lebensmittelausgab',
    'log_mc_urlaubsausgaben',
    'log_mc_shoppingausgaben',
    # 'log_mc_sonstige_ausgaben',
    # 'log_mc_sonstige_einnahmen',
    'log_mc_sparen_vorsorgeaus',
    'log_mc_mobilitaetsausgabe',
    'log_mc_versicherungensaus',
    'log_mc_wohnhaushaltsausga',
    'avg_monthn'
]

demographics_features = [
    'i_male_recoded',
    'age_imp',
    'i_married_imp',
    'i_student',
    # 'i_retired',
    # 'i_selfemployed',
    'i_employed_imp',
    'i_homeowner_imp'
]

financial_account_features = [
    'has_depot_recoded',
    'n_insurance',
    'number_of_depots',
    'total_depot_balance',
    # 'avg_monthly_avg_balance',
    'numberofcurrentaccounts',
    'daysinoverdraft',
    'numberofaccounts',
    # 'numberofdepotaccounts'
]

consumption_subcategories = [
    "log_sc_abfallbeseitigung",
    "log_sc_gas",
    "log_sc_kindergarten",
    "log_sc_rundfunkgebuehren",
    "log_sc_supermarkt",
    "log_sc_apotheke",
    # "log_sc_gebaeudeversicheru",
    "log_sc_kino",
    "log_sc_sharing_angebote",
    "log_sc_tanken",
    "log_sc_arzt",
    "log_sc_geldautomat",
    "log_sc_krankenversicherun",
    # "log_sc_sonstige_ausgaben",
    # "log_sc_taschengeld",
    "log_sc_augenoptik",
    # "log_sc_geraeteversicherun",
    "log_sc_kredit",
    # "log_sc_sonstige_essen_tri",
    "log_sc_taxi",
    "log_sc_automobilclub",
    "log_sc_gesetzliche_kranke",
    "log_sc_kreditkartenabrech",
    # "log_sc_sonstige_finanzaus",
    "log_sc_tickets",
    "log_sc_bahntickets",
    "log_sc_getraenkehandel",
    "log_sc_leasing",
    # "log_sc_sonstige_freizeita",
    "log_sc_tierarzt",
    # "log_sc_baufinanzierung",
    "log_sc_gewerkschaften",
    "log_sc_lebensmittel_abo",
    # "log_sc_sonstige_gesundhei",
    "log_sc_tierbedarf",
    "log_sc_baumarkt",
    "log_sc_grundbesitzabgaben",
    "log_sc_lebensversicherung",
    # "log_sc_sonstige_haushalts",
    "log_sc_unfallversicherung",
    "log_sc_bausparvertrag",
    "log_sc_haftpflichtversich",
    "log_sc_lieferservice",
    # "log_sc_sonstige_kinderaus",
    # "log_sc_unterhalt",
    "log_sc_bekleidungshandel",
    # "log_sc_hausgeld",
    "log_sc_lotterie",
    # "log_sc_sonstige_lebensmit",
    # "log_sc_verein_sonstige",
    # "log_sc_berufsunfaehigkeit",
    "log_sc_hausratversicherun",
    "log_sc_miete",
    # "log_sc_sonstige_shopping",
    "log_sc_verein_sport",
    "log_sc_blumenhandel",
    "log_sc_hochschule",
    "log_sc_mobilfunk",
    # "log_sc_sonstige_urlaubsau",
    "log_sc_verlag_zeitung",
    "log_sc_buecher_medien",
    "log_sc_hotel_urlaubswohnu",
    "log_sc_oepnv",
    # "log_sc_sonstige_verkehrsa",
    "log_sc_versandhandel",
    "log_sc_domain_hosting",
    "log_sc_inkasso",
    "log_sc_parken",
    # "log_sc_sonstige_versicher",
    "log_sc_virtuelle_gueter",
    "log_sc_drogerie",
    "log_sc_internet_telefon",
    # "log_sc_pflegeversicherung",
    "log_sc_sparen",
    "log_sc_wasser",
    "log_sc_einrichtung",
    # "log_sc_kantine",
    "log_sc_prime_mitgliedscha",
    "log_sc_spende",
    "log_sc_werkstatt_service",
    "log_sc_elektrohandel",
    "log_sc_kapitalanlage",
    "log_sc_rechtsschutzversic",
    "log_sc_spielsachen",
    "log_sc_zinsen_entgelte",
    "log_sc_fahrrad",
    "log_sc_kaufhaus_gemischt",
    "log_sc_reisekrankenversic",
    "log_sc_sport",
    "log_sc_fitnessstudio",
    "log_sc_kaution",
    # "log_sc_rentenversicherung",
    "log_sc_steuern_abgaben",
    "log_sc_fluege",
    "log_sc_kfz_steuer",
    "log_sc_restaurant",
    "log_sc_streaming_paytv",
    "log_sc_friseur",
    "log_sc_kfz_versicherung",
    # "log_sc_risiko_lebensversi",
    "log_sc_strom"
]

feature_names_nicely = {
    'log_mc_barentnahmen':          'cash withdrawals',
    'log_mc_bildungberufsausga':    'educational/vocational expenses',
    'log_mc_einkommen':             'income',
    'log_mc_essen_trinken_ausg':    'expenses for food and beverages',
    'log_mc_finanzausgaben':        'financial expenses',
    'log_mc_finanzeinnahmen':       'financial receipts',
    'log_mc_freizeitausgaben':      'leisure',
    'log_mc_gesundheitsausgabe':    'health',
    'log_mc_haustierausgaben':      'pets',
    'log_mc_kinderausgaben':        'children',
    'log_mc_lebensmittelausgab':    'food',
    'log_mc_urlaubsausgaben':       'holiday',
    'log_mc_shoppingausgaben':      'shopping',
    'log_mc_sonstige_ausgaben':     'other expenses',
    'log_mc_sonstige_einnahmen':    'other receipts',
    'log_mc_sparen_vorsorgeaus':    'savings',
    'log_mc_mobilitaetsausgabe':    'mobility',
    'log_mc_versicherungensaus':    'insurance',
    'log_mc_wohnhaushaltsausga':    'living expenses',
    'avg_monthn':                   'monthly transactions',
    'i_male_recoded':               'male',
    'age_imp':                      'age',
    'i_married_imp':                'married',
    'i_student':                    'student',
    'i_homeowner_imp':              'homeowner',
    'i_employed_imp':               'employed',
    'has_depot_recoded':            'has security accounts',
    'n_insurance':                  'number of insurances',
    'number_of_depots':             'number of security accounts',
    'total_depot_balance':          'total security accounts balance',
    'numberofcurrentaccounts':      'number of current accounts',
    'daysinoverdraft':              'days in overdraft',
    'numberofaccounts':             'number of accounts',
    'numberofdepotaccounts':        'number of security accounts accounts',
    'avg_sc_elektrohandel':         'electronics',
    'avg_sc_virtuelle_gueter':      'virtual goods',
    'avg_sc_geraeteversicherung':   'device insurance',
    'avg_sc_werkstatt_service':     'repair services',
    'avg_sc_prime_mitgliedschaft':  'amazon prime subscription',
    'avg_sc_lebensmittel_abo':      'food subscription',
    'avg_sc_kino':                  'cinema',
    'avg_sc_kapitalanlage':         'capital investments',
    'avg_sc_krankenversicherung':   'health insurance',
    'avg_sc_tierbedarf':            'pet needs',
    'avg_sc_tanken':                'refueling',
    'avg_sc_apotheke':              'pharmacy',
    'avg_sc_bahntickets':           'train tickets',
    'avg_sc_hotel_urlaubswohnung':  'hotel and vacation home',
    'avg_sc_kantine':               'canteen',
    'avg_sc_kreditkartenabrechnu':  'credit card statement',
    'avg_sc_taxi':                  'taxi',
    'avg_sc_drogerie':              'drugstore',
    'avg_sc_tickets':               'concert tickets',
    'avg_sc_verlag_zeitung':        'newspaper and publishers',
    'avg_sc_versandhandel':         'mail orderings',
    'avg_sc_fluege':                'flights',
    'avg_sc_lieferservice':         'food delivery',
    'avg_sc_rentenversicherung':    'pension insurance',
    'avg_sc_restaurant':            'restaurant',
    'risk_std':                     'risk preference',
    'trust_std':                    'general trust',
    'patience_std':                 'patience',
    'procrastination_std':          'procrastination',
    'effort_std':                   'effort',
    'selfcontrol_std':              'self-control',
    'selfcontrol_factor':           'self-control',
    'lossaversion_std':             'loss-aversion',
    'lossaversion_factor':          'loss-aversion',
    'risk':                         'risk preference',
    'trust':                        'general trust',
    'patience':                     'patience',
    'procrastination':              'procrastination',
    'effort':                       'effort',
    'selfcontrol':                  'self-control',
    'lossaversion':                 'loss-aversion',
    'i_risk':                       'dummy risk-seeking',
    'i_trust':                      'dummy trust',
    'i_patience':                   'dummy patience',
    'i_procrastination':            'dummy procrastination',
    'i_effort':                     'dummy effort',
    'i_selfcontrol':                'dummy self-control',
    'i_lossaversion':               'dummy loss-aversion',
    "log_sc_abfallbeseitigung":     "waste disposal",
    "log_sc_altersvorsorge":        "pension scheme",
    "log_sc_apotheke":              "pharmarcy",
    "log_sc_arzt":                  "doctor",
    "log_sc_aufrunden":             "round up (German Aufrunden)",
    "log_sc_augenoptik":            "optometry",
    "log_sc_auslandseinsatzentge":  "remuneration for foreign assignment",
    "log_sc_automobilclub":         "automobile club",
    "log_sc_autovermietung":        "car rental",
    "log_sc_bahncard":              "Bahncard Deutsche Bahn",
    "log_sc_bahntickets":           "train ticket",
    "log_sc_bargeldeinzahlung":     "cash deposit",
    "log_sc_baufinanzierung":       "home loans",
    "log_sc_baumarkt":              "construction market",
    "log_sc_bausparvertrag":        "building loan contract",
    "log_sc_beihilfe":              "grant",
    "log_sc_bekleidungshandel":     "clothing",
    "log_sc_berufsunfaehigkeit":    "disability insurance",
    "log_sc_blumenhandel":          "flowers",
    "log_sc_brillenversicherung":   "eyeglass insurance",
    "log_sc_buecher_medien":        "books, media",
    "log_sc_domain_hosting":        "domain, hosting",
    "log_sc_drogerie":              "drugstore",
    "log_sc_einrichtung":           "furniture",
    "log_sc_elektrohandel":         "electronics",
    "log_sc_elterngeld":            "parental benefit",
    "log_sc_erstattungen":          "refunds",
    "log_sc_fahrrad":               "bicycle",
    "log_sc_fernbus":               "coach",
    "log_sc_fitnessstudio":         "sports gym",
    "log_sc_fluege":                "flights",
    "log_sc_friseur":               "hair dresser",
    "log_sc_gas":                   "gas",
    "log_sc_gebaeudeversicheru":    "property insurance",
    "log_sc_geldautomat":           "ATM",
    "log_sc_geraeteversicherun":    "device insurance",
    "log_sc_gesetzliche_kranke":    "statutory health insurance",
    "log_sc_getraenkehandel":       "beverages",
    "log_sc_gewerkschaften":        "unions",
    "log_sc_grundbesitzabgaben":    "property levies",
    "log_sc_haftpflichtversich":    "liability insurance",
    "log_sc_handwerksleistungen":   "handcraft services",
    "log_sc_hausgeld":              "house money",
    "log_sc_hausratversicherun":    "homeowner's insurance",
    "log_sc_haustierversicherung":  "pet insurance",
    "log_sc_hochschule":            "university",
    "log_sc_hotel_urlaubswohnu":    "hotel",
    "log_sc_hypothek":              "mortgage",
    "log_sc_inkasso":               "debt collection",
    "log_sc_internet_telefon":      "internet, telephone",
    "log_sc_kantine":               "cantine",
    "log_sc_kapitalanlage":         "capital investment",
    "log_sc_kapitalertraege":       "capital gains",
    "log_sc_kaufhaus_gemischt":     "store miscellaneous",
    "log_sc_kaution":               "deposit",
    "log_sc_kaution_gutschrift":    "deposit credit",
    "log_sc_kfz_steuer":            "vehicle tax",
    "log_sc_kfz_versicherung":      "vehicle insurance",
    "log_sc_kindergarten":          "kindergarden",
    "log_sc_kindergeld":            "child benefits",
    "log_sc_kino":                  "cinema",
    "log_sc_krankenversicherun":    "health insurance",
    "log_sc_krankenzusatzversich":  "supplementary health insurance",
    "log_sc_kredit":                "loan",
    "log_sc_kreditauszahlung":      "loan disbursement",
    "log_sc_kreditkartenabrech":    "credit card statement",
    "log_sc_leasing":               "leasing",
    "log_sc_lebensmittel_abo":      "food subsriptions",
    "log_sc_lebensversicherung":    "life insurance",
    "log_sc_leistungen_der_bunde":  "benefits from federal labour office",
    "log_sc_lieferservice":         "delivery service",
    "log_sc_lohn_gehalt":           "salary",
    "log_sc_lotterie":              "lottery",
    "log_sc_miete":                 "rent",
    "log_sc_mieteinnahmen":         "rental income",
    "log_sc_mobilfunk":             "mobile",
    "log_sc_oel":                   "oil",
    "log_sc_oepnv":                 "public transport",
    "log_sc_parken":                "parking",
    "log_sc_pflegeversicherung":    "nursing care insurance",
    "log_sc_prime_mitgliedscha":    "Amazon Prime subscription",
    "log_sc_rechtsschutzversic":    "legal expense insurance",
    "log_sc_reisekrankenversic":    "travel health insurance",
    "log_sc_rentenversicherung":    "pension insurance",
    "log_sc_rente_pension":         "pension",
    "log_sc_restaurant":            "restaurant",
    "log_sc_risiko_lebensversi":    "term life insurance",
    "log_sc_ruecklastschrift":      "return debit",
    "log_sc_rundfunkgebuehren":     "broadcasting fees",
    "log_sc_schule":                "school",
    "log_sc_selbststaendigkeit":    "self-employment",
    "log_sc_sharing_angebote":      "Sharing",
    "log_sc_sondertilgung":         "special repayment",
    "log_sc_sonstiges_einkommen":   "other income sources",
    "log_sc_sonstige_ausgaben":     "other expenses",
    "log_sc_sonstige_ausgaben_bi":  "other educational and professional expenses",
    "log_sc_sonstige_einnahmen":    "other revenues",
    "log_sc_sonstige_essen_tri":    "other food and beverages",
    "log_sc_sonstige_finanzaus":    "other finance expenses",
    "log_sc_sonstige_freizeita":    "other leisure",
    "log_sc_sonstige_gesundheits":  "other health expenses",
    "log_sc_sonstige_haushalts":    "other budgetary expenses",
    "log_sc_sonstige_kinderaus":    "other children expenses",
    "log_sc_sonstige_lebensmit":    "other foot expenses",
    "log_sc_sonstige_sachversich":  "other property insurance",
    "log_sc_sonstige_shopping":     "other shopping",
    "log_sc_sonstige_tierausgabe":  "other pet expenses",
    "log_sc_sonstige_urlaubsau":    "other holiday expenses",
    "log_sc_sonstige_verkehrsa":    "other travelling expenses",
    "log_sc_sonstige_versicherun":  "other insurance",
    "log_sc_sparen":                "saving",
    "log_sc_spende":                "gifts",
    "log_sc_spielsachen":           "toys",
    "log_sc_sport":                 "spots",
    "log_sc_steuern_abgaben":       "taxes",
    "log_sc_streaming_paytv":       "streaming and paytv",
    "log_sc_strom":                 "electricity",
    "log_sc_studiengebuehren":      "study fees",
    "log_sc_studiengeld":           "studying money",
    "log_sc_supermarkt":            "super market",
    "log_sc_tanken":                "refueling",
    "log_sc_taschengeld":           "pocket money",
    "log_sc_taschengeld_gutschri":  "pocket money credit",
    "log_sc_taxi":                  "taxi",
    "log_sc_tickets":               "tickets",
    "log_sc_tierarzt":              "vet",
    "log_sc_tierbedarf":            "pet",
    "log_sc_unfallversicherung":    "accident insurance",
    "log_sc_unterhalt":             "alimony",
    "log_sc_unterhalt_gutschrift":  "alimony credit",
    "log_sc_verein_sonstige":       "other clubs and associations",
    "log_sc_verein_sport":          "clubs and associations",
    "log_sc_verkaufserloes":        "sales revenue",
    "log_sc_verlag_zeitung":        "newspaper and publisher",
    "log_sc_versandhandel":         "mail-order",
    "log_sc_virtuelle_gueter":      "virtual goods",
    "log_sc_wasser":                "water",
    "log_sc_werkstatt_service":     "repair shop service",
    "log_sc_zahnzusatzversicheru":  "dental supplementary insurance",
    "log_sc_zinsen_entgelte":       "interest",
}
