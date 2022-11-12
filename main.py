import warnings
import time

import pandas as pd
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

from config import model, feature_sets, random_forest_param_grid, tables_path, graphs_path
from load_data import df_prep
from model_evaluation import ModelEvaluation
from utils import store_results, extract_model_specification
from targets import behavioral_traits

# capture warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DataConversionWarning)

# Timer starts
starttime = time.time()

"""
Specify # of selected feature_sets and dependent var
"""

# TEST
# i_ = 0
# feat_set_ = "feature_set_2"
# depvar = behavioral_traits[0]
# depvar='selfcontrol_factor'

if __name__ == "__main__":

    for i_, feat_set_ in enumerate(feature_sets):

        # store predictions and models of all behavioral traits in here
        y_tests_collected = {}
        y_preds_collected = {}
        y_trains_collected = {}
        models_collected = {}
        model_specs_collected = {}

        for depvar in behavioral_traits:

            print(f"{feat_set_} {depvar}")
            feature_set_ = feature_sets[feat_set_]

            model_specs_ = extract_model_specification(method=model['method'],
                                                       selected_feature_set=feat_set_,
                                                       depvar=depvar)

            # if not model['do_grid_seary']: print(model['current_model'])

            # set dataframe, X, y
            df = df_prep[feature_sets[feat_set_] + [depvar]].dropna(axis=0)
            X = df[feature_sets[feat_set_]]
            y = df[depvar]

            """
            Train and test
            """

            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=model['test_size'],
                                                                random_state=model['random_state'])
            y_tests_collected[depvar] = y_test
            y_trains_collected[depvar] = y_train

            model_ = None

            if model['method'] == 'logistic':
                # instantiate the model (using the default parameters)
                model_ = LogisticRegressionCV(Cs=model['Cs'],
                                              cv=model['cv'],
                                              random_state=model['random_state'],
                                              verbose=1,
                                              tol=1e-5,
                                              solver=model['solver'],
                                              penalty=model['penalty']
                                              )
                # If set to True, the scores are averaged across all folds, and the coefs and the C that corresponds to the
                # best score is taken, and a final refit is done using these parameters. Otherwise the coefs, intercepts
                # and C that correspond to the best scores across folds are averaged.

                # fit the model with data
                model_.fit(X_train, y_train)

            """
            Evaluate models
            """

            if model['method'] == 'randomforest':

                # grid search cv and initiate best_param_rf-model
                if model['do_grid_seary']:
                    model_ = GridSearchCV(estimator=RandomForestRegressor(),
                                          param_grid=random_forest_param_grid,
                                          cv=model['cv'],
                                          verbose=4)
                    model_.fit(X_train, y_train)
                    print("best parameters of grid search:\n", model_.best_params_)
                    rf = RandomForestRegressor(**model_.best_params_)

                else:
                    rf = RandomForestRegressor(min_samples_split=model['min_samples_split'],
                                               max_depth=model['max_depth'],
                                               min_samples_leaf=model['min_samples_leaf'],
                                               max_features=model['max_features'],
                                               n_estimators=model['n_estimators'])

                # fit and predict
                model_ = rf.fit(X_train, y_train)
                y_pred = model_.predict(X_test)

                # collect all for this iteration depvar
                models_collected[depvar] = model_
                model_specs_collected[depvar] = extract_model_specification(method=model['method'],
                                                                            selected_feature_set=feat_set_,
                                                                            depvar=depvar)
                y_preds_collected[depvar] = y_pred

        """
        Model Evaluations
        """

        model_eval_results = ModelEvaluation(X_train=X_train, X_test=X_test,
                                             y_trains_collected=y_trains_collected, y_tests_collected=y_tests_collected,
                                             model_method=model['method'],
                                             model_specs_collected=model_specs_collected,
                                             targets=behavioral_traits,
                                             models_collected=models_collected,
                                             y_preds_collected=y_preds_collected,
                                             plot=True, save_plot=True)

        df_feat_importances = model_eval_results.get_feature_importances(filepath=graphs_path)
        df_critereons = model_eval_results.criterions(verbose=True)
        model_eval_results.violin_plots(filepath=graphs_path)

        """
        Store results
        """

        store_results(df=df_feat_importances, filename="feature_importances", filepath=tables_path)
        store_results(df=df_critereons, filename="criterions", filepath=tables_path)

        endtime = time.time()
        print("time:", round(endtime - starttime, 2), "seconds")
