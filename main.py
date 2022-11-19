import warnings
import time

import pandas as pd
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from config import model, feature_sets, random_forest_param_grid, tables_path, graphs_path
from load_data import df_prep
from model_evaluation import ModelEvaluation
from utils import store_results, extract_model_specification, set_model_params
from targets import behavioral_traits, targets

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
# target = targets[0]

if __name__ == "__main__":

    for i_, feat_set_ in enumerate(feature_sets):

        # store predictions and models of all behavioral traits in here
        y_tests_collected = {}
        y_preds_collected = {}
        y_trains_collected = {}
        models_collected = {}
        model_specs_collected = {}

        # set dataframe, X, y
        df = df_prep[feature_sets[feat_set_] + targets]
        df_nonnan = df.dropna()
        X = df_nonnan[feature_sets[feat_set_]]
        y = df_nonnan[targets]

        """
        Train and test
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=model['test_size'],
                                                            random_state=model['random_state'])

        """
        Iterate over each trait and estimate and predict model
        """

        for target in targets:

            print(f"{feat_set_} {target}")
            feature_set_ = feature_sets[feat_set_]

            model_specs_ = extract_model_specification(method=model['method'],
                                                       selected_feature_set=feat_set_,
                                                       target=target)

            y_tests_collected[target] = y_test[[target]]
            y_trains_collected[target] = y_train[[target]]

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
                model_.fit(X_train, y_train[[target]])

            """
            Evaluate models
            """

            if model['method'] == 'randomforest':

                # grid search cv and initiate best_param_rf-model
                if model['do_grid_search']:
                    model_ = GridSearchCV(estimator=RandomForestClassifier(),
                                          param_grid=random_forest_param_grid,
                                          cv=model['cv'],
                                          verbose=4)
                    model_.fit(X_train, y_train)
                    print("best parameters of grid search:\n", model_.best_params_)
                    rf = RandomForestClassifier(**model_.best_params_)

                    # set best params to model specification
                    set_model_params(best_params=model_.best_params_, model=model)
                    model_specs_ = extract_model_specification(method=model['method'],
                                                               selected_feature_set=feat_set_,
                                                               target=target)
                else:
                    rf = RandomForestClassifier(min_samples_split=model['min_samples_split'],
                                                max_depth=model['max_depth'],
                                                min_samples_leaf=model['min_samples_leaf'],
                                                max_features=model['max_features'],
                                                n_estimators=model['n_estimators'],
                                                class_weight=model['class_weight'])

                # fit and predict
                model_ = rf.fit(X_train, y_train[[target]])
                y_pred = model_.predict(X_test)

                # collect all for this iteration target
                models_collected[target] = model_
                model_specs_collected[target] = extract_model_specification(method=model['method'],
                                                                            selected_feature_set=feat_set_,
                                                                            target=target)
                y_preds_collected[target] = [round(x) for x in y_pred]

        """
        Model Evaluations
        """

        model_eval_results = ModelEvaluation(X_train=X_train,
                                             X_test=X_test,
                                             y_trains_collected=y_trains_collected,
                                             y_tests_collected=y_tests_collected,
                                             model_method=model['method'],
                                             model_specs_collected=model_specs_collected,
                                             targets=targets,
                                             models_collected=models_collected,
                                             y_preds_collected=y_preds_collected,
                                             plot=True,
                                             save_plot=True)

        df_feat_importances = model_eval_results.get_feature_importances(filepath=graphs_path)
        df_critereons = model_eval_results.criterions(verbose=True)
        df_cnf_matrix_measures = model_eval_results.get_confusion_matrix(filepath=graphs_path, verbose=True)
        model_eval_results.roc_curve(filepath=graphs_path)
        df_combined_accuracies = model_eval_results.get_combined_accuracies(filepath=graphs_path)
        # model_eval_results.violin_plots(filepath=graphs_path)
        # model_eval_results.boxplots(filepath=graphs_path)

        """
        Store results
        """

        store_results(df=df_feat_importances, filename="feature_importances", filepath=tables_path)
        store_results(df=df_critereons, filename="criterions", filepath=tables_path, model_specs=model_specs_collected)
        store_results(df=df_cnf_matrix_measures, filename="cnf_matrix", filepath=tables_path,
                      model_specs=model_specs_collected)
        store_results(df=df_combined_accuracies, filename="combined_accuracies", filepath=tables_path,
                      model_specs=model_specs_collected)

        endtime = time.time()
        print("time:", round(endtime - starttime, 2), "seconds")
