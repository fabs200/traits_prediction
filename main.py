import warnings
import time

import pandas as pd
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from config import model, feature_sets, random_forest_param_grid, tables_path, graphs_path
from load_data import df_crypto_prep
from model_evaluation import model_evaluation, store_criterions_results
from utils import store_feature_importance, extract_model_specification

# capture warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DataConversionWarning)

# Timer starts
starttime = time.time()

"""
Specify # of selected feature_sets and dependent var
"""


if __name__ == "__main__":

    for i_, feat_set_ in enumerate(feature_sets):

        print(i_, feat_set_, "\n\n", "#"*40, "\n\n")
        feature_set_ = feature_sets[feat_set_]
        depvar = ['has_crypto']

        model['current_model'] = extract_model_specification(method=model['method'], selected_feature_set=feat_set_)

        if not model['do_grid_seary']: print(model['current_model'])

        # set dataframe, X, y
        df = df_crypto_prep[feature_sets[feat_set_] + depvar].dropna(axis=0)
        X = df[feature_sets[feat_set_]]
        y = df[depvar]

        """
        Train and test
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=model['test_size'],
                                                            random_state=model['random_state'],
                                                            stratify=y)

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
                model_ = GridSearchCV(estimator=RandomForestClassifier(),
                                      param_grid=random_forest_param_grid,
                                      cv=model['cv'],
                                      verbose=4)
                model_.fit(X_train, y_train)
                print("best parameters of grid search:\n", model_.best_params_)
                rf = RandomForestClassifier(**model_.best_params_)

            else:
                rf = RandomForestClassifier(min_samples_split=model['min_samples_split'],
                                            max_depth=model['max_depth'],
                                            min_samples_leaf=model['min_samples_leaf'],
                                            max_features=model['max_features'],
                                            n_estimators=model['n_estimators'],
                                            class_weight=model['class_weight'])

            # fit the model with data
            model_ = rf.fit(X_train, y_train.values.ravel())

        # predict y
        y_pred = model_.predict(X_test)

        # probabilities obs being {0,1}
        y_prob = model_.predict_proba(X)

        # Return the mean accuracy on the given test data and labels.
        score_train = model_.score(X_train, y_train)
        score_test = model_.score(X_test, y_test)

        cnf_matrix, measures = model_evaluation.get_confusion_matrix(
            y_test=y_test,
            y_pred=y_pred,
            plot=True,
            save_plot=True,
            verbose=True,
            filepath=graphs_path + f"confusion_matrix/confmat_{model['current_model']}.png"
        )

        f1_score = model_evaluation.get_f1_score(
            y_test=y_test,
            y_pred=y_pred,
            verbose=True
        )

        fpr, tpr, auc_score = model_evaluation.get_auc_score(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            y_pred=y_pred,
            model_=model_,
            plot=True, save_plot=True,
            filepath=graphs_path + f"auc/auc_{model['current_model']}.png"
        )

        importance, feat_importances = model_evaluation.get_feature_importances(
            model_method=model['method'],
            X_test=X_test,
            model_=model_,
            plot=True,
            save_plot=True,
            filepath=graphs_path + f"feature_importance/feat_imp_{model['current_model']}.png"
        )

        """
        Store results
        """

        store_feature_importance(filename=tables_path + 'model_evaluation.xlsx',
                                 df=feat_importances,
                                 current_model=model['current_model'])

        critereon_results = {}
        critereon_results['score_train'] = score_train
        critereon_results['score_test'] = score_test
        critereon_results['auc_score'] = auc_score
        critereon_results['f1_score'] = f1_score
        critereon_results['tp'] = measures['tp']
        critereon_results['fn'] = measures['fn']
        critereon_results['tn'] = measures['tn']
        critereon_results['fp'] = measures['fp']
        critereon_results['model'] = model['current_model']

        store_criterions_results(filename=tables_path + 'criterions_results.csv',
                                 critereon_results=critereon_results)

        endtime = time.time()
        print("time:", round(endtime - starttime, 2), "seconds")
