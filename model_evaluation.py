import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import os.path
from utils import set_figsize
from features_sets import feature_names_nicely


class ModelEvaluation:
    def __init__(self, X_train=None, X_test=None, y_trains_collected=None, y_tests_collected=None,
                 targets=None, models_collected=None, model_specs_collected=None, y_preds_collected=None,
                 model_method='randomforest', plot=True, save_plot=True):
        self._X_train = X_train
        self._X_test = X_test
        self._y_trains_collected = y_trains_collected
        self._y_tests_collected = y_tests_collected
        self._targets = targets
        self._models_collected = models_collected
        self._model_specs_collected = model_specs_collected
        self._y_preds_collected = y_preds_collected
        self._model_method = model_method
        self._plot = plot
        self._save_plot = save_plot

    def X_train(self):
        return self._X_train

    @property
    def X_test(self):
        return self._X_test

    @property
    def y_trains_collected(self):
        return self._y_trains_collected

    @property
    def y_tests_collected(self):
        return self._y_tests_collected

    @property
    def targets(self):
        return self._targets

    @property
    def models_collected(self):
        return self._models_collected

    @property
    def model_specs_collected(self):
        return self._model_specs_collected

    @property
    def y_preds_collected(self):
        return self._y_preds_collected

    @property
    def model_method(self):
        return self._model_method

    @property
    def plot(self):
        return self._plot

    @property
    def save_plot(self):
        return self._save_plot

    def get_feature_importances(self, filepath=None) -> pd.DataFrame:

        importances_collected = {}
        df_feat_importances_collected = pd.DataFrame([], columns=['feature', 'importance', 'target', 'model'])

        if self._model_method == 'logistic':
            for i, target in enumerate(self._targets):
                importances = self.models_collected[i].coef_[0]
                curr_model_ = self._model_specs_collected[i]
                feat_importances = pd.Series(importances, index=self._X_test.columns)
                if self._plot:
                    feat_importances.nlargest(20).plot(kind='barh', title='Feature Importance')
                    plt.tight_layout()
                    if self._save_plot:
                        plt.savefig(filepath + f'feat_importance_{curr_model_}.png')
                    plt.show()
                importances_collected[target] = importances

                # put feature importances for current target into df
                df_feat_importance_curr = pd.DataFrame(feat_importances, columns=["importance"])
                df_feat_importance_curr['feature'] = df_feat_importance_curr.index
                df_feat_importance_curr['target'] = target
                df_feat_importance_curr['model'] = curr_model_

                # collect dfs
                df_feat_importances_collected = df_feat_importances_collected.append(df_feat_importance_curr,
                                                                                     ignore_index=True)

        else:
            for i, target in enumerate(self._targets):
                importances = self.models_collected[target].feature_importances_
                curr_model_ = self.model_specs_collected[target]
                # std = np.std([tree.feature_importances_ for tree in model_.estimators_], axis=0)
                # feat_importances = pd.Series(importances, index=model_.feature_names_in_)
                feat_importances = pd.Series(importances,
                                             index=[feature_names_nicely[el] for el in
                                                    self.models_collected[target].feature_names_in_])
                if self._plot:
                    feat_importances.sort_values().plot(kind='barh',
                                                        title='Feature Importance',
                                                        figsize=set_figsize(len(importances))
                                                        # ylabel='Mean decrease in impurity'
                                                        )
                    # plt.rcParams["figure.figsize"] = set_figsize(len(model_.feature_importances_))
                    plt.tight_layout()
                    if self._save_plot:
                        plt.savefig(filepath + f'feat_importance_{curr_model_}.png')
                    plt.show()

                    # set back to default figsize
                    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]

                # put feature importances for current target into df
                df_feat_importance_curr = pd.DataFrame(feat_importances, columns=["importance"])
                df_feat_importance_curr['feature'] = df_feat_importance_curr.index
                df_feat_importance_curr['target'] = target
                df_feat_importance_curr['model'] = curr_model_

                # collect dfs
                df_feat_importances_collected = df_feat_importances_collected.append(df_feat_importance_curr,
                                                                                     ignore_index=True)

        return df_feat_importances_collected

    def criterions(self, verbose=False) -> pd.DataFrame:
        """
        https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
        :param verbose: boolean, default False, display results during estimation
        :return: pd.DataFrame with all criterions collected
        """

        r2_collected, mae_collected, rmse_collected, evs_collected = {}, {}, {}, {}

        for target in self.targets:

            print("self.y_test\n", self._y_tests_collected[target])
            print("self._y_preds_collected[target]\n", self._y_preds_collected[target])

            r2_collected[target] = metrics.r2_score(y_true=self._y_tests_collected[target],
                                                    y_pred=self._y_preds_collected[target])
            mae_collected[target] = metrics.mean_absolute_error(y_true=self._y_tests_collected[target],
                                                                y_pred=self._y_preds_collected[target])
            rmse_collected[target] = metrics.mean_squared_error(y_true=self._y_tests_collected[target],
                                                                y_pred=self._y_preds_collected[target], squared=False)
            evs_collected[target] = metrics.explained_variance_score(y_true=self._y_tests_collected[target],
                                                                     y_pred=self._y_preds_collected[target])
            if verbose:
                print(f"{target}")
                print(f"r2: {round(r2_collected[target], 2)}")
                print(f"mae: {round(mae_collected[target], 2)}")
                print(f"rmse: {round(rmse_collected[target], 2)}")
                print(f"explained_variance_score_collected: {round(evs_collected[target], 2)}")

        df = pd.DataFrame([], columns=["criterion"]+self.targets)
        for i, df_metrics in enumerate([r2_collected, mae_collected, rmse_collected, evs_collected]):
            df_metrics['criterion'] = [k.split("_")[0] for k, v in locals().items() if v is df_metrics][0]
            df = df.append(pd.DataFrame(df_metrics, index=[i]))

        return df

    def violin_plots(self):
        pass
        # TODO
