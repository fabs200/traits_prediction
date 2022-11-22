import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from utils import set_figsize, color_palette, corrmat_with_pval
from features_sets import feature_names_nicely

sns.set(style="whitegrid")
sns.set_palette(palette=color_palette)

class ModelEvaluation:
    def __init__(self, X_train=None, X_test=None, y_trains_collected=None, y_tests_collected=None,
                 targets=None, models_collected=None, model_specs_collected=None, y_preds_collected=None,
                 model_method='randomforest', plot=True, save_plot=True, graph_format=".png"):
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
        self._graph_format = graph_format

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

    @property
    def graph_format(self):
        return self._graph_format

    def get_feature_importances(self, filepath=None) -> pd.DataFrame:

        importances_collected = {}
        df_feat_importances_collected = pd.DataFrame([], columns=['feature', 'importance', 'target', 'model'])

        if self._model_method == 'logistic':
            for i, target in enumerate(self._targets):
                importances = self.models_collected[i].coef_[0]
                filename_ = f"{target}_{self._model_specs_collected[i]}{self._graph_format}"
                feat_importances = pd.Series(importances, index=self._X_test.columns)
                if self._plot:
                    feat_importances.nlargest(10).sort_values(ascending=True) \
                        .plot(kind='barh', title='Feature Importance', color=color_palette[0])
                    plt.xlabel('Mean decreasing in impurity')
                    plt.tight_layout()
                    if self._save_plot:
                        plt.savefig(filepath + f'featimp_{filename_}')
                    plt.show()
                importances_collected[target] = importances

                # put feature importances for current target into df
                df_feat_importance_curr = pd.DataFrame(feat_importances, columns=["importance"])
                df_feat_importance_curr['feature'] = df_feat_importance_curr.index
                df_feat_importance_curr['target'] = target
                df_feat_importance_curr['model'] = {self._model_specs_collected[i]}

                # collect dfs
                df_feat_importances_collected = df_feat_importances_collected.append(df_feat_importance_curr,
                                                                                     ignore_index=True)

        else:
            for i, target in enumerate(self._targets):
                importances = self.models_collected[target].feature_importances_
                filename_ = f"{target}_{self.model_specs_collected[target]}{self._graph_format}"
                # std = np.std([tree.feature_importances_ for tree in model_.estimators_], axis=0)
                # feat_importances = pd.Series(importances, index=model_.feature_names_in_)
                feat_importances = pd.Series(importances,
                                             index=[feature_names_nicely[el] for el in
                                                    self.models_collected[target].feature_names_in_])
                if self._plot:
                    feat_importances.nlargest(10).sort_values(ascending=True).plot(kind='barh',
                                                                                   title='Feature Importance',
                                                                                   color=color_palette[0],
                                                                                   figsize=set_figsize(len(importances))
                                                                                   # ylabel='Mean decreasing in impurity'
                                                                                   )
                    plt.xlabel('Mean decreasing in impurity')
                    # plt.rcParams["figure.figsize"] = set_figsize(len(model_.feature_importances_))
                    plt.tight_layout()
                    if self._save_plot:
                        plt.savefig(filepath + f'featimp_{filename_}')
                    plt.show()

                    # set back to default figsize
                    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]

                # put feature importances for current target into df
                df_feat_importance_curr = pd.DataFrame(feat_importances, columns=["importance"])
                df_feat_importance_curr['feature'] = df_feat_importance_curr.index
                df_feat_importance_curr['target'] = target
                df_feat_importance_curr['model'] = self.model_specs_collected[target]

                # collect dfs
                df_feat_importances_collected = df_feat_importances_collected.append(df_feat_importance_curr,
                                                                                     ignore_index=True)

        return df_feat_importances_collected

    def criterions(self, verbose=False) -> pd.DataFrame:
        """
        https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
        https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
        :param verbose: boolean, default False, display results during estimation
        :return: pd.DataFrame with all criterions collected
        """
        accuracy_score_collected, f1_score_collected, precision_score_collected, recall_score_collected, \
        roc_auc_score_collected = {}, {}, {}, {}, {}

        for target in self.targets:
            # print(target, len(self._y_tests_collected[target]), len(self._y_preds_collected[target]))

            accuracy_score_collected[target] = metrics.accuracy_score(y_true=self._y_tests_collected[target],
                                                                      y_pred=self._y_preds_collected[target])
            f1_score_collected[target] = metrics.f1_score(y_true=self._y_tests_collected[target],
                                                          y_pred=self._y_preds_collected[target])
            precision_score_collected[target] = metrics.precision_score(y_true=self._y_tests_collected[target],
                                                                        y_pred=self._y_preds_collected[target])
            recall_score_collected[target] = metrics.recall_score(y_true=self._y_tests_collected[target],
                                                                  y_pred=self._y_preds_collected[target])
            roc_auc_score_collected[target] = metrics.roc_auc_score(y_true=self._y_tests_collected[target],
                                                                    y_score=self._y_preds_collected[target])

            if verbose:
                print(f"{target}")
                print(f"score: {round(accuracy_score_collected[target], 2)}")
                print(f"f1 score: {round(f1_score_collected[target], 2)}")
                print(f"precision_score: {round(precision_score_collected[target], 2)}")
                print(f"recall_score: {round(recall_score_collected[target], 2)}")
                print(f"roc_auc_score: {round(roc_auc_score_collected[target], 2)}")

        df = pd.DataFrame([], columns=["criterion"] + self.targets)
        for i, df_metrics in enumerate([accuracy_score_collected, f1_score_collected, precision_score_collected,
                                        recall_score_collected, roc_auc_score_collected]):
            df_metrics['criterion'] = [k.split("_")[0] for k, v in locals().items() if v is df_metrics][0]
            df = df.append(pd.DataFrame(df_metrics, index=[i]))

        return df

        # r2_collected, mae_collected, rmse_collected, evs_collected = {}, {}, {}, {}
        #
        # for target in self.targets:
        #     print(target, len(self._y_tests_collected[target]), len(self._y_preds_collected[target]))
        #
        # r2_collected[target] = metrics.r2_score(y_true=self._y_tests_collected[target],
        #                                         y_pred=self._y_preds_collected[target])
        # mae_collected[target] = metrics.mean_absolute_error(y_true=self._y_tests_collected[target],
        #                                                     y_pred=self._y_preds_collected[target])
        # rmse_collected[target] = metrics.mean_squared_error(y_true=self._y_tests_collected[target],
        #                                                     y_pred=self._y_preds_collected[target], squared=False)
        # evs_collected[target] = metrics.explained_variance_score(y_true=self._y_tests_collected[target],
        #                                                          y_pred=self._y_preds_collected[target])
        # Return the mean accuracy on the given test data and labels.
        # if verbose:
        #     print(f"{target}")
        # print(f"r2: {round(r2_collected[target], 2)}")
        # print(f"mae: {round(mae_collected[target], 2)}")
        # print(f"rmse: {round(rmse_collected[target], 2)}")
        # print(f"evs: {round(evs_collected[target], 2)}")

    def violin_plots(self, filepath=None):

        # prepare filename and cut out target, as we investigate all violinplots at once
        filename_ = self.model_specs_collected[self.targets[0]].replace(self.targets[0], "")
        filename_ = f"{filename_}{self._graph_format}"

        # prepare y_tests and y_preds, and put together into one df
        df_y_tests_collected = pd.DataFrame.from_dict(self._y_tests_collected)
        df_y_tests_collected['y'] = 'test data'
        # for x in self._y_preds_collected:
        #     print(x, len(self._y_preds_collected[x]))
        df_y_preds_collected = pd.DataFrame(self._y_preds_collected)
        df_y_preds_collected['y'] = 'predicted data'
        df = df_y_tests_collected.append(df_y_preds_collected, ignore_index=True)
        # rename cols (drop _std, _factor)
        df.columns = df.columns.str.replace('_std', '')
        df.columns = df.columns.str.replace('_factor', '')
        # pivot df for plotting
        df = pd.melt(df, id_vars='y', value_vars=[col for col in df.columns if col != "y"])
        if self._plot:
            g = sns.violinplot(data=df, x="variable", y="value", hue="y", color=color_palette[0])
            plt.legend(title='')
            plt.xlabel("")
            plt.tight_layout()
            if self._save_plot:
                plt.savefig(filepath + f'violinplot_{filename_}')
            plt.show()

    def boxplots(self, filepath=None):

        # prepare filename and cut out target, as we investigate all violinplots at once
        filename_ = self.model_specs_collected[self.targets[0]].replace(self.targets[0], "")
        filename_ = f"{filename_}{self._graph_format}"

        # prepare y_tests and y_preds, and put together into one df
        df_y_tests_collected = pd.DataFrame.from_dict(self._y_tests_collected)
        df_y_tests_collected['y'] = 'test data'
        # for x in self._y_preds_collected:
        #     print(x, len(self._y_preds_collected[x]))
        df_y_preds_collected = pd.DataFrame(self._y_preds_collected)
        df_y_preds_collected['y'] = 'predicted data'
        df = df_y_tests_collected.append(df_y_preds_collected, ignore_index=True)
        # rename cols (drop _std, _factor)
        df.columns = df.columns.str.replace('_std', '')
        df.columns = df.columns.str.replace('_factor', '')
        # pivot df for plotting
        df = pd.melt(df, id_vars='y', value_vars=[col for col in df.columns if col != "y"])
        if self._plot:
            g = sns.boxplot(data=df, x="variable", y="value", hue="y", color=color_palette[0])
            plt.legend(title='')
            plt.xlabel("")
            plt.tight_layout()
            if self._save_plot:
                plt.savefig(filepath + f'boxplot_{filename_}')
            plt.show()

    def get_confusion_matrix(self, filepath=None, verbose=False):

        cnf_matrix_collected, measures_collected = {}, {}
        df_cnf_matrix_measures = pd.DataFrame([], columns=['tp', 'fn', 'fp', 'tn', 'sensitivity', 'specificity',
                                                           'accuracy', 'fp_rate', 'fn_rate',
                                                           'target', 'model'])

        for target in self.model_specs_collected.keys():

            filename_ = f"confmat_{target}_{self.model_specs_collected[target]}{self.graph_format}"
            cnf_matrix = metrics.confusion_matrix(self._y_tests_collected[target], self._y_preds_collected[target])

            '''
            <true> diagonal
            <false> off-diagonal
            <positive> bottom row
            <negative> top row    
            '''
            tn = cnf_matrix[[0]][0][0]  # top left
            tp = cnf_matrix[[1]][0][1]  # bottom right
            fn = cnf_matrix[[0]][0][1]  # top right
            fp = cnf_matrix[[1]][0][0]  # bottom left

            measures = {"tp": tp,
                        "fn": fn,
                        "fp": fp,
                        "tn": tn,
                        "sensitivity": tp / (tp + fn),
                        "specificity": tn / (tn + fp),
                        "accuracy": (tp + tn) / (tp + tn + fp + fn),
                        "fp_rate": fp / (fp + tn),
                        "fn_rate": fn / (fn + tp)}

            # collect all results and store to df
            cnf_matrix_collected[target] = cnf_matrix
            measures_collected[target] = measures

            # Store measures to df
            df_measures_temp = pd.DataFrame(measures, index=[0])
            df_measures_temp['model'] = self._model_specs_collected[target]
            df_measures_temp['target'] = target
            df_cnf_matrix_measures = df_cnf_matrix_measures.append(df_measures_temp)

            if verbose is True:
                print(target)
                # print(cnf_matrix)
                print(measures)

            if self.plot:
                class_names = [0, 1]  # name of classes
                fig, ax = plt.subplots()
                tick_marks = np.arange(len(class_names))
                plt.xticks(tick_marks, class_names)
                plt.yticks(tick_marks, class_names)
                # create heatmap
                sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
                ax.xaxis.set_label_position("top")
                plt.tight_layout()
                plt.title('Confusion matrix', y=1.1)
                plt.ylabel('Actual label')
                plt.xlabel('Predicted label')
                plt.tight_layout()
                if self.save_plot:
                    plt.savefig(f"{filepath}{filename_}")
                plt.show()

        return df_cnf_matrix_measures

    def get_combined_metric(self, filepath=None, metric='accuracy'):
        """
        this function produces all accuracies/f1-scores for all targets, puts them into a df and plots data
        :param filepath: str, specify filepath where graph will be saved
        :param metric: str, specify metric, 'accuracy' or 'f1'
        :return: df, pd.DataFrame with accuracies, targets and model_specs
        """

        dict_temp = {}
        metric_score_, model_specs_ = [], []
        filename_ = f"{metric}_{self.model_specs_collected[self.targets[0]]}{self.graph_format}"

        model_temp_ = self._model_specs_collected[self.targets[0]].split("_")
        model_spec_ = "_".join([j for i, j in enumerate(model_temp_) if i not in [2, 3]])

        # feature set
        fset_ = "_".join(model_temp_[2:4])

        if metric == 'accuracy':
            for target in self.targets:
                metric_score_.append(metrics.accuracy_score(y_true=self._y_tests_collected[target],
                                                            y_pred=self._y_preds_collected[target]))
                model_specs_.append(self._model_specs_collected[target])
        else:
            for target in self.targets:
                metric_score_.append(metrics.f1_score(y_true=self._y_tests_collected[target],
                                                      y_pred=self._y_preds_collected[target]))
                model_specs_.append(self._model_specs_collected[target])

        dict_temp[f'{metric}_score'] = metric_score_
        dict_temp['target'] = self.targets
        dict_temp['model_specs'] = model_spec_
        dict_temp['feature_set'] = fset_

        df = pd.DataFrame(dict_temp, columns=[f'{metric}_score', 'target', 'model_specs', 'feature_set'])

        if self.plot:
            x_pos = range(len(df['target']))
            targets_ = [t.replace("i_", "") for t in df['target']]
            plt.bar(x_pos, df[f'{metric}_score'], color=color_palette[0])
            # plt.xlabel('behavioral traits')
            plt.ylabel(f'{metric}')
            plt.xticks(x_pos, targets_, rotation=45)
            plt.tight_layout()
            if self.save_plot:
                plt.savefig(f"{filepath}{filename_}")
            plt.show()

        return df

    # @staticmethod
    # def get_feature_importances(model_method='logistic', model_=None, X_test=None,
    #                             plot=True, save_plot=False, filepath=None):
    #
    #     if model_method == 'logistic':
    #         importances = model_.coef_[0]
    #         feat_importances = pd.Series(importances, index=X_test.columns)
    #         if plot:
    #             feat_importances.nlargest(20).plot(kind='barh', title='Feature Importance')
    #             plt.tight_layout()
    #             if save_plot:
    #                 plt.savefig(filepath)
    #             plt.show()
    #     else:
    #         importances = model_.feature_importances_
    #         # std = np.std([tree.feature_importances_ for tree in model_.estimators_], axis=0)
    #         # feat_importances = pd.Series(importances, index=model_.feature_names_in_)
    #         feat_importances = pd.Series(importances, index=[feature_names_nicely[el] for el in model_.feature_names_in_])
    #         if plot:
    #             feat_importances.sort_values().plot(kind='barh',
    #                                                 title='Feature Importance',
    #                                                 figsize=set_figsize(len(model_.feature_importances_))
    #                                                 # ylabel='Mean decreasing in impurity'
    #                                                 )
    #             # plt.rcParams["figure.figsize"] = set_figsize(len(model_.feature_importances_))
    #             plt.tight_layout()
    #             if save_plot:
    #                 plt.savefig(filepath)
    #             plt.show()
    #
    #             # set back to default figsize
    #             plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
    #
    #     return importances, pd.DataFrame(feat_importances, columns=["importance"]).sort_values("importance")

    def roc_curve(self, filepath=None):

        for target in self.model_specs_collected.keys():

            filename_ = f"auc_{target}_{self.model_specs_collected[target]}{self.graph_format}"

            if self.plot:
                auc_plot_train = metrics.plot_roc_curve(estimator=self.models_collected[target],
                                                        X=self._X_train,
                                                        y=self._y_trains_collected[target],
                                                        # sample_weight=model['sample_weight']
                                                        color=color_palette[0])
                auc_plot_test = metrics.plot_roc_curve(estimator=self.models_collected[target],
                                                       X=self._X_test,
                                                       y=self._y_tests_collected[target],
                                                       # sample_weight=model['sample_weight']
                                                       ax=auc_plot_train.ax_,
                                                       color=color_palette[1])
                auc_plot_test.figure_.suptitle("ROC curve")
                plt.tight_layout()
                if self.save_plot:
                    plt.savefig(f"{filepath}/{filename_}")
                plt.show()

        #     y_pred_proba = self._models_collected[target].predict_proba(self.X_test)[::, 1]
        #     auc_score = metrics.roc_auc_score(self._y_tests_collected[target], y_pred_proba)
        #     fpr, tpr, thresholds = metrics.roc_curve(self._y_tests_collected[target],
        #                                              self._y_preds_collected[target], pos_label=2)
        #     auc_score = metrics.auc(fpr, tpr)
        # return fpr, tpr, auc_score

    # @staticmethod
    # def plot_roc_curve(fprs, tprs, plot=True, save_plot=False, filepath=None):
    #     """Plot the Receiver Operating Characteristic from a list
    #     of true positive rates and false positive rates."""
    #
    # # Initialize useful lists + the plot axes.
    # global ax
    # tprs_interp = []
    # aucs = []
    # mean_fpr = np.linspace(0, 1, 100)
    # if plot:
    #     f, ax = plt.subplots(figsize=(14, 10))
    #
    # # Plot ROC for each K-Fold + compute AUC scores.
    # for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
    #     tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
    #     tprs_interp[-1][0] = 0.0
    #     roc_auc = metrics.auc(fpr, tpr)
    #     aucs.append(roc_auc)
    #     if plot:
    #         ax.plot(fpr, tpr, lw=1, alpha=0.3,
    #                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    #
    # if plot:
    #     # Plot the luck line.
    #     plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
    #              label='Luck', alpha=.8)
    #
    # # Plot the mean ROC.
    # mean_tpr = np.mean(tprs_interp, axis=0)
    # mean_tpr[-1] = 1.0
    # mean_auc = metrics.auc(mean_fpr, mean_tpr)
    # std_auc = np.std(aucs)
    # if plot:
    #     ax.plot(mean_fpr, mean_tpr, color='b',
    #             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
    #             lw=2, alpha=.8)
    #
    # # Plot the standard deviation around the mean ROC.
    # std_tpr = np.std(tprs_interp, axis=0)
    # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # if plot:
    #     ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
    #                     label=r'$\pm$ 1 std. dev.')
    #
    #     # Fine tune and show the plot.
    #     ax.set_xlim([-0.05, 1.05])
    #     ax.set_ylim([-0.05, 1.05])
    #     ax.set_xlabel('False Positive Rate')
    #     ax.set_ylabel('True Positive Rate')
    #     ax.set_title('Receiver operating characteristic')
    #     ax.legend(loc="lower right")
    #     plt.tight_layout()
    #     if save_plot:
    #         plt.savefig(filepath)
    #     plt.show()
    # # return (f, ax)
    #
    # @staticmethod
    # def compute_roc_auc(index, model_, X, y, plot=True, save_plot=False, filepath=None) -> (list, list, float):
    #     y_predict = model_.predict_proba(X.iloc[index])[:, 1]
    #     fprs, tprs, thresholds = metrics.roc_curve(y.iloc[index], y_predict)
    #     auc_score = metrics.auc(fprs, tprs)
    #
    #     if plot:
    #         # Initialize useful lists + the plot axes.
    #         global ax
    #         tprs_interp = []
    #         aucs = []
    #         mean_fpr = np.linspace(0, 1, len(fprs))
    #         if plot:
    #             f, ax = plt.subplots(figsize=(14, 10))
    #
    #         # Plot ROC for each K-Fold + compute AUC scores.
    #         for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
    #             print(i, (fpr, tpr))
    #             tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
    #             tprs_interp[-1][0] = 0.0
    #             roc_auc = metrics.auc(fpr, tpr)
    #             aucs.append(roc_auc)
    #             if plot:
    #                 ax.plot(fpr, tpr, lw=1, alpha=0.3,
    #                         label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    #
    #         if plot:
    #             # Plot the luck line.
    #             plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
    #                      label='Luck', alpha=.8)
    #
    #         # Plot the mean ROC.
    #         mean_tpr = np.mean(tprs_interp, axis=0)
    #         mean_tpr[-1] = 1.0
    #         mean_auc = metrics.auc(mean_fpr, mean_tpr)
    #         std_auc = np.std(aucs)
    #         if plot:
    #             ax.plot(mean_fpr, mean_tpr, color='b',
    #                     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
    #                     lw=2, alpha=.8)
    #
    #         # Plot the standard deviation around the mean ROC.
    #         std_tpr = np.std(tprs_interp, axis=0)
    #         tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    #         tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    #         if plot:
    #             ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
    #                             label=r'$\pm$ 1 std. dev.')
    #
    #             # Fine tune and show the plot.
    #             ax.set_xlim([-0.05, 1.05])
    #             ax.set_ylim([-0.05, 1.05])
    #             ax.set_xlabel('False Positive Rate')
    #             ax.set_ylabel('True Positive Rate')
    #             ax.set_title('Receiver operating characteristic')
    #             ax.legend(loc="lower right")
    #             plt.tight_layout()
    #             if save_plot:
    #                 plt.savefig(filepath)
    #             plt.show()
    #         # return (f, ax)
    #
    #     return fpr, tpr, auc_score

    @staticmethod
    def get_auc_scores_random_forest_CV(cv=5, X=None, y=None, clf=None):
        random_forestCV = StratifiedKFold(n_splits=cv, random_state=123, shuffle=True)
        auc_scores = []

        for train_index, test_index in random_forestCV.split(X, y):
            print("TRAIN:", len(train_index), "TEST:", len(test_index))
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model_ = clf.fit(X=X_train, y=y_train.values.ravel())
            _, _, auc_score = clf.compute_roc_auc(X_train.index, model_)
            auc_scores.append(auc_score)

        return auc_scores

    def plot_metric_combined(self, load_from_tables_path=None, filepath=None, metric='accuracy'):
        """
        this function gathers all accuracies of all traits and all feature_sets and plots them altogether.
        :param load_from_tables_path: table_path where table is saved to from get_combined_accuracies()
        :param metric: str, accuracy or f1
        :param filepath: str, path where plot will be saved to
        :return: None
        """

        # select model, drop "set_*_"
        model_temp_ = self._model_specs_collected[self.targets[0]].split("_")
        model_ = "_".join([j for i, j in enumerate(model_temp_) if i not in [2, 3]])
        filename_ = f"combined_{metric}_{model_}{self.graph_format}"

        # read in accuracies and select current model_specs
        df = pd.read_excel(f"{load_from_tables_path}combined_{metric}_scores.xlsx")
        df = df[df['model_specs'] == model_]

        # extract feature sets from model
        df['feature_set'] = df['model'].str.replace('_'.join(model_.split("_")[:2]), '')
        df['feature_set'] = df['feature_set'].str.replace('_'.join(model_.split("_")[2:]), '')
        df['feature_set'] = 'feature' + df['feature_set'].str[:-1]
        df['feature_set'] = df['feature_set'].str.replace("_", " ")
        df['target'] = df['target'].str.replace('i_', '')

        # Draw a nested barplot to show survival for class and sex
        sns.catplot(x="target", y=f"{metric}_score", hue="feature_set", data=df, kind="bar")
        plt.ylabel(f"{metric} score")
        plt.xlabel("")
        plt.xticks(rotation=45)
        plt.legend("", bbox_to_anchor=(1.3, 1.0), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{filepath}{filename_}")
        plt.show()

    def correlations_between_traits(self, filepath=None):
        """
        this function generates an excel sheet with all behavioral traits correlated with each other
        :param filepath: str, specify where excel sheet should be saved to
        :return: None
        """

        y_tests_dict = {}
        for t in self.y_tests_collected.keys():
            y_tests_dict[t] = self.y_tests_collected[t].reset_index()[t].to_list()

        y_trains_dict = {}
        for t in self.y_trains_collected.keys():
            y_trains_dict[t] = self.y_trains_collected[t].reset_index()[t].to_list()

        df_y_tests_collected = pd.DataFrame(y_tests_dict)
        df_y_trains_collected = pd.DataFrame(y_trains_dict)
        df_actual_behavioral_traits = df_y_trains_collected.append(df_y_tests_collected)

        # correlate actual behavioral traits among each other
        # (self.y_trains_collected+self.y_tests_collected)
        df_actual_behavioral_traits_corr = corrmat_with_pval(df_actual_behavioral_traits)

        # correlate predicted behavioral traits among each other (self.y_preds_collected)
        df_y_preds_collected = pd.DataFrame(self._y_preds_collected)
        df_y_preds_collected_corr = corrmat_with_pval(df_y_preds_collected)

        # correlate predicted vs. actual behavioral traits among each other
        # (self.y_preds_collected['i_risk'] vs self.y_tests_collected['i_risk'])

        df_y_preds_collected.columns = ['predicted_' + col for col in df_y_preds_collected.columns]
        df_y_tests_collected.columns = ['actual_' + col for col in df_y_tests_collected.columns]
        df_predicted_vs_actual = pd.merge(df_y_preds_collected, df_y_tests_collected, left_index=True, right_index=True)
        df_predicted_vs_actual_corr = corrmat_with_pval(df_predicted_vs_actual)

        results = {
            "actual_behavioral_traits": df_actual_behavioral_traits_corr,
            "y_preds_collected": df_y_preds_collected_corr,
            "predicted_vs_actual": df_predicted_vs_actual_corr
        }

        """
        Store results to Excel
        """

        # filename
        model_temp_ = self._model_specs_collected[self.targets[0]].split("_")
        model_spec_ = "_".join([j for i, j in enumerate(model_temp_) if i not in [0]])
        filename_ = f"corrmat_{model_spec_}.xlsx"

        # create folder "correlations/"
        if not os.path.exists(filepath + "correlations/"):
            # Create a new directory because it does not exist
            os.makedirs(filepath + "correlations/")

        # if not exists, create empty Excel file
        if not os.path.exists(filepath + "correlations/" + filename_):
            writer = pd.ExcelWriter(filepath + "correlations/" + filename_, engine='xlsxwriter')
            writer.save()
        # loop over each correlation matrix and append to new sheet in Excel file
        for res in results:
            with pd.ExcelWriter(filepath + "correlations/" + filename_,
                                engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                results[res].to_excel(writer, res, index=True)
        print(f"{filename_} stored to {filepath}")
