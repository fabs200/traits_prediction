import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import os.path
from utils import set_figsize
from features_sets import feature_names_nicely

class model_evaluation:
    def __int__(self, X_train=None, X_test=None, y_train=None, y_test=None, y_pred=None,
                model_=None, model_method='logistic',
                plot=True, save_plot=False, filepath=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = y_pred
        self.model_ = model_
        self.model_method = model_method
        self.plot = plot
        self.save_plot = save_plot
        self.filepath = filepath

    @staticmethod
    def get_confusion_matrix(y_test=None, y_pred=None, plot=True, save_plot=False, filepath=None, verbose=False):
        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

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
                    "sensitivity": tp/(tp+fn),
                    "specificity": tn/(tn+fp),
                    "accuracy": (tp+tn)/(tp+tn+fp+fn),
                    "fp_rate": fp/(fp+tn),
                    "fn_rate": fn/(fn+tp)}

        if verbose is True:
            print(cnf_matrix)
            print(measures)

        if plot:
            class_names = [0, 1]  # name  of classes
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
            if save_plot:
                plt.savefig(filepath)
            plt.show()

        return cnf_matrix, measures

    @staticmethod
    def get_predicted_probabilities(model_=None, X_test=None, y_test=None, plot=True, save_plot=False, filepath=None):
        y_pred_proba = model_.predict_proba(X_test)[::, 1]
        auc = metrics.roc_auc_score(y_test, y_pred_proba)

        print('auc:', auc)

        if plot:
            fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label="data 1, auc=" + str(round(auc, 3)))
            plt.legend(loc=4)
            if save_plot:
                plt.savefig(filepath)
            plt.show()

        return y_pred_proba, auc

    @staticmethod
    def get_feature_importances(model_method='logistic', model_=None, X_test=None,
                                plot=True, save_plot=False, filepath=None):

        if model_method == 'logistic':
            importances = model_.coef_[0]
            feat_importances = pd.Series(importances, index=X_test.columns)
            if plot:
                feat_importances.nlargest(20).plot(kind='barh', title='Feature Importance')
                plt.tight_layout()
                if save_plot:
                    plt.savefig(filepath)
                plt.show()
        else:
            importances = model_.feature_importances_
            # std = np.std([tree.feature_importances_ for tree in model_.estimators_], axis=0)
            # feat_importances = pd.Series(importances, index=model_.feature_names_in_)
            feat_importances = pd.Series(importances, index=[feature_names_nicely[el] for el in model_.feature_names_in_])
            if plot:
                feat_importances.sort_values().plot(kind='barh',
                                                    title='Feature Importance',
                                                    figsize=set_figsize(len(model_.feature_importances_))
                                                    # ylabel='Mean decrease in impurity'
                                                    )
                # plt.rcParams["figure.figsize"] = set_figsize(len(model_.feature_importances_))
                plt.tight_layout()
                if save_plot:
                    plt.savefig(filepath)
                plt.show()

                # set back to default figsize
                plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]

        return importances, pd.DataFrame(feat_importances, columns=["importance"]).sort_values("importance")

    @staticmethod
    def get_f1_score(y_test, y_pred, average='macro', verbose=False):
        f1_score = metrics.f1_score(y_test, y_pred, average=average)
        if verbose:
            print(f1_score)
        return f1_score

    @staticmethod
    def get_accuracy(y_test, y_pred, verbose=False):
        # same as score
        accuracy = metrics.accuracy_score(y_test, y_pred)
        if verbose:
            print(accuracy)
        return accuracy

    @staticmethod
    def get_auc_score(X_train=None, y_train=None, X_test=None, y_test=None, y_pred=None, model_=None,
                      plot=True, save_plot=False, filepath=None):

        if plot:
            auc_plot_train = metrics.plot_roc_curve(model_, X_train, y_train)
            auc_plot_test = metrics.plot_roc_curve(model_, X_test, y_test, ax=auc_plot_train.ax_)
            auc_plot_test.figure_.suptitle("ROC curve")
            plt.tight_layout()
            if save_plot:
                plt.savefig(filepath)
            plt.show()

        y_pred_proba = model_.predict_proba(X_test)[::, 1]
        auc_score = metrics.roc_auc_score(y_test, y_pred_proba)

        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=2)
        # auc_score = metrics.auc(fpr, tpr)

        return fpr, tpr, auc_score

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
            _, _, auc_score = model_evaluation.compute_roc_auc(X_train.index, model_)
            auc_scores.append(auc_score)

        return auc_scores


def store_criterions_results(filename=None, critereon_results=None):
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            for key in critereon_results.keys():
                f.write(f"{key}, ")
            f.write("\n")

    with open(filename, 'a') as f:
        for key in critereon_results.keys():
            f.write(f"{critereon_results[key]}, ")
        f.write("\n")

