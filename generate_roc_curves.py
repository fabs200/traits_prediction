from matplotlib import pyplot as plt
from model_evaluation import model_evaluation
from config import feature_sets
from config import model
from utils import extract_model_specification
from load_data import df_crypto_prep
from itertools import cycle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

fprs, tprs, auc_scores = [], [], []

if __name__ == "__main__":

    for i_, feat_set_ in enumerate(feature_sets):

        print(i_, feat_set_, "\n\n", "#" * 40, "\n\n")
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


        fpr, tpr, auc_score = model_evaluation.get_auc_score(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            y_pred=y_pred,
            model_=model_,
            plot=False, save_plot=False
        )

        fprs.append(fpr), tprs.append(tpr), auc_scores.append(auc_score)

colors = cycle(["aqua", "darkorange", "cornflowerblue", 'c', 'm', 'y', 'k', "grey", "orange", "pink", "red", "brown"])
for i, color in enumerate(colors):
    plt.plot(
        fprs[i][1],
        tprs[i][1],
        color=color,
        lw=2,
        label="ROC curve of class {0} (area = {1:0.2f})".format(i, auc_scores[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Some extension of Receiver operating characteristic to multiclass")
plt.legend(loc="lower right")
plt.show()
