# Infering behavioral traits from PSD2-Data

This project uses PSD2 data and infers behavioral traits of bank customers.
PSD2 data includes data such as financial transaction, demographic and other meta data that is readily available to financial institutions due to PSD2 regulation.
The objective of this project is to demonstrate how financial institutions can make use of their data to generate new insights of their customers.

# Procedure
* We specify different feature sets that include different variables. These feature sets vary in data availability, level of detail, vertical and horizontal information on customers.
* We split the sample into train and test datasets
* We span a large grid search of random forest parameters with which we train our model with which we estimate each single behavioral trait.
* We select the best parameter set according to different evaluation measures such as accuracy, f1 score or auc-roc score.
* Based on the best parameter set, we then predict the test data and evaluate the goodness of fit.
* Finally, we repeat the steps for each behavioral trait with all different feature sets and report accuracies, parameters, etc. in tables and plots.

### Model Selection
1. best parameters:
  - ts0.3
  - mss0.1
  - md9 (or md10)
  - msl1
  - mfsqrt
  - ne400
  - GSFalse
  - wbalanced

2. Grid searches:
   - first round:
     - bootstrap: [False]
     - max_depth: [5, 10, 15, 30, 60, 120]
     - max_features: ["sqrt", "log2", None]
     - min_samples_leaf: [1]
     - min-sample_split: [2]
     - n_estimators: [200, 300, 400, 500]
     - best model majority: md10, mfsqrt, ne400 
   - second round: 
     - bootstrap: [False]
     - max_depth: [5, 7, 10, 12]
     - max_features: ["sqrt"]
     - min_samples_leaf: [1]
     - min-sample_split: [2]
     - n_estimators: [200, 300, 400]
     - best model majority: md10, ne400

