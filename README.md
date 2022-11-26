# infer behavioral traits from psd2 data
infer behavioral traits from psd2 data

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

