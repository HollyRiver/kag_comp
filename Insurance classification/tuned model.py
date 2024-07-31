from kag_comp import analysis
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

## full data loading

load = analysis.file_load("kaggle competitions download -c playground-series-s4e7")
df_train, df_test, df_submission = load.load_tr_tst_sub_data()

df_train2 = pd.read_csv("/root/kag_comp/Insurance classification/train.csv")
sum_train = pd.concat([df_train, df_train2], axis = 0)

## data preprocessing

train_data = pd.get_dummies(sum_train.drop(["id"], axis = 1).drop_duplicates().assign(Region_Code = lambda _df : _df.Region_Code.astype(int).astype(str)), drop_first = True)\
    .rename({'Vehicle_Age_< 1 Year' : "Vehicle_Age_1_Under", "Vehicle_Age_> 2 Years" : "Vehicle_Age_2_Under"}, axis = 1)
    
X = train_data.drop("Response", axis = 1)
y = train_data.Response
XX = pd.get_dummies(df_test.drop(["id"], axis = 1).assign(Region_Code = lambda _df : _df.Region_Code.astype(int).astype(str)), drop_first = True)\
    .rename({'Vehicle_Age_< 1 Year' : "Vehicle_Age_1_Under", "Vehicle_Age_> 2 Years" : "Vehicle_Age_2_Under"}, axis = 1)
    
## hyperparameter tuning

param_dist = {
    "gamma" : [0.01*1 for i in range(100)],
    "colsample_bytree" : [0.01*i for i in range(100)]
}

predictr = xgb.XGBClassifier(tree_method = "hist", device = "cuda", n_estimators = 121, min_child_weight = 46, max_depth = 40)
rscv = RandomizedSearchCV(estimator = predictr, param_distributions = param_dist, cv = StratifiedKFold(n_splits = 5, shuffle = True), refit = True, scoring = "roc_auc")
rscv.fit(X, y)

param_dist2 = {
    "n_estimators" : [50+5*i for i in range(40)],
    "min_child_weight" : [25+i for i in range(40)],
    "max_depth" : [30+i for i in range(40)]
}

predictr2 = rscv.best_estimator_
rscv2 = RandomizedSearchCV(estimator = predictr2, param_distributions = param_dist2, cv = StratifiedKFold(n_splits = 5, shuffle = True), refit = True, scoring = "roc_auc")
rscv2.fit(X, y)

yyhat = rscv2.best_estimator_.predict(XX)
submission = df_test[["id"]].assign(Response = yyhat)
load.submit_file(submission)

rscv2.best_estimator_.get_params()

final_predictr = xgb.XGBClassifier(tree_method = "hist", device = "cuda", max_depth = 0, gamma = "0.01", colsample_bytree = 0.19, n_estimators = 175, eval_metric = "auc", min_child_weight = 59)
final_predictr.fit(X, y)

yyhat = final_predictr.predict(XX)
submission = df_test[["id"]].assign(Response = yyhat)
load.submit_file(submission)