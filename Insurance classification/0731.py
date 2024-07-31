from kag_comp import analysis
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import gc

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
    
del train_data
    
    
## Region_code for integer(Region_Code를 분리하지 않고 정수형 그대로 인식)

train_data2 = pd.get_dummies(sum_train.drop(["id"], axis = 1).drop_duplicates().assign(Region_Code = lambda _df : _df.Region_Code.astype(int)), drop_first = True)\
    .rename({'Vehicle_Age_< 1 Year' : "Vehicle_Age_1_Under", "Vehicle_Age_> 2 Years" : "Vehicle_Age_2_Under"}, axis = 1)
    
X = train_data2.drop("Response", axis = 1)
y = train_data2.Response
XX = pd.get_dummies(df_test.drop(["id"], axis = 1).assign(Region_Code = lambda _df : _df.Region_Code.astype(int)), drop_first = True)\
    .rename({'Vehicle_Age_< 1 Year' : "Vehicle_Age_1_Under", "Vehicle_Age_> 2 Years" : "Vehicle_Age_2_Under"}, axis = 1)
    
predictr = xgb.XGBClassifier(tree_method = "hist", device = "cuda", max_depth = 40)  ## 0.60388
predictr.fit(X, y)
predictr

yyhat = predictr.predict(XX)
submission = df_test[["id"]].assign(Response = yyhat)
load.submit_file(submission)

param_dict = {
    "max_depth" : [30+2*i for i in range(10)]
}

predictr = xgb.XGBClassifier(tree_method = "hist", device = "cuda")

tunr = GridSearchCV(predictr, param_grid = param_dict, cv = StratifiedKFold(n_splits = 4, shuffle = True), scoring = "roc_auc")
tunr.fit(X, y)

yyhat = tunr.best_estimator_.predict(XX) ## max_depth = 30, 0.59692
submission = df_test[["id"]].assign(Response = yyhat)
load.submit_file(submission)

param_dict2 = {
    "max_depth" : [25+i for i in range(7)]
}

tunr2 = GridSearchCV(predictr, param_grid = param_dict2, cv = StratifiedKFold(n_splits = 4, shuffle = True), scoring = "roc_auc")
tunr2.fit(X, y)

yyhat = tunr2.best_estimator_.predict(XX) ## max_depth = 30, 0.59692
submission = df_test[["id"]].assign(Response = yyhat)
load.submit_file(submission)

param_dict3 = {
    "max_depth" : [19+i for i in range(7)]
}

predictr = xgb.XGBClassifier(tree_method = "hist", device = "cuda", max_depth = 45)
predictr.fit(X, y)

predictr.score(X, y)

yyhat = predictr.predict(XX) ## max_depth = 45, 0.60481
submission = df_test[["id"]].assign(Response = yyhat)
load.submit_file(submission)

predictr = xgb.XGBClassifier(tree_method = "hist", device = "cuda", max_depth = 45, eval_metric = "auc")
predictr.fit(X, y)

predictr.score(X, y)

yyhat = predictr.predict(XX) ## max_depth = 45, eval_metric = "auc"
submission = df_test[["id"]].assign(Response = yyhat)
load.submit_file(submission)

## another parameters

param_dict3 = {
    "colsample_bytree" : [1.0 - 0.1*i for i in range(6)]
}

tunr3 = GridSearchCV(predictr, param_grid = param_dict3, cv = StratifiedKFold(n_splits = 4, shuffle = True), scoring = "roc_auc")
tunr3.fit(X, y)

yyhat = tunr3.best_estimator_.predict(XX) ## map_depth = 45, eval_metric = "auc", colsample_bytree = 
submission = df_test[["id"]].assign(Response = yyhat)
load.submit_file(submission)