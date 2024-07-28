from kag_comp import analysis
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

## data load
load = analysis.file_load("kaggle competitions download -c playground-series-s4e7")
df_train, df_test, df_submission = load.load_tr_tst_sub_data()

## preprocessing

train_data = pd.get_dummies(df_train.drop(["id"], axis = 1).assign(Region_Code = lambda _df : _df.Region_Code.astype(int).astype(str)), drop_first = True)\
    .rename({'Vehicle_Age_< 1 Year' : "Vehicle_Age_1_Under", "Vehicle_Age_> 2 Years" : "Vehicle_Age_2_Under"}, axis = 1)
X = train_data.drop("Response", axis = 1)
y = train_data.Response

XX = pd.get_dummies(df_test.drop(["id"], axis = 1).assign(Region_Code = lambda _df : _df.Region_Code.astype(int).astype(str)), drop_first = True)\
    .rename({'Vehicle_Age_< 1 Year' : "Vehicle_Age_1_Under", "Vehicle_Age_> 2 Years" : "Vehicle_Age_2_Under"}, axis = 1)

## create predictor and fitting model

predictr = xgb.XGBClassifier()
predictr.fit(X, y)

yy_hat = predictr.predict(XX)

submission = df_test[["id"]].assign(Response = yy_hat)
load.submit_file(submission)

##--------------------------------------------------

## add data
df_train2 = pd.read_csv("/root/kag_comp/Insurance classification/train.csv")
sum_train = pd.concat([df_train, df_train2], axis = 0)

train_data = pd.get_dummies(sum_train.drop(["id"], axis = 1).assign(Region_Code = lambda _df : _df.Region_Code.astype(int).astype(str)), drop_first = True)\
    .rename({'Vehicle_Age_< 1 Year' : "Vehicle_Age_1_Under", "Vehicle_Age_> 2 Years" : "Vehicle_Age_2_Under"}, axis = 1)
    
X = train_data.drop("Response", axis = 1)
y = train_data.Response

XX = pd.get_dummies(df_test.drop(["id"], axis = 1).assign(Region_Code = lambda _df : _df.Region_Code.astype(int).astype(str)), drop_first = True)\
    .rename({'Vehicle_Age_< 1 Year' : "Vehicle_Age_1_Under", "Vehicle_Age_> 2 Years" : "Vehicle_Age_2_Under"}, axis = 1)

## new predictor

predictr_add = xgb.XGBClassifier()
predictr.fit(X, y)

yy_hat = predictr.predict(XX)

submission = df_test[["id"]].assign(Response = yy_hat)
load.submit_file(submission)

##-------------------------------------------------데이터 추가에 따른 이점 없었음. 그럼에도 불구하고 추가하는 것을 추천하긴 한다.

## full data loading

load = analysis.file_load("kaggle competitions download -c playground-series-s4e7")
df_train, df_test, df_submission = load.load_tr_tst_sub_data()

df_train2 = pd.read_csv("/root/kag_comp/Insurance classification/train.csv")
sum_train = pd.concat([df_train, df_train2], axis = 0)

## data preprocessing

train_data = pd.get_dummies(sum_train.drop(["id"], axis = 1).assign(Region_Code = lambda _df : _df.Region_Code.astype(int).astype(str)), drop_first = True)\
    .rename({'Vehicle_Age_< 1 Year' : "Vehicle_Age_1_Under", "Vehicle_Age_> 2 Years" : "Vehicle_Age_2_Under"}, axis = 1)
    
X = train_data.drop("Response", axis = 1)
y = train_data.Response
XX = pd.get_dummies(df_test.drop(["id"], axis = 1).assign(Region_Code = lambda _df : _df.Region_Code.astype(int).astype(str)), drop_first = True)\
    .rename({'Vehicle_Age_< 1 Year' : "Vehicle_Age_1_Under", "Vehicle_Age_> 2 Years" : "Vehicle_Age_2_Under"}, axis = 1)

## predictor, cross validation (hyperparameter tuning)

predictr = xgb.XGBClassifier(tree_method = 'hist', device = "cuda", gamma = 0.1, colsample_bytree = 0.7, eval_metric = "auc")
predictr.fit(X, y)

yy_hat = predictr.predict(XX)

submission = df_test[["id"]].assign(Response = yy_hat)
load.submit_file(submission)

params = {"max_depth" : [10+i*5 for i in range(11)],
          "min_child_weight" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ## 관측치 가중합의 최소
          "gamma" : [i/10.0 for i in range(7)], ## 리프 노드의 추가 분할을 결정할 최소손실 감소값
          "n_estimators" : [100, 150, 200, 300],
          "colsample_bytree" : [0.5 + i*0.05 for i in range(8)]} ## features sampling

param_depth = {"max_depth" : [0]+[30+i*5 for i in range(8)]}
param_child = {"min_child_weight" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
param_gamma = {"gamma" : [i/10.0 for i in range(7)]}
param_estim = {"n_estimators" : [100, 150, 200, 300]}
param_colsample = {"colsample_bytree" : [0.5 + i*0.05 for i in range(8)]}

# gscv_depth = GridSearchCV(estimator = predictr, param_grid = param_depth, cv = StratifiedKFold(n_splits = 5, shuffle = True), refit = True, scoring = "roc_auc")
gscv_depth = GridSearchCV(estimator = predictr, param_grid = param_depth, cv = 3, refit = True, scoring = "roc_auc")
gscv_child = GridSearchCV(estimator = predictr, param_grid = param_child, cv = StratifiedKFold(n_splits = 5, shuffle = True), refit = True, scoring = "roc_auc")
gscv_gamma = GridSearchCV(estimator = predictr, param_grid = param_gamma, cv = StratifiedKFold(n_splits = 5, shuffle = True), refit = True, scoring = "roc_auc")
gscv_estim = GridSearchCV(estimator = predictr, param_grid = param_estim, cv = StratifiedKFold(n_splits = 5, shuffle = True), refit = True, scoring = "roc_auc")
gscv_colsample = GridSearchCV(estimator = predictr, param_grid = param_colsample, cv = StratifiedKFold(n_splits = 5, shuffle = True), refit = True, scoring = "roc_auc")

import cupy as cp
import numpy as np

gscv_depth.fit(cupy.array(X), y)
gscv_child.fit(X, y)
gscv_gamma.fit(X, y)
gscv_estim.fit(X, y)
gscv_colsample.fit(X, y)

GridSearchCV(estimator = predictr, scoring = "roc_auc")

df_arr = np.array(X)
cp.array(df_arr)

df_arr

params = {"device" : "cuda", "tree_method" : "hist", "gamma" : 0.1, "colsample_bytree" : 0.7, "eval_metric" : "auc"}
Xy = xgb.QuantileDMatrix(X, y)
modl = xgb.train(params, Xy)

predictr = xgb.XGBClassifier(tree_method = 'hist', device = "cuda", gamma = 0.1, colsample_bytree = 0.7, eval_metric = "auc", objective = "binary:logistic")

predictr

gscv_depth = GridSearchCV(estimator = predictr, param_grid = param_depth, cv = 3, refit = True, scoring = "roc_auc")

dtrain = xgb.DMatrix(X, y)
dtest = xgb.DMatrix(XX)

gscv_depth.fit(X, y)

from scipy.stats import randint

param_distribs = {
    'n_estimators' : randint(low=1, high=200),
    'max_depth' : randint(low=3, high=100),
    'min_child_weight' : randint(low=1, high=100),
    # 'learning_rate' : randint(low=0.01, high=0.1),
}

rscv_depth = RandomizedSearchCV(estimator = predictr, param_distributions = param_distribs, cv = 3, scoring = "roc_auc")
rscv_depth.fit(X, y)

rscv_depth.best_estimator_

# XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=0.7, device='cuda', early_stopping_rounds=None,
#               enable_categorical=False, eval_metric='auc', feature_types=None,
#               gamma=0.1, grow_policy=None, importance_type=None,
#               interaction_constraints=None, learning_rate=None, max_bin=None,
#               max_cat_threshold=None, max_cat_to_onehot=None,
#               max_delta_step=None, max_depth=22, max_leaves=None,
#               min_child_weight=46, missing=nan, monotone_constraints=None,
#               multi_strategy=None, n_estimators=121, n_jobs=None,
#               num_parallel_tree=None, predictor='gpu_predictor', ...)

randint(low = 0, high = 100)

param_distibs2 = {
    "gamma" : [0.01*i for i in range(100)],
    "colsample_bytree" : [0.01*i for i in range(100)]
}


best_predictr = xgb.XGBClassifier(tree_method = 'hist', device = "cuda", n_estimators = 121, min_child_weight = 46, max_depth = 22)

rscv_another = RandomizedSearchCV(estimator = best_predictr, param_distributions = param_distibs2, cv = 3, scoring = "roc_auc")
rscv_another.fit(X, y)

predictr_final = rscv_another.best_estimator_
yyhat = predictr_final.predict(XX)

submission = df_test[["id"]].assign(Response = yyhat)
load.submit_file(submission)

predictr_final