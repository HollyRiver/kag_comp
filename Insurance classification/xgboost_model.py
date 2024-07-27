from kag_comp import analysis
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

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

## predictor, cross validation

predictr = xgb.XGBClassifier(tree_method = 'gpu_hist', gpu_id = 0)
predictr.fit(X, y)

help(xgb.XGBClassifier)

params = {"max_depth" : [10+i*5 for i in range(11)],
          "min_child_weight" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ## 관측치 가중합의 최소
          "gamma" : [i/10.0 for i in range(7)], ## 리프 노드의 추가 분할을 결정할 최소손실 감소값
          "n_estimators" : [100, 150, 200, 300],
          "colsample_bytree" : [0.5 + i*0.05 for i in range(8)]} ## features sampling

param_depth = {"max_depth" : [10+i*5 for i in range(11)]}
param_child = {"min_child_weight" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
param_gamma = {"gamma" : [i/10.0 for i in range(7)]}
param_estim = {"n_estimators" : [100, 150, 200, 300]}
param_colsample = {"colsample_bytree" : [0.5 + i*0.05 for i in range(8)]}

gscv_depth = GridSearchCV(estimator = predictr, param_grid = param_depth, cv = StratifiedKFold(n_splits = 5, shuffle = True), refit = True, scoring = "roc_auc")
gscv_child = GridSearchCV(estimator = predictr, param_grid = param_child, cv = StratifiedKFold(n_splits = 5, shuffle = True), refit = True, scoring = "roc_auc")
gscv_gamma = GridSearchCV(estimator = predictr, param_grid = param_gamma, cv = StratifiedKFold(n_splits = 5, shuffle = True), refit = True, scoring = "roc_auc")
gscv_estim = GridSearchCV(estimator = predictr, param_grid = param_estim, cv = StratifiedKFold(n_splits = 5, shuffle = True), refit = True, scoring = "roc_auc")
gscv_colsample = GridSearchCV(estimator = predictr, param_grid = param_colsample, cv = StratifiedKFold(n_splits = 5, shuffle = True), refit = True, scoring = "roc_auc")

gscv_depth.fit(X, y)
gscv_child.fit(X, y)
gscv_gamma.fit(X, y)
gscv_estim.fit(X, y)
gscv_colsample.fit(X, y)

