from autogluon.multimodal import MultiModalPredictor
from autogluon.tabular import TabularPredictor
from kag_comp import analysis
import pandas as pd
import sklearn.feature_selection

load = analysis.file_load("kaggle competitions download -c playground-series-s4e7")

df_train, df_test, df_submission = load.load_tr_tst_sub_data()

predictr = MultiModalPredictor(label = "Response", eval_metric = 'acc')
## 순서대로 target variable name, loss function(score) method
## eval metric : acc(Accuracy)

predictr.fit(df_train, time_limit = 7200)

train_score = predictr.evaluate(df_train, metrics = ['acc', 'f1'])
print(train_score)

yhat = predictr.predict(df_train)

yyhat = predictr.predict(df_test)
submission = df_test[["id"]].assign(Response = yyhat)
load.submit_file(submission)


## formal predictor

df_train.info()
df_train.drop(["id"], axis = 1)
pd.get_dummies(df_train, drop_first = True).drop(["id"], axis = 1)\
    .assign(Region_Code = lambda _df : _df.Region_Code.astype(int).astype(str))
    
X = pd.get_dummies(df_train.drop(["id"], axis = 1).assign(Region_Code = lambda _df : _df.Region_Code.astype(int).astype(str)), drop_first = True)

## sklearn.feature_selection

predictr = TabularPredictor(label = "Response", problem_type = "binary")
predictr.fit(df_train)

predictr = TabularPredictor.load("/root/AutogluonModels/ag-20240717_085901")
predictr.leaderboard()

predictr.predict(df_train)

yyhat = predictr.predict(df_test)
submission = df_test[["id"]].assign(Response = yyhat)
load.submit_file(submission)

yyhat.value_counts()
df_train.Response.value_counts()

## 개망함