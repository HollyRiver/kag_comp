from autogluon.multimodal import MultiModalPredictor
from kag_comp import analysis
import pandas as pd

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