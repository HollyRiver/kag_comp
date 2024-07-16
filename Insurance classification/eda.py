# from matplotlib import pyplot as plt
# import numpy as np

# a = np.array([1, 2, 3, 4])
# plt.hist(a)
# plt.show()
# plt.savefig('test.png')

from autogluon.tabular import TabularDataset
from autogluon.tabular import TabularPredictor
import plotly.express as px
import plotly.io as pio



import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PowerTransformer

import os
import analysis

load = analysis.file_load("kaggle competitions download -c playground-series-s4e7")
print(load)

df_train, df_test, df_submit = load.load_tr_tst_sub_data()

## 드디어 됐다.

## using pipeline

Pipee = Pipeline([('Feature_Selection', SelectKBest(f_classif, k = 2)),  ## variable selection
                  ('Standardization', PowerTransformer())  ## Standarization
])

X = pd.get_dummies(df_train.drop("Response", axis = 1), drop_first = True)
y = pd.get_dummies(df_train.Response, drop_first = True)

selectr = SelectKBest(f_classif, k = 8)
selectr.fit(X, y)
selectr.get_feature_names_out()

featured_X = X.loc[:, selectr.get_feature_names_out()]
featured_X.head()

transfr = PowerTransformer()
transfr.fit(featured_X, y)
temp = transfr.transform(featured_X)
temp2 = pd.DataFrame(temp, columns = selectr.get_feature_names_out())
X.loc[:5, selectr.get_feature_names_out()]
temp2[["Previously_Insured", "Gender_Male", "Vehicle_Age_< 1 Year", "Vehicle_Age_> 2 Years", "Vehicle_Damage_Yes"]]\
.map(lambda x : True if x > 0 else False)

featured_X = pd.concat([temp2[["Age", "Annual_Premium", "Policy_Sales_Channel"]], temp2[["Previously_Insured", "Gender_Male", "Vehicle_Age_< 1 Year", "Vehicle_Age_> 2 Years", "Vehicle_Damage_Yes"]]\
.map(lambda x : True if x > 0 else False)], axis = 1)

featured_X.head()
y

from sklearn.ensemble import GradientBoostingClassifier

predictr = GradientBoostingClassifier()
predictr.fit(featured_X, y)

tst_temp = transfr.transform(pd.get_dummies(df_test, drop_first = True)[selectr.get_feature_names_out()])
pd.DataFrame(tst_temp, columns = selectr.get_feature_names_out())

featured_tst_X = pd.concat([pd.DataFrame(tst_temp, columns = selectr.get_feature_names_out())[["Age", "Annual_Premium", "Policy_Sales_Channel"]],
           pd.DataFrame(tst_temp, columns = selectr.get_feature_names_out())[["Previously_Insured", "Gender_Male", "Vehicle_Age_< 1 Year", "Vehicle_Age_> 2 Years", "Vehicle_Damage_Yes"]].map(lambda x : True if x > 0 else False)], axis = 1)

yyhat = predictr.predict(featured_tst_X)

yyhat

submission = df_test[["id"]].assign(Response = yyhat)

submission.Response = submission.Response.astype(int)
submission


submission.to_csv("submission.csv", index = False)

os.system(f"kaggle competitions submit -c {load.competition} -f submission.csv -m .")

os.system(f"rm submission.csv")