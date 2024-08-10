from kag_comp import analysis
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import gc

## data load

load = analysis.file_load("kaggle competitions download -c playground-series-s4e8")
print(load)

df_train, df_test, df_submission = load.load_tr_tst_sub_data()

df_train.info()
df_train.nunique()

X = df_train.drop(["id", "class"], axis = 1).drop(["stem-root", "veil-type", "veil-color", "spore-print-color"], axis = 1)
X.nunique()
X.info()
X.isnull().sum()
y = pd.get_dummies(df_train["class"], drop_first = True)

predictr = xgb.XGBClassifier(max_depth = 50)
predictr.fit(X, y)

del predictr
del X
del y
gc.collect()

## 이게 말이 되나

df_train.columns
df_train["cap-diameter"].describe()
df_train[["cap-diameter", "stem-height", "stem-width"]]

## step 1 : Normalization

from sklearn.preprocessing import PowerTransformer

transpr = PowerTransformer()
transpr.fit(df_train[["cap-diameter", "stem-height", "stem-width"]])
transd_subdf = pd.DataFrame(transpr.transform(df_train[["cap-diameter", "stem-height", "stem-width"]]), columns = ["cap-diameter", "stem-height", "stem-width"])

## step 2 : Converting objective value -> Delete

temp = df_train.drop(["id", "class", "cap-diameter", "stem-height", "stem-width"], axis = 1)

print(temp[["cap-shape"]].groupby("cap-shape").agg({"cap-shape" : "count"}).to_markdown()) # b, c, x, f, k, s

print(temp[["cap-shape"]].map(lambda x : str(x).split()[-1] if str(x).split()[-1] >= "a" else pd.NA).groupby("cap-shape").agg({"cap-shape" : "count"}).to_markdown())

featured_cap = temp[["cap-shape"]].map(lambda x : str(x).split()[-1] if str(x).split()[-1] >= "a" else pd.NA)
featured_cap.groupby("cap-shape").agg({"cap-shape" : "count"}) # 많음 : b, c, x, f, s, 적음 : k, 뜬금없는 값 : o, p

featured_cap.map(lambda x : x if not pd.isna(x) and x in ["b", "c", "x", "f", "k", "s", "o", "p"] else pd.NA).groupby("cap-shape").agg({"cap-shape" : "count"}).assign()



# Index(['cap-shape', 'cap-surface', 'cap-color', 'does-bruise-or-bleed',
#        'gill-attachment', 'gill-spacing', 'gill-color', 'stem-root',
#        'stem-surface', 'stem-color', 'veil-type', 'veil-color', 'has-ring',
#        'ring-type', 'spore-print-color', 'habitat', 'season'],
#       dtype='object')

df_train.nunique()
df_test.nunique()