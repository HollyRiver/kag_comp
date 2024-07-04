import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics  ## curculate score
from autogluon.tabular import TabularPredictor  ## predictor
# import autogluon.eda.auto as auto  ## eda
import os

exit()
kaggle competitions download -c playground-series-s4e5
unzip playground-series-s4e5.zip -d data
python

df_test = pd.read_csv("~/kag_comp/Flood Prediction/data/test.csv")
df_train = pd.read_csv("~/kag_comp/Flood Prediction/data/train.csv")
df_train.info()  ## non-null, one float, other integers. id not required to analysis
df_train.nunique()``
df_train.FloodProbability.value_counts()  ## units : 0.05

predictr = TabularPredictor(label = "FloodProbability")
predictr.fit(df_train)  ## why don't working...

## 로지스틱으로 접근한 후에 확률값을 반환하는 편이 더 좋아보이긴 함.

auto.quick_fit(
    train_data = df_train,
    label = "FloodProbability",
)