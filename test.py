import analysis
import pandas as pd
from autogluon.tabular import TabularPredictor
# import autogluon.eda.auto as auto

## kaggle download instance
kag = analysis.file_load("kaggle competitions download -c playground-series-s4e5")  ## no specific directory
print(kag)

df_train, df_test, df_submit = kag.load_data()
print(df_train.columns)

predictr = TabularPredictor(label = "FloodProbability")
predictr.fit(df_train, ag_args_fit={'num_gpus': 1})
yhat = predictr.predict(df_train)

predictr.leaderboard(silent = True)

# pip uninstall lightgbm -y       pip install lightgbm --install-option=--gpu