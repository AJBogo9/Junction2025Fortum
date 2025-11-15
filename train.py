import pandas as pd
from lightgbm import LGBMRegressor
import joblib

# load data
X_train = pd.read_csv('data/first/X_train.csv')
y_train = pd.read_csv('data/first/y_train.csv')

# train model
model = LGBMRegressor(objective='regression', n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train,
          categorical_feature=['hour', 'day_of_week', 'is_weekend'])

# save model
joblib.dump(model, "data/first/lgbm_model.pkl")