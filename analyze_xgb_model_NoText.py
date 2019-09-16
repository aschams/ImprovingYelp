import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from itertools import compress
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import xgboost as xgb
from scipy import sparse
import pickle

data = pd.read_csv("/users/aschams/scratch/Complete_reviews_barebones_noTFIDF.csv")
stars = pd.read_csv("/users/aschams/scratch/Complete_reviews_stars.csv", header = None)
stars = stars.iloc[:, 1]
X_train, X_test, y_train, y_test = train_test_split(data, stars, test_size = 0.4, random_state = 70)


xgb_rmse = xgb.Booster({'nthread': 1})
xgb_mae = xgb.Booster({'nthread': 1})
model_rmse = xgb_rmse.load_model("/users/aschams/scratch/basic_rmse_model.model")
mode_mae = xgb_mae.load_model("/users/aschams/scratch/basic_mae_model.model")

rmse_preds = model_rmse.predict(X_test)
mae_preds = model_rmse.predict(X_test)

print("RMSE model Error: " + str( mean_squared_error(y_test, rmse_preds) ) )
print("MAE Model Error:" + str( mean_absolute_error( y_test, mae_preds) ) )
