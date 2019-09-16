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
data.drop(['stars', 'Unnamed: 0', 'latitude', 'longitude'], axis = 1, inplace = True)
data.rename({'stars.1': 'avg_stars'}, axis=1, inplace =True)
stars = pd.read_csv("/users/aschams/scratch/Complete_reviews_stars.csv", header = None)
stars = stars.iloc[:, 1].values.reshape(-1,1)
stars = stars - 1
X_train, X_test, y_train, y_test = train_test_split(data, stars, test_size = 0.4, random_state = 70)

dtrain = xgb.DMatrix(X_train, label=y_train)
# dtrain.save_binary("scratch/training_set.buffer")
dtest = xgb.DMatrix(X_test, label = y_test)
# dtest.save_binary("scratch/test_set.buffer")

evallist = [(dtest, 'eval'), (dtrain, 'train')]
param_lin = {'max_depth': 10, 'eta': .3, 'silent': 1, 'objective': 'reg:linear', 'eval_metric' :['rmse', 'mae']}
param_log = {'max_depth': 10, 'eta': .3, 'silent': 1, 'objective': 'multi:softmax', 'num_class' : 5, 'eval_metric' : ['merror']}

print (y_train.shape)
print(y_test.shape)
print(X_train.shape)

print("Linear Regression Model")
model_lin = xgb.train(param_lin, dtrain, 20, evallist)
print("---------------")
print("Logistic Regression Model")
model_log = xgb.train(param_log, dtrain, 20, evallist)

# model_lin.save_model( "scratch/model_lin.model")
# model_lin.dump_model('scratch/model_lin_raw.txt')
# model_log.save_model("scratch/model_log.model")
# model_log.dump_model('scratch/model_log_raw.txt')

model_lin_preds = model_lin.predict(dtest)
model_log_preds = model_log.predict(dtest)

print("Linear Regression Objective Function")
print("RMSE model Error: " + str( mean_squared_error(y_test, model_lin_preds) ) )
print("MAE Model Error:" + str( mean_absolute_error( y_test, model_lin_preds) ) )
print("------------------")


print("Linear Regression Model Feature Importance")
print(model_lin.get_fscore())
print('----------------')
print('Logistic Regression Model Feature Importance')
print(model_log.get_fscore())
