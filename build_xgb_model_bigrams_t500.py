import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from itertools import compress
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import xgboost as xgb
from scipy import sparse
import pickle

data = pd.read_csv("/users/aschams/scratch/Complete_reviews.csv")

data_sample = data.sample(frac = 0.01, random_state = 4919)

data_text = data_sample['text'].copy()

stars = data_sample['stars'].copy()
stars = stars - 1

data_sample.drop(['Unnamed: 0', 'Unnamed: 0.1', 'business_id', "date", 'text',
                  'review_id', 'stars', 'user_id', 'name', 'Unnamed: 0_y',
                  'postal_code', "BestNights_business_id", "Music", 'latitude',
                  'longitude'],
            axis = 1,
            inplace = True)

# data_sample = pd.read_csv("scratch/reviews_sample_1percent.csv")
data_sample.to_csv("scratch/reviews_sample_1percent.csv")


data_sample.rename({'stars.1': 'avg_stars'}, axis=1, inplace =True)


data_sparse = sparse.csr_matrix(data_sample.astype(float))

def analyze_model(data_sparse, data_text, stars, max_words = None):
    print("Number of features: " + str(max_words))
    vectorizer = TfidfVectorizer(stop_words = "english",
                                min_df = .001,
                                ngram_range = (1,2),
                                token_pattern = '[A-Za-z][A-Za-z]+',
                                max_features=max_words)

    tfidf = vectorizer.fit_transform(data_text)
    full_sparse_matrix = sparse.hstack([data_sparse, tfidf])

    text_features = vectorizer.get_feature_names()
    text_features = [w + '_' for w in text_features]
    nontext_features = data_sample.columns

    features = np.concatenate((np.array(nontext_features, dtype = np.object), text_features), axis = None)
    features = features.tolist()
    np.savetxt('scratch/tfidf_features_names.txt', features, delimiter = ',', fmt = '%s' )

    print("Length of Vocabulary: " + str(len(vectorizer.get_feature_names())))
    X_train, X_test, y_train, y_test = train_test_split(full_sparse_matrix, stars, test_size = 0.4, random_state = 70)


    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names = features)
    # dtrain.save_binary("scratch/training_set.buffer")
    dtest = xgb.DMatrix(X_test, label = y_test, feature_names = features)
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

    model_lin_preds = model_lin.predict(dtest)
    model_log_preds = model_log.predict(dtest)

    print("Linear Regression Objective Function")
    print("RMSE model Error: " + str( mean_squared_error(y_test, model_lin_preds) ) )
    print("MAE Model Error:" + str( mean_absolute_error( y_test, model_lin_preds) ) )
    print("------------------")

    print("Logistic Regression Accuracy: " + str(accuracy_score(y_test, model_log_preds)))
    print("Linear Regression Model Feature Importance")
    print(model_lin.get_fscore())
    print()
    print(model_lin.get_score(importance_type='gain'))
    print('----------------')
    print('Logistic Regression Model Feature Importance')
    print(model_log.get_fscore())
    print()
    print(model_log.get_score(importance_type='gain'))

    print()
    print()
    print()

for vocab_size in [100, 200, 300, 400, 500, 750, 1000,1500,2000,3000,4000,5000]:
    analyze_model(data_sparse, data_text, stars, max_words = vocab_size)
