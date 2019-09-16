import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, recall_score, precision_score, average_precision_score, accuracy_score
from itertools import compress
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import xgboost as xgb
from scipy import sparse
import pickle

data = pd.read_csv("/users/aschams/scratch/Complete_reviews.csv")

def analyze_business(popular_id, data):
    data_sample = data[data['business_id'] == popular_id]
    data_text = data_sample['text'].copy()

    data_stars = data_sample['stars'].copy()
    data_stars = data_stars - 1

    clean_data = data_sample.drop(['Unnamed: 0', 'Unnamed: 0.1', 'business_id', "date", 'text',
                      'review_id', 'stars', 'user_id', 'name', 'Unnamed: 0_y',
                      'postal_code', "BestNights_business_id", "Music", 'latitude',
                      'longitude'],
                axis = 1)

    clean_data.rename({'stars.1': 'avg_stars'}, axis=1, inplace =True)


    data_sparse = sparse.csr_matrix(clean_data.astype(float))

    vectorizer = TfidfVectorizer(stop_words = "english",
                                max_df = 0.7,
                                min_df = .001,
                                token_pattern = '[A-Za-z][A-Za-z]+',
                                ngram_range = (2,2))

    tfidf = vectorizer.fit_transform(data_text)
    full_sparse_matrix = sparse.hstack([data_sparse, tfidf])

    text_features = vectorizer.get_feature_names()
    # text_features = [w + '_' for w in text_features]
    nontext_features = clean_data.columns

    features = np.concatenate((np.array(nontext_features, dtype = np.object), text_features), axis = None)
    features = features.tolist()
    np.savetxt('scratch/tfidf_features_names_digrams' + popular_id + '.txt', features, delimiter = ',', fmt = '%s' )
    print(popular_id)
    print("Length of Vocabulary: " + str(len(vectorizer.get_feature_names())))
    X_train, X_test, y_train, y_test = train_test_split(full_sparse_matrix, data_stars, test_size = 0.4, random_state = 70)


    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names = features)
    dtest = xgb.DMatrix(X_test, label = y_test, feature_names = features)

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


    print("Linear Regression Model Feature Importance")
    print(model_lin.get_fscore())
    print('----------------')
    print('Logistic Regression Model Feature Importance')
    print(model_log.get_fscore())

    print('----------------------------')


most_popular = ['4JNXUYY8wbaaDmk3BPzlWw', 'RESDUcs7fIiihp38-d6_6g',
                'K7lWdNUhCbcnEvI0NhGewg', 'cYwJA2A6I12KNkm2rtXd5g',
                'f4x1YBxkLrZg652xt2KR5g', 'DkYS3arLOhA8si5uUEmHOw',
                '2weQS-RnoOBhb1KsHKyoSQ', 'ujHiaprwCQ5ewziu0Vi9rw',
                'iCQpiavjjPzJ5_3gPD5Ebg', 'KskYqH1Bi7Z_61pH6Om8pg']

for restaurant in most_popular:
    analyze_business(restaurant, data)
