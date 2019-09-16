import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
    clean_data = np.nan_to_num(clean_data)
    clean_data = pd.DataFrame(clean_data)

    data_sparse = sparse.csr_matrix(clean_data.astype(float))

    vectorizer = TfidfVectorizer(stop_words = "english",
                                max_df = 0.7,
                                min_df = .001,
                                token_pattern = '[A-Za-z][A-Za-z]+',
                                ngram_range = (3,3))

    tfidf = vectorizer.fit_transform(data_text)
    full_sparse_matrix = sparse.hstack([data_sparse, tfidf])

    text_features = vectorizer.get_feature_names()
    # text_features = [w + '_' for w in text_features]
    nontext_features = clean_data.columns

    features = np.concatenate((np.array(nontext_features, dtype = np.object), text_features), axis = None)
    features = features.tolist()
    np.savetxt('scratch/tfidf_features_names_trigrams_lr_' + popular_id + '.txt', features, delimiter = ',', fmt = '%s' )
    print(popular_id)
    print("Length of Vocabulary: " + str(len(vectorizer.get_feature_names())))
    X_train, X_test, y_train, y_test = train_test_split(full_sparse_matrix, data_stars, test_size = 0.4, random_state = 70)

    LR_clf = LogisticRegression(C=5, class_weight=None, dual=False, fit_intercept=True,
                                intercept_scaling=1, max_iter=1000000, multi_class='multinomial',
                                n_jobs=None, penalty='l2', random_state=50, solver='newton-cg',
                                tol=0.1, verbose=0, warm_start=False)
    LR_clf.fit(X_train, y_train)
    LR_preds = LR_clf.predict(X_test)

    print("Logistic Regression Performance: ")
    print( "Accuracy: "+ str(accuracy_score(LR_preds, y_test)))

    print("-------------------")
    print(LR_clf.coef_)
    print('-------------------')


most_popular = ['4JNXUYY8wbaaDmk3BPzlWw', 'RESDUcs7fIiihp38-d6_6g',
                'K7lWdNUhCbcnEvI0NhGewg', 'cYwJA2A6I12KNkm2rtXd5g',
                'f4x1YBxkLrZg652xt2KR5g', 'DkYS3arLOhA8si5uUEmHOw',
                '2weQS-RnoOBhb1KsHKyoSQ', 'ujHiaprwCQ5ewziu0Vi9rw',
                'iCQpiavjjPzJ5_3gPD5Ebg', 'KskYqH1Bi7Z_61pH6Om8pg']

for restaurant in most_popular:
    analyze_business(restaurant, data)
