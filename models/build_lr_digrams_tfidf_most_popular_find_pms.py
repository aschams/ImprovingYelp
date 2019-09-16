import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.model_selection import train_test_split, GridSearchCV
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
                                ngram_range = (2,2))

    tfidf = vectorizer.fit_transform(data_text)
    full_sparse_matrix = sparse.hstack([data_sparse, tfidf])

    text_features = vectorizer.get_feature_names()
    # text_features = [w + '_' for w in text_features]
    nontext_features = clean_data.columns

    features = np.concatenate((np.array(nontext_features, dtype = np.object), text_features), axis = None)
    features = features.tolist()
    print(popular_id)
    X_train, X_test, y_train, y_test = train_test_split(full_sparse_matrix, data_stars, test_size = 0.4, random_state = 70)

    LR_GS_params = {'C':[0.25, 0.5, 0.75, 1, 2, 5],
                    'class_weight':[None, 'balanced'],
                    'random_state': [50],
                    'solver': ['lbfgs', 'newton-cg'],
                    'multi_class': ['multinomial'],
                    'tol': [0.005],
                    'max_iter': [10000]
                    }
    LR_clf = LogisticRegression()
    LR_GS = GridSearchCV(LR_clf, LR_GS_params, 'accuracy', cv=5)
    LR_GS.fit(X_train, y_train)
    print(LR_GS.best_estimator_)


most_popular = ['4JNXUYY8wbaaDmk3BPzlWw']

for restaurant in most_popular:
    analyze_business(restaurant, data)
