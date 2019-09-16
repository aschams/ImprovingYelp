import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, recall_score, precision_score, average_precision_score, accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from itertools import compress
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import xgboost as xgb
from scipy import sparse
import pickle

data = pd.read_csv("/users/aschams/scratch/Complete_reviews.csv")

data_sample = data.sample(frac = 0.1, random_state = 4919)

data_text = data_sample['text'].copy()

stars = data_sample['stars'].copy()
stars = stars - 1
data_sample['compound'] = data_sample['compound'] + 1

data_sample.drop(['Unnamed: 0', 'Unnamed: 0.1', 'business_id', "date", 'text',
                  'review_id', 'stars', 'user_id', 'name', 'Unnamed: 0_y',
                  'postal_code', "BestNights_business_id", "Music", 'latitude',
                  'longitude'],
            axis = 1,
            inplace = True)

data_sample.rename({'stars.1': 'avg_stars'}, axis=1, inplace =True)
data_sample = np.nan_to_num(data_sample)

data_sparse = sparse.csr_matrix(data_sample.astype(float))

vectorizer = TfidfVectorizer(stop_words = "english",
                            max_df = 0.7,
                            min_df = .001,
                            ngram_range = (1,1),
                            token_pattern = '[A-Za-z][A-Za-z]+',
                            max_features = 500)

tfidf = vectorizer.fit_transform(data_text)
full_sparse_matrix = sparse.hstack([data_sparse, tfidf])

print("Length of Vocabulary: " + str(len(vectorizer.get_feature_names())))

X_train, X_test, y_train, y_test = train_test_split(full_sparse_matrix, stars, test_size = 0.4, random_state = 70)
LR_clf = LogisticRegression(C=2, class_weight=None, dual=False, fit_intercept=True,
                            intercept_scaling=1, max_iter=10000, multi_class='multinomial',
                            n_jobs=None, penalty='l2', random_state=50, solver='newton-cg',
                            tol=0.005, verbose=0, warm_start=False)
LR_clf.fit(X_train, y_train)
LR_preds = LR_clf.predict(X_test)

print("Logistic Regression Performance: ")
print( "Accuracy: "+ str(accuracy_score(LR_preds, y_test)))
