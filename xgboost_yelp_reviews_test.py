import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from itertools import compress
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from scipy import sparse



reviews = pd.read_csv("/users/aschams/scratch/Complete_reviews_head_barebones.csv")
y = reviews['stars']
reviews_text = reviews['text'].copy()
reviews.groupby("Review_Num").mean().to_csv("/users/aschams/scratch/Average_by_review_number_head.csv")
reviews.groupby("Review_Num").std().to_csv("/users/aschams/scratch/stdev_by_review_number_head.csv")
# reviews.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'business_id', "date", 'text', 'review_id', 'stars', 'user_id', 'name', 'Unnamed: 0_y', 'postal_code'],
#             axis = 1,
#             inplace = True)
reviews_sparse = sparse.csr_matrix(reviews.astype(float))


vectorizer = TfidfVectorizer(stop_words = "english",
                            max_df = 0.7,
                            min_df = 5,
                            token_pattern = '[A-Za-z][A-Za-z]+')

tfidf = vectorizer.fit_transform(reviews_text)
full_sparse_matrix = sparse.hstack([reviews_sparse, tfidf])
print(dir(sparse))
sparse.save_npz("/users/aschams/scratch/full_sparse_matrix_head.npz", full_sparse_matrix)
reviews_columns = np.concatenate((reviews.columns, np.array(vectorizer.get_feature_names())))

np.save("/users/aschams/scratch/sparse_matrix_head_columns.npy", reviews_columns)
print(reviews.columns)
reviews_text.to_csv("/users/aschams/scratch/Complete_reviews_head_bb_text.csv")
reviews.to_csv("/users/aschams/scratch/Complete_reviews_head_barebones.csv")
