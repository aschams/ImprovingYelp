import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from itertools import compress
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from scipy import sparse



reviews = pd.read_csv("/users/aschams/scratch/Complete_reviews.csv")
print(reviews.dtypes[reviews.dtypes == np.object])
y = reviews['stars'].copy()
reviews_text = reviews[['text','review_id']].copy()
reviews.groupby("Review_Num").mean().to_csv("/users/aschams/scratch/Average_by_review_number.csv")
reviews.groupby("Review_Num").std().to_csv("/users/aschams/scratch/stdev_by_review_number.csv")
reviews.drop(['Unnamed: 0', 'Unnamed: 0.1','review_id', 'business_id', "date", 'text', 'stars', 'user_id', 'name', 'Unnamed: 0_y', 'postal_code', "BestNights_business_id", "Music"],
            axis = 1,
            inplace = True)

reviews_sparse = sparse.csr_matrix(reviews.astype(float))
print("---------------------")
print(reviews.dtypes[reviews.dtypes == np.object])
vectorizer = TfidfVectorizer(stop_words = "english",
                            max_df = 0.7,
                            min_df = 500,
                            token_pattern = '[A-Za-z][A-Za-z]+')

tfidf = vectorizer.fit_transform(reviews_text)
full_sparse_matrix = sparse.hstack([reviews_sparse, tfidf])
print(full_sparse_matrix.shape)

sparse.save_npz("/users/aschams/scratch/full_sparse_matrix.npz", full_sparse_matrix)
reviews_columns = np.concatenate((reviews.columns, np.array(vectorizer.get_feature_names())))

np.save("/users/aschams/scratch/sparse_matrix_columns.npy", reviews_columns)
print(reviews.columns)
reviews_text.to_csv("/users/aschams/scratch/Complete_reviews_text.csv")
reviews.to_csv("/users/aschams/scratch/Complete_reviews_barebones.csv")
y.to_csv("/users/aschams/scratch/Complete_reviews_stars.csv")
