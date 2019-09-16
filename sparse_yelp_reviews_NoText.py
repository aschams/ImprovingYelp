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
data_sample.drop(['Unnamed: 0', 'Unnamed: 0.1', 'business_id', "date", 'text',
                  'review_id', 'stars', 'user_id', 'name', 'Unnamed: 0_y',
                  'postal_code', "BestNights_business_id", "Music", 'latitude',
                  'longitude'],
            axis = 1,
            inplace = True)

print("---------------------")
print(reviews.dtypes[reviews.dtypes == np.object])
print(reviews.columns)
reviews_text.to_csv("/users/aschams/scratch/Complete_reviews_text.csv")
reviews.to_csv("/users/aschams/scratch/Complete_reviews_barebones_noTFIDF.csv")
y.to_csv("/users/aschams/scratch/Complete_reviews_stars.csv")
