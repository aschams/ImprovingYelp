import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import ast
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from itertools import compress
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

reviews = pd.read_csv("/users/aschams/scratch/Complete_numbered_reviews.csv")
reviews_head = reviews.head()

sents = pd.DataFrame(reviews['review_id'])
sents['Sentiment'] = reviews['text'].apply(lambda x: analyser.polarity_scores(x))

sents = pd.concat([sents, sents['Sentiment'].map(eval).apply(pd.Series)], axis = 1)

sents.to_csv("/users/aschams/scratch/full_review_sentiments.csv")
