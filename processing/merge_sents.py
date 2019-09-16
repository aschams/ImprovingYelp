import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import ast

sents = pd.read_csv('/users/aschams/scratch/full_review_sentiments.csv')
numbered_reviews = pd.read_csv("/users/aschams/scratch/Complete_numbered_reviews2.csv")

sent_numbered_reviews = numbered_reviews.merge(sents, on = 'review_id')

sent_numbered_reviews.to_csv("/users/aschams/scratch/sentiment_numbered_reviews.csv")
