import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import ast

reviews = pd.read_csv("/users/aschams/scratch/full_review_sentiments.csv")
reviews.drop('Unnamed: 0', axis = 1, inplace = True)
sents = pd.concat([reviews, reviews['Sentiment'].map(eval).apply(pd.Series)], axis = 1)

sents.to_csv("/users/aschams/scratch/full_review_sentiments2.csv")
