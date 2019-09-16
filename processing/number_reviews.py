import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import ast
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from itertools import compress


reviews = pd.read_csv("/users/aschams/scratch/Complete_review_df2.csv")

reviews['Review_Num'] = reviews.groupby('business_id_x').cumcount()
reviews.to_csv("/users/aschams/scratch/Complete_numbered_reviews.csv")
