import pandas as pd

reviews = pd.read_csv("/users/aschams/scratch/Complete_reviews.csv")
reviews_head = reviews.head(100)

reviews_head.to_csv("/users/aschams/scratch/Complete_reviews_head_barebones.csv")
