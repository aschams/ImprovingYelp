# ImprovingYelp
The full paper for this project can be downloaded [here](https://scholar.smu.edu/datasciencereview/vol2/iss1/13/).

There are two purposes to this project. 
1. Try to predict the user rating of a restaurant using the text of their review.
2. Use the text from reviews for a restaurant or a cuisine to identify high-quality items on the menu. 

### Data

Data is from [Yelp Dataset Challenge](https://www.yelp.com/dataset/challenge)

Data Cleaning found in DataCleaning.ipynb. Individual steps found in processing folder.

Data Cleaning Pipeline:

number_reviews.py -> sent_analysis.py -> merge_sents.py -> to_dummy.py -> sparse_yelp_reviews


### Results

With respect to purpose 1, this project was largely successful. It produced predictions with RMSEs comparable to other top papers using more complicated methods such as RNNs. One weakness of this model is in predicting reviews between 2-4 stars. It appears that there is very little difference between your average 4 star review and a 5 star review, as well as 2 stars compared to 1.

![Confusion Matrix of Results](img/confusion_matrix.png)

Full results can be found in the paper above.
