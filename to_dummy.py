import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import ast
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from itertools import compress


reviews = pd.read_csv("/users/aschams/scratch/sentiment_numbered_reviews.csv")

dummies = pd.get_dummies(reviews,
                         prefix = {'state': 'St',
                                   'neighborhood':'nbrhd',
                                   'city': 'city',
                                   'Alcohol': 'alcohol',
                                   'RestaurantsAttire': 'Attire',
                                   'RestaurantsReservations':'Reservations',
                                   'WiFi': 'wifi',
                                   'Caters': 'caters',
                                   'HasTV':'TV',
                                   'NoiseLevel':'Noise',
                                   'WheelchairAccessible':'Wheelchair',
                                   'BikeParking': 'BikePark',
                                   'BusinessAcceptsCreditCards' : 'CreditCards',
                                   'BusinessAcceptsBitcoin':'Bitcoin',
                                   'ByAppointmentOnly': 'AppointmentOnly',
                                   'CoatCheck': "CoatCheck",
                                   "HappyHour": 'happyHour',
                                   'OutdoorSeating': "OutdoorSeating",
                                   "RestaurantsDelivery":"Delivery",
                                   "RestaurantsGoodForGroups": "GoodForGroups",
                                   "RestaurantsTableService": "TableService",
                                   "RestaurantsTakeOut": "TakeOut",
                                   "Smoking":"Smoking"},
                         columns = ['state', 'neighborhood', 'city', 'Alcohol','Caters', 'HasTV', 'NoiseLevel',
                                    'RestaurantsAttire', 'RestaurantsReservations', 'WheelchairAccessible',
                                    'WiFi','BikeParking', 'BusinessAcceptsCreditCards', 'BusinessAcceptsBitcoin',
                                    'ByAppointmentOnly', 'CoatCheck', 'HappyHour', 'OutdoorSeating', 'RestaurantsDelivery',
                                    'RestaurantsGoodForGroups', 'RestaurantsTableService', "RestaurantsTakeOut", "Smoking"])

dummies.drop(['attributes', 'categories', 'address', 'categories', 'hours', 'Ambience_business_id', 'BusinessParking_business_id',
       'GoodForMeal_business_id', "Music_business_id"], axis = 1, inplace = True)
dummies.to_csv("/users/aschams/scratch/Complete_reviews.csv")
