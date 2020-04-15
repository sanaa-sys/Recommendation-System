#Content-Based Recommender Systems
#Nearest Neighbors Algorithm
import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors
cars = pd.read_csv('mtcars.csv')
cars.columns = ['car_names', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb'] #name columns of dataframe
cars.head()
t = [15, 300, 160, 3.2] #list for shoppers specs
X = cars.ix[:,(1, 3, 4, 6)].values #find a dataset to select only the values needed
X[0:5] #view first 5 records
#Nearest Neighbors model, find a model that is nearest to shoppers spec
nbrs = NearestNeighbors(n_neighbors=1).fit(X)
print(nbrs.kneighbors([t]))
