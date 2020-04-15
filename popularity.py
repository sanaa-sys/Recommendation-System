import pandas as pd
import numpy as np
frame = pd.read_csv("rating_final.csv") #Reading file
frame.head() #Returns top 5 rows of csv file by default
cuisine = pd.read_csv("chefmozcuisine.csv") #Reading file
cuisine.head()
#Recommendation based on counting
#Find out which place is popular
rating_count = pd.DataFrame(frame.groupby('placeID')['rating'].count()) #count ratings of each place and convert into dataframe
rating_count.sort_values('rating',ascending = False).head() #sort places in descendinng orders of rating
#finding similiarities with top5 places
most_rated_places = pd.DataFrame([135085, 132825, 135032, 135052, 132834], index=np.arange(5), columns=['placeID'])#making dataframe of top 5 places
summary = pd.merge(most_rated_places, cuisine, on='placeID') #merging most_rated_places with cuisine data set
cuisine['Rcuisine'].describe() #types of cuisine available in the dataset in total
