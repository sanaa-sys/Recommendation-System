import pandas as pd
import numpy as np
frame = pd.read_csv("rating_final.csv") #Reading file
frame.head() #Returns top 5 rows of csv file by default
cuisine = pd.read_csv("chefmozcuisine.csv") #Reading file
cuisine.head()
geodata = pd.read_csv("geoplaces2.csv") #Reading file
geodata.head()
places = geodata[['placeID', 'name']]
places.head()
#grouping and ranking data
rating = pd.DataFrame(frame.groupby('placeID')['rating'].mean()) #create a data frame bu grouping with place id and generate mean of rating given to each place
rating.head()
rating['rating_count'] = pd.DataFrame(frame.groupby('placeID')['rating'].count())#add a new column to rating to count how many ratings each place got
rating.head()
rating.describe() #to access statistical description of rating
rating.sort_values('rating_count', ascending=False).head() #to access the most popular place with highest rating with place id
places[places['placeID'] == 135085] #find name of popular place
cuisine[cuisine['placeID'] == 135085] #find cuisine of popular place
#Prepare data for analysis, build user by item utility matrix
places_crosstab = pd.pivot_table(data=frame, values='rating', index='userID', columns='placeID')#pivot table() will cross tabulate each user against each place
places_crosstab.head()
Tortas_ratings = places_crosstab[135085] #isolate user rating from resturants called Tortas
Tortas_ratings[Tortas_ratings>=0] #getting all user ratings for Tortas with ratings >= 0
#Evaluations based on corelations
similar_to_Tortas = places_crosstab.corrwith(Tortas_ratings) #find restuarants with ratings similiar to tortas based on similarities and user reviews
corr_Tortas = pd.DataFrame(similar_to_Tortas, columns=['PearsonR']) #convert matrix to dataframe
corr_Tortas.dropna(inplace=True)#drop null values
corr_Tortas.head()
Tortas_corr_summary = corr_Tortas.join(rating['rating_count']) #Also add the number of ratings to further enhance similarities
Tortas_corr_summary[Tortas_corr_summary['rating_count']>=10].sort_values('PearsonR', ascending=False).head(10) #Seeing only places with at least 10 rating values and sorting pearson r sorted in descending column
places_corr_Tortas = pd.DataFrame([135085, 132754, 135045, 135062, 135028, 135042, 135046], index = np.arange(7), columns=['placeID']) #remove pearson value of 1 as they have rated by 1 source only, so we will remove those values and places that seve fast food
summary = pd.merge(places_corr_Tortas, cuisine,on='placeID') #create a dataset to find cuisines with corelation based on pearson value
summary.head()#Only 5 places as not all places were listed in cuisines dataset
places[places['placeID']==135046] #getting name of place which serves fast food
cuisine['Rcuisine'].describe()