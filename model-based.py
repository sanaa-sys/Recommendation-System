#Segment 2 - Model-based Collaborative Filtering Systems with SVD Matrix Factorization
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn.decomposition import TruncatedSVD
#preparing data
columns = ['user_id', 'item_id', 'rating', 'timestamp']
frame = pd.read_csv('u.data', sep='\\t', names=columns)
frame.head()
columns = ['item_id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
              'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
            'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('u.item', sep='|', names=columns, encoding='latin-1')
movie_names = movies[['item_id', 'movie title']]
movie_names.head()
#Combining the 2 data frames
combined_movies_data = pd.merge(frame, movie_names, on='item_id')
combined_movies_data.head()
#Access movies with most reviews, group movies the item id and count the rating and sort them in descending order
combined_movies_data.groupby('item_id')['rating'].count().sort_values(ascending=False).head()
#find name of movie
filter = combined_movies_data['item_id']==50 #an array will be generated with bool values
combined_movies_data[filter]['movie title'].unique() #find unique movie title whre id is 50
#building utility matrix, contain value for each user and movie
rating_crosstab = combined_movies_data.pivot_table(values='rating', index='user_id', columns='movie title', fill_value=0)#TruncatedSVD DOESNT ACCEPT NULL SO FILL WITH 0
rating_crosstab.head()
#Transposing matric
X = rating_crosstab.values.T
#Decomposing matrix
SVD = TruncatedSVD(n_components=12, random_state=17) #instanstiate SVD object with 12 dimensions
resultant_matrix = SVD.fit_transform(X)#fits SVD model to X
resultant_matrix.shape
#Generate corelation matrix by finding Pearson R value for every movie pair in resultant matrix based on user preferences
corr_mat = np.corrcoef(resultant_matrix)
corr_mat.shape
#Isolating Star Wars From the Correlation Matrix
movie_names = rating_crosstab.columns #generate movie names index
movies_list = list(movie_names) #convert numpy array to list
star_wars = movies_list.index('Star Wars (1977)') #find index of star wars
corr_star_wars = corr_mat[1398] #isolate array that represents star wars
corr_star_wars.shape
#Recommending a Highly Correlated Movie, generating list of movies with Pearson R value close to 1
list(movie_names[(corr_star_wars<1.0) & (corr_star_wars > 0.9)])
list(movie_names[(corr_star_wars<1.0) & (corr_star_wars > 0.95)])