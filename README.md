# Recommendation-System
This project is inspired from a LinkedIn Learning course "Building a Recommendation System with Python Machine Learning & AI". 
This project includes the following files: 
 - Popularity based recommender.Items are recommended to users based on how popular those items are among other users.
 - Basic corelation based recommendation system.It uses Pearson's R correlation to recommend an item that is most similar to the item a user has already 
   chosen.In other words, it recommends an item that has a review score that correlates with another item that a user has already chosen.
 - Classification-based corelation system. It uses logistic regression to make personalised recommendations as it takes into account user attributes like
   relevent to the context.
 - Model-based collaborative filtering systems.It involves building a model from user ratings, and then make recommendations based on that model.It involves 
   building a Utility Matrix which contains values for each user, each item, and the rating each user gave to each item which is truncated with SVD to get 
   valuable information.
 - Content-based recommendation system. Recommend an item based on its features and how similar those are to features of other items in a dataset based on the nearest-neighbor algorithm.
 - Evaluation the models based on precision (model's relevancy ie. number of items that I liked that were also recommended to me divided by the number of items that were recommended to me)
   and recall (model's completeness ie. number of items that I liked that were also recommended to me divided by the number of items that I liked).
   
   
 Pearson value (R): Measure of linear correlation between two variables, or in this case, two items ratings.R value that's close to one or negative one than you know you have a strong 
 linear relationship between two variables. As R values get closer to zero, you know that the two variables are not linearly correlated. 
 Logistic Regression: Machine learning method used to predict the value of a numeric categorical variable based on its relationship with predictor variables.
 Nearest-neighbour algorithm: Also known as a memory-based system, because it memorizes instances and then recommends an item, or a single instance, based on how quantitatively similar it is to a new incoming instance.
