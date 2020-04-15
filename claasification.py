#Classification-based Collaborative Filtering Systems
#Logistic Regression as a classifier
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn.linear_model import LogisticRegression
bank_full = pd.read_csv('bank_full_w_dummy_vars.csv')
bank_full.head()
bank_full.info() #print summary of dataframe
#y_binary is biniary version is binary version of target variable. Will be used to see id people will subscribe based on user attributes.Trained
#based on 19 variables such as housing, loans, employment
X = bank_full.ix[:,(18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36)].values
y = bank_full.ix[:,17].values
#training model with log. reg
LogReg = LogisticRegression()
LogReg.fit(X, y)
#attributes of new user
new_user = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
y_pred = LogReg.predict(new_user)#predict if new user will subscribe or not