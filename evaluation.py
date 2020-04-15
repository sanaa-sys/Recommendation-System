#Evaluating Recommendation Systems
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
bank_full = pd.read_csv('bank_full_w_dummy_vars.csv')
bank_full.head()
#select variables needed for model to see if user will accept offer or not
X = bank_full.ix[:,(18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36)].values
y = bank_full.ix[:,17].values
#using log reg for prediction
LogReg = LogisticRegression()
LogReg.fit(X, y)
y_pred = LogReg.predict(X)
#evaluate performance
print(classification_report(y, y_pred))