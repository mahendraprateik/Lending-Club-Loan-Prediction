# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 17:28:04 2017

@author: Prateikm
"""
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_auc_score

import pandas as pd

#importing the data
loan2016 = pd.read_csv("2016CleanData.csv")

#Setting Y variable
Y = loan2016.pop("Loan status")

#Describing numerical variables
loan2016.describe()

#Creating dummy variales
categorical = ["grade", "home_ownership"] #, "addr_state"]

for variable in categorical:
        dummies = pd.get_dummies(loan2016[variable], prefix = variable)
        loan2016 = pd.concat([loan2016, dummies], axis =1)
        loan2016.drop([variable], axis = 1, inplace = True)
        
loan2016.head()

#model
%%timeit

model = RandomForestRegressor(100, oob_score = True, n_jobs = -1, random_state = 42)
model.fit(loan2016, Y)
print "C-stat:", roc_auc_score(Y, model.oob_prediction_)

#feature importance
feature_importances = pd.Series(model.feature_importances_, index = loan2016.columns)
feature_importances.sort_values(inplace = True)
feature_importances.plot(kind = "barh", figsize = (20,20))


#Using different values of no of trees
result = []
n_estimator_options = [30, 50, 100, 200, 500, 1000, 2000]

for tree in n_estimator_options:
    model = RandomForestRegressor(tree, oob_score = True, n_jobs = -1, random_state = 42)
    model.fit(loan2016, Y)
    print tree, "trees"
    roc = roc_auc_score(Y, model.oob_prediction_)   
    print"C-stat:", roc
    result.append(roc)
    print ""
    
pd.Series(result, n_estimator_options).plot()
