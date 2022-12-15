# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 18:31:58 2022

@author: LENOVO
"""

import numpy as np
import pandas as pd
df = pd.read_csv("50_Startups.csv")
df.head()
df1=df.rename({'R&D Spend':'RDS','Administration':'ADMS','Marketing Spend':'MKTS'},axis=1)
df1
df1.corr() 
# Profit and RDS has the highest correlation of 0.9729 following MKTS and ADMS


# Splitting the variables
Y = df1[["Profit"]]
# X = df1[["RDS"]] # Model 1
# X = df1[["RDS","ADMS"]] # Model 2
X = df1[["RDS","ADMS","MKTS"]] # Model 3


import matplotlib.pyplot as plt
plt.scatter(x = X["RDS"],y = Y, color= 'black')
plt.show()
# Here, the plot shows strong positive correlation
import matplotlib.pyplot as plt
plt.scatter(x = X["ADMS"],y = Y, color= 'black')
plt.show()
# Here, the plot shows low correlation
import matplotlib.pyplot as plt
plt.scatter(x = X["MKTS"],y = Y, color= 'black')
plt.show()
# Here, the plot shows moderate positive correlation

# Boxplot
df1.boxplot("RDS",vert=False)
df1.boxplot("ADMS",vert=False)
df1.boxplot("MKTS",vert=False)
# There are no outliers


# Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,Y_train)

# Predictions
Y_predtrain=LR.predict(X_train)
Y_predtest=LR.predict(X_test)
    

from sklearn.metrics import mean_squared_error, r2_score
mse1 = mean_squared_error(Y_train,Y_predtrain) 
mse2 = mean_squared_error(Y_test,Y_predtest)

rmse1 = np.sqrt(mse1).round(2)
rmse2 = np.sqrt(mse2).round(2)

r2_score1 = r2_score(Y_train,Y_predtrain)
r2_score2 = r2_score(Y_test,Y_predtest)


# Model Validation - Multicollinearity, KFold
# pip install statsmodels
import statsmodels.api as sma
X_new = sma.add_constant(X)
lm = sma.OLS(Y,X_new).fit()
lm.summary()
# p value of ADMS, MKTS is >0.05. Therefore it has multicollinearity issues

from sklearn.model_selection import KFold, cross_val_score
k=13
k_fold=KFold(n_splits=k, random_state=None)
cv_scores=cross_val_score(LR, X_train, Y_train, cv=k_fold)
mean_acc_score=sum(cv_scores)/len(cv_scores)


# Model Deletion - Cooks Distance
# Suppress scientific notation
import numpy as np
np.set_printoptions(suppress=True)
# Create instance of influence
influence = lm.get_influence()
# Obtain Cook's distance for each observation
cooks = influence.cooks_distance
# Display Cook's distances
print(cooks)

import matplotlib.pyplot as plt
plt.scatter(df1.RDS, cooks[0])
plt.xlabel('X')
plt.ylabel('Cooks Distance')
plt.show()

import matplotlib.pyplot as plt
plt.scatter(df1.ADMS, cooks[0])
plt.xlabel('X')
plt.ylabel('Cooks Distance')
plt.show()

import matplotlib.pyplot as plt
plt.scatter(df1.MKTS, cooks[0])
plt.xlabel('X')
plt.ylabel('Cooks Distance')
plt.show()



# MODEL 1
# MSE1 = 81445821.5764
# MSE2 = 80510457.0198
# RMSE1 = 9024.73
# RMSE2 = 8272.76
# r2_score1 = 0.96321
# r2_score2 = 0.94616
# Mean_accuracy = 95%

# MODEL 2
# MSE1 = 87479410.1205
# MSE2 = 68761805.0140
# RMSE1 = 9353.04
# RMSE2 = 8592.27
# r2_score1 = 0.94505
# r2_score2 = 0.93779
# Mean_accuracy = 94%

# MODEL 3
# MSE1 = 88540951.2358
# MSE2 = 75875986.8478
# RMSE1 = 9409.62
# RMSE2 = 8710.68
# r2_score1 = 0.94937
# r2_score2 = 0.89819
# Mean_accuracy = 91%

# Inference : Here Model 1 where Y=df1[["Profit"]] and X=df1[["RDS"]] is selected for profit of
# 50_startups data, since its r2 for train and test are 0.96321 and 0.94616, mean_accuracy = 95%, 
# mse and rmse and lower than the other models and for less expense and more profit the first model is the best.
