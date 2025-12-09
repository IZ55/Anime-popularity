import matplotlib.pyplot as plt
import math
import random
import pandas as pd

import numpy as np
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#read data
df = pd.read_csv("anime_dataset.csv")
# for every analysis I will delete the same data so that I can fairly compare models
df = df.dropna(axis = 'index', subset = ['score', 'popularity', 'genres', 'episodes'])


#Linear regression for score and popularity

score = df["score"]
pop = df["popularity"]
df["pop_mod"] = (df["popularity"] / 100)
popularity = df["pop_mod"]



X_score = df[["score"]]
Y_score = df["pop_mod"]

# print(X_score)

X_train_score, X_test_score, y_train_score, y_test_score = train_test_split(X_score, Y_score, test_size = 0.2, random_state = 10)

model = LinearRegression()
model.fit(X_train_score, y_train_score)
y_pred_score = model.predict(X_test_score)

print('Intercept:', model.intercept_)
print('Coefficients:', model.coef_)

#residual plot
sns.residplot(x = X_score, y = Y_score)
plt.xlabel("Score")
plt.ylabel("Popularity")
plt.show()

plt.scatter(X_test_score, y_test_score)
plt.plot(X_test_score, y_pred_score)
plt.title("Linear Regression of Score and Popularity")
plt.ylabel("Popularity")
plt.xlabel("Score")
plt.gca().invert_yaxis()
plt.show()



#Compute the Standard Error of the Slope
y_train_pred = model.predict(X_train_score)

residuals = y_train_score - y_train_pred

# compute the residual sum of squares
# rss says how much prediction error the model has on the training data
rss = np.sum(residuals**2)

# get the number of rows
print(X_train_score) # there are 837 rows
df = 837-2

sigma_squared = rss/df

x = X_train_score['score']

SSx = np.sum((x- x.mean())**2)

SE_b1 = np.sqrt(sigma_squared/SSx)

print('SE:', SE_b1)



#Evaluating the model
mse_score = mean_squared_error(y_test_score, y_pred_score)
r2_score = r2_score(y_test_score, y_pred_score)
print('mse', mse_score)
print('r2', r2_score)

#Double sided t-test

import scipy.stats as stats

b1 = model.coef_
t_stat = b1 / SE_b1

# get the p value
p = 2*(1 - stats.t.cdf(abs(t_stat), df = 835))
print('pvalue', p)
