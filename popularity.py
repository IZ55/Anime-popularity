import matplotlib.pyplot as plt
import math
import random
import pandas as pd

import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#read data
df = pd.read_csv("anime_dataset.csv")
df.dropna()

# score = df["score"]
# popularity = df["popularity"]
# episodes = df["episodes"]
# genres = df["genres"]
# description = df["synopsis"]

#show graph by score and popularity
# by_score = df[['score', 'popularity']]
# sns.lmplot(x = "score", y = "popularity", data = by_score)
# plt.gca().invert_yaxis()
# plt.show()


"""
Linear regression for score and popularity
"""

X_score = df[["score"]]
Y_score = df["popularity"]

# model = LinearRegression()
# model.fit(X_score, Y_score)
# y_pred_score = model.predict(X_score)


X_train_score, X_test_score, y_train_score, y_test_score = train_test_split(X_score, Y_score, test_size = 0.2)

model = LinearRegression()
model.fit(X_train_score, y_train_score)
y_pred_score = model.predict(X_test_score)


by_score = df[['score', 'popularity']]
sns.lmplot(x = "score", y = "popularity", data = by_score)
plt.gca().invert_yaxis()
plt.show()

#Evaluating the model
mse_score = mean_squared_error(y_test_score, y_pred_score)
r2_score = r2_score(y_test_score, y_pred_score)
print(mse_score) # result is 82453.98214699262
print(r2_score) # result is 0.1025333599059467
