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
from sklearn.tree import DecisionTreeRegressor, plot_tree

#read data
df = pd.read_csv("anime_dataset.csv")
df = df.dropna(axis = 'index', subset = ['score', 'popularity', 'genres', 'episodes'])


#Linear regression for score, episodes and popularity

score = df["score"]
pop = df["popularity"]
df["pop_mod"] = (df["popularity"] / 100)
popularity = df["pop_mod"]

X = df[["score", "episodes"]]
Y = df["pop_mod"]

# print(X)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 10)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print('Intercept:', model.intercept_)
print('Coefficients:', model.coef_)


#Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('mse', mse) # 8.779
print('r2', r2) # 0.041

#Adjusted R2
n2 = len(y_pred)
adj_r2 = 1 - (1 - r2) * (n2 - 1) / (n2 - 2 - 1)
print('r2 adjusted', adj_r2) # result is 0.032

# decision tree
tree_model = DecisionTreeRegressor(random_state = 10, max_depth = 3)

tree_model.fit(x_train, y_train)

y_tree = tree_model.predict(x_test)

plot_tree(tree_model, feature_names = ["score", "episodes"])
plt.show()
