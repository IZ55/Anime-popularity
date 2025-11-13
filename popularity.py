import matplotlib.pyplot as plt
import math
import random
import pandas as pd

import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer



#read data
df = pd.read_csv("anime_dataset.csv")
df.dropna()

score = df["score"]
popularity = df["popularity"]
episodes = df["episodes"]
genres = df["genres"]
description = df["synopsis"]

#show graph by score and popularity
# by_score = df[['score', 'popularity']]
# sns.lmplot(x = "score", y = "popularity", data = by_score)
# plt.gca().invert_yaxis()
# plt.show()


"""
Linear regression
"""
