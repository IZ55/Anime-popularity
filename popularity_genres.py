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
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
import scipy.stats as stats



#Linear regression for genres with tf-IDF
df = pd.read_csv("anime_dataset.csv")
df = df.dropna(axis = 'index', subset = ['score', 'popularity', 'genres', 'episodes'])


score = df["score"]
pop = df["popularity"]
df["pop_mod"] = (df["popularity"] / 100)
popularity = df["pop_mod"]


Y_dif = df["pop_mod"]
X_dif = TfidfVectorizer()
X_tfid = X_dif.fit_transform(df["genres"]) # matrix of tf-idf values

# print(X_tfid)

X_train_if, X_test_if, y_train_if, y_test_if = train_test_split(X_tfid, Y_dif, test_size = 0.2, random_state = 10)

# train the model and make a prediction on test data
model = LinearRegression()
model.fit(X_train_if, y_train_if)
y_pred_if = model.predict(X_test_if)

print('Coefficients:', model.coef_)

# #Evaluating the model
mse_score_if = mean_squared_error(y_test_if, y_pred_if)
r2_score_if = r2_score(y_test_if, y_pred_if)
print('mse', mse_score_if) # result is 9.099
print('r2', r2_score_if) # result is 0.004

#Adjusted R2
n2 = len(y_pred_if)
adj_r2_if = 1 - (1 - r2_score_if) * (n2 - 1) / (n2 - 95 - 1)
print('r2 adjusted', adj_r2_if) # result is -0.824


#Clustering
#do the elbow method
inertias = []

# for each potential n of clusters, fit a KMeans model and store inertia
for k in range(1, 11):
    kmeans = KMeans(n_clusters = k, random_state = 10)
    kmeans.fit(X_tfid)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 11), inertias, marker = 'o')
plt.title("Elbow")
plt.xlabel("N of clusters")
plt.ylabel("Inertia")
plt.show()

#Based on elbow diagram, I picked 3 clusters

#Create KMeans model with 3 clusters
kmeans_clustering = KMeans(n_clusters = 3, random_state =10)
# fit the KMeans model and assign each anime to a cluster
cluster_labels = kmeans_clustering.fit_predict(X_tfid)

df["cluster"] = cluster_labels
# per cluster, get the avg popularity
cluster_popularity = df.groupby("cluster")["pop_mod"].mean()
print(cluster_popularity)

# ANOVA
from scipy.stats import f_oneway

# get the popularity values for each cluster
groups_pop = df.groupby("cluster")["pop_mod"].apply(list).tolist()
# print('group_pop', groups_pop)

f_stat, p_value = f_oneway(groups_pop[0], groups_pop[1], groups_pop[2])
print('F', f_stat)
print('P', p_value)


#Which clusters are which
# get the names of genres
feature_names = X_dif.get_feature_names_out()

# convert the td-idf matrix to data frame with genre column titles
X_df = pd.DataFrame(X_tfid.toarray(), columns = feature_names)

print(feature_names)
print(X_df)

# get the average TD-IDF score per genre in that cluster, then return genre name with highest score
for cluster in range(3):
    cluster_rows = X_df[cluster_labels == cluster]
    avg = cluster_rows.mean()
    top_genres = avg.idxmax()
    print("Cluster", cluster, top_genres)


#PCA
# create a PCA model that reduces the data to 2 dimensions
pca = PCA(n_components=2, random_state=10)
# fit the PCA model and transform the data
X_test_2d = pca.fit_transform(X_tfid.toarray())

# print(X_test_2d)
print(X_test_2d.shape) # only 2 columns, pca has reduced all TF-IDF dimensions to 2

plt.scatter(X_test_2d[:,0], X_test_2d[:, 1], c=cluster_labels)
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.gca().invert_yaxis()

# convert the cluster centers of the Kmeans model to 2 dimensions
centers_cluster = pca.transform(kmeans_clustering.cluster_centers_)

print(centers_cluster)

# for each cluster, get the x and y coordinates and add the label to the diagram
for cluster in range(3):
    x, y = centers_cluster[cluster]
    plt.text(x, y, cluster, weight='bold')

plt.show()
