
## importing the dependencies
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt



## data colection and preprocessin
data = pd.read_csv("/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")

data.isnull().sum()

data.info()

data.describe()

x = data.iloc[:, 3:].values



## finding the wcss value
wcss = list()

for i in range(1,11):
    model = KMeans(n_clusters=i, init="k-means++", random_state=42)
    model.fit(x)
    wcss.append(model.inertia_)

sns.set()
plt.xlabel("wcss")
plt.ylabel("no. of cluster")
plt.title("gRAPH")
plt.plot(range(1,11),wcss)
plt.show()



## training the model
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)

kmeans.fit(x)

y = kmeans.predict(x)


## plotting the clusters
plt.xlabel("Spending")
plt.ylabel("Annual Income")
plt.title("CLUSTERING GRAPH")
plt.scatter(x[y==0,1], x[y==0,0], s=100, c="g", marker="*", alpha=1 )
plt.scatter(x[y==1,1], x[y==1,0], s=100, c="b", marker="*", alpha=1 )
plt.scatter(x[y==2,1], x[y==2,0], s=100, c="y", marker="*", alpha=1 )
plt.scatter(x[y==3,1], x[y==3,0], s=100, c="r", marker="*", alpha=1 )
plt.scatter(x[y==4,1], x[y==4,0], s=100, c="m", marker="*", alpha=1 )



plt.scatter(kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,0], s=50, marker="D", c="k")
plt.show()
