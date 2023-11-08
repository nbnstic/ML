import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Mall_Customers.csv')
df

x = df.iloc[:,3:]
x

plt.title('Unclustered Data')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.scatter(x['Annual Income (k$)'], x['Spending Score (1-100)'])

from sklearn.cluster import KMeans, AgglomerativeClustering
km = KMeans(n_clusters=3)
km.fit_predict(x)

x.shape

#sse
km.inertia_

sse = []
for k in range(1,16):
    km = KMeans(n_clusters=k)
    km.fit_predict(x)
    sse.append(km.inertia_)
    
plt.title('Elbow Method')
plt.xlabel('Values of k')
plt.ylabel('SSE')
plt.plot(range(1,16), sse, marker= '.', color='red')

silh = []
for k in range(2,16):
    km=KMeans(n_clusters=k)
    labels = km.fit_predict(x)
    score = silhouette_score(x, labels)
    silh.append(score)
    
plt.title('Silhouette Method')
plt.xlabel('Values of k')
plt.ylabel('Silhouette Score')
plt.grid()
plt.bar(range(2,16), silh, color='red')

km = KMeans(n_clusters = 5)
labels =km.fit_predict(x)

km.cluster_centers_
plt.subplot(1,2,1)
plt.title('Unclustered Data')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.scatter(x['Annual Income (k$)'], x['Spending Score (1-100)'])

plt.subplot(1,2,2)
plt.title('clustered Data')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.scatter(x['Annual Income (k$)'], x['Spending Score (1-100)'], c=labels)
plt.scatter(cent[:,0], cent[:,1], s=50, color='k')

km.inertia_
km.labels_
df[labels==4]

agl = AgglomerativeClustering(n_clusters=5)
alabels =agl.fit_predict(x)
alabels

plt.subplot(1,2,1)
plt.title('Agglomerative')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.scatter(x['Annual Income (k$)'], x['Spending Score (1-100)'], c=alabels)

plt.subplot(1,2,2)
plt.title('Kmeans')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.scatter(x['Annual Income (k$)'], x['Spending Score (1-100)'], c=labels)
plt.scatter(cent[:,0], cent[:,1], s=50, color='k')
