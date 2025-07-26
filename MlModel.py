from sklearn.cluster import KMeans
import pandas as pd
df=pd.read_csv(r"D:\Projects to do\Liveable Neighbourhood planner\urbanf1.csv")
df.head()

#dropping area as it is not needed for model

df=df.drop(['Area'],axis="columns")

df.head(10)


# Filter rows with any null values
rows_with_nulls = df[df.isnull().any(axis=1)]
print(rows_with_nulls)

df.tail(10)

df_cleaned = df.dropna(how='all')



df_cleaned.tail()

#building model

km=KMeans(n_clusters=3)
x=df_cleaned[['Commercial Shops','Education Institutions','Hospital','Bus','Recreation','Food','Pharmacies','Salon']]
y=df_cleaned[['ATM','Small shops','Mart','Air Quality index','Theater','1BHK rent','2BHK Rent','3 BHK Rent']]
y_predicted=km.fit_predict(x,y)

df_cleaned['cluster']=y_predicted
df1=df_cleaned[df_cleaned.cluster==0]
df2=df_cleaned[df_cleaned.cluster==1]
df3=df_cleaned[df_cleaned.cluster==2]
df2


#applying PCA for visualization

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Perform PCA to reduce the data to 2 dimensions
pca = PCA(n_components=8)
x_pca = pca.fit_transform(x)
x_pca.shape

# Scatter plot of the clusters
plt.figure(figsize=(10, 6))
plt.scatter(x_pca[df_cleaned['cluster'] == 0, 0], x_pca[df_cleaned['cluster'] == 0, 1], label='Cluster 0', alpha=0.6)
plt.scatter(x_pca[df_cleaned['cluster'] == 1, 0], x_pca[df_cleaned['cluster'] == 1, 1], label='Cluster 1', alpha=0.6)
plt.scatter(x_pca[df_cleaned['cluster'] == 2, 0], x_pca[df_cleaned['cluster'] == 2, 1], label='Cluster 2', alpha=0.6)

# Adding labels and title
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D Visualization of Clusters using PCA')
plt.legend()
plt.show()#

#metrics to  analyse accuracy

from sklearn.metrics import silhouette_score

# Calculate the silhouette score
sil_score = silhouette_score(x_pca, km.labels_)
print("Silhouette Score:", sil_score)
