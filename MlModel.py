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
