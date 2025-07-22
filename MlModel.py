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
