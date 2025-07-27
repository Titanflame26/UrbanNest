import pandas as pd
from sklearn.cluster import KMeans
df=pd.read_csv(r"C:\Users\bhara\Downloads\UrbanplannerF1.csv")
df.head()

rows_with_nulls = df[df.isnull().any(axis=1)]
print(rows_with_nulls)

df=df.drop(['Area'],axis="columns")

df.head(15)

print(df.columns)

#data preprocessing
import math
# First, remove any commas and ensure all data is in string format
df['1BHK rent'] = df['1BHK rent'].astype(str).str.replace(',', '')

# Convert the column to numeric, invalid parsing will be set to NaN (use 'coerce' to handle errors)
df['1BHK rent'] = pd.to_numeric(df['1BHK rent'], errors='coerce')

# Now calculate the mean
mn = math.floor(df['1BHK rent'].mean())
print(mn)
df['1BHK rent'] = df['1BHK rent'].fillna(mn)
