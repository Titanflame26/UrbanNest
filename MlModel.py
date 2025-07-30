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


df['2BHK Rent'] = df['2BHK Rent'].astype(str).str.replace(',', '')

# Convert the column to numeric, invalid parsing will be set to NaN (use 'coerce' to handle errors)
df['2BHK Rent'] = pd.to_numeric(df['2BHK Rent'], errors='coerce')
mn1 = math.floor(df['2BHK Rent'].mean())
df['2BHK Rent'] = df['2BHK Rent'].fillna(mn1)


df['3 BHK Rent'] = df['3 BHK Rent'].astype(str).str.replace(',', '')

# Convert the column to numeric,(use 'coerce' to handle errors)
df['3 BHK Rent'] = pd.to_numeric(df['3 BHK Rent'], errors='coerce')
mn2 = math.floor(df['3 BHK Rent'].mean())
df['3 BHK Rent'] = df['3 BHK Rent'].fillna(mn2)

df.head(20)

# Remove commas and convert to numeric (forcing errors to NaN)
df['Population'] = df['Population'].astype(str).str.replace(',', '').astype(float)

# Now calculate the mean or perform any other operation

 
from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
kmmodel=KMeans(n_clusters=3)
x=df[['Population','Commercial Shops','Education Institutions','Hospital','Bus','Recreation','Food','Pharmacies','Salon','ATM','Small shops','Mart','Air Quality index','Theater','1BHK rent','2BHK Rent','3 BHK Rent']]
#x_scaled=scalar.fit_transform(x)
y_predicted=kmmodel.fit_predict(x)
df['cluster']=y_predicted
