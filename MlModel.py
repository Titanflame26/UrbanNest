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

y_predicted

df.head(20)


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Assuming x is defined and contains your data

# Apply PCA to reduce the dimensions to 2D
pca = PCA(0.95)
x_pca = pca.fit_transform(x)

# Fit the KMeans model
kmmodel = KMeans(n_clusters=3)
kmmodel.fit(x_pca)

# Get the cluster labels for each data point
labels = kmmodel.labels_

# Compute centroids of the clusters
centroids = kmmodel.cluster_centers_

# Plot the data points and color them by their cluster labels
plt.figure(figsize=(10, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=200, label='Centroids')

plt.title("KMeans Clusters (PCA 2D projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.colorbar(label="Cluster")
plt.show()

labels = kmmodel.labels_
labels

k_range=range(1,10)
sse=[]
for k in k_range:
    km1=KMeans(n_clusters=k)
    km1.fit(x)
    sse.append(km1.inertia_)
sse

plt.plot(k_range,sse)

from sklearn.metrics import silhouette_score

# Compute silhouette score
score = silhouette_score(x, kmmodel.labels_)
print("Silhouette Score:", score)

print(kmmodel.n_clusters)

print("""Choose a Sub-area from the given list:-
      Ideal Homes
Kenchenhalli
BEML Layout
Sri Krishna Garden Layout
Pattanagere
Gattigere
Channasandra
Aditya Layout
Jaware Gowda Nagara
Dubasipalya
Malyasandra
Vinayaka Layout
Kengeri Upanagara
Kodipalya
Dwaraka Nagar
Avalahalli
Byatarayanapura
Srinagar
Srinivas Nagar
Vidyapeeta Layout
Kathreguppe
Karesandra
Yarabnagar
Kadarenahalli
Banagirinagara
Ittamadu
Ganapathi Nagar
Chickpet
Nagarthpet
Avenue Road
Chamrajpete
Mamoolpet
Balepet
Kalassipalya
Tharagupet
Kodigehalli
Sonnenahalli
Kurudusonnenahalli
Basavanapura
Seegehalli
Kumbena Agrahara
Sadaramangala
Belathur
Duravani Nagar
ITI Colony
VB Layout
Bhattarahalli
Dominic Layout
Bharathi Nagar
Domsandra
Chaithanya Ananya
Govindaraja Nagar
Mudalapalya
Amarajyoti Nagar
Cauvery Nagar
Pattegarhpalya
Prashant Nagar
MC Layout
Marenahalli
RPC Layout
Bull Temple Road
Gandhi Bazar
BNC Area
Chamarajpet
Hanumantha Nagar
Thyagaraja Nagar
Chikkalsandra
Gowdanpalya
RK Layout
Ramanjeyanagar
Koramangala 1st Block
Koramangala 2nd Block
Koramangala 3rd Block
Koramangala 4th Block
Koramangala 5th Block
Koramangala 6th Block
Koramangala 7th Block
Koramangala 8th Block
Jayanagar 1st Block
Jayanagar 2nd Block
Jayanagar 3rd Block
Jayanagar 4th Block
Jayanagar 4th 'T' Block
Jayanagar 5th Block
Jayanagar 6th Block
Jayanagar 7th Block
Jayanagar 8th Block
Jayanagar 9th Block
JP Nagar 1st Phase
JP Nagar 2nd Phase
JP Nagar 3rd Phase
JP Nagar 4th Phase
JP Nagar 5th Phase
JP Nagar 6th Phase
JP Nagar 7th Phase
JP Nagar 8th Phase
JP Nagar 9th Phase
BDA Layout
ITI Layout
Agara Village
Sector 2
Sector 3
Sector 4
Sector 5
Sector 6
Sector 7
Rajiv Gandhi Nagar
Basaweshwaranagar
Gayatri Nagar
ISKCON
West Of Chord Road
Agrahara Dasarahalli
Manjunath Nagar
Annapoorneshwari Nagara
ITI Layout
MPM Layout
NGEF Layout
Papareddy Palya
Kalyana Nagar
Nagarabhavi Village
Canara Bank Colony
Bhairaveshwara Nagar
Moodala Palya
Neotown
Shikaripalya
Siraj Layout
Neeladri Nagar
Celebrity Classic Layout
Konappana Agrahara
Doddanagamangala
Bhovi Palya
Shanthi Pura
Ananth Nagar
Kammasandra
Andapura
RS Gardens
Bommasandra Industrial Layout
Thirupalya
Maragondanahalli
Veersandra
Appareddy Palya
Eshwara Layout
Laksmipura
Hoysala Nagar
Defence Colony
HAL 2nd Stage
Narasimha Layout
Rajaindustrial Layout
Goraguntepalya
Muniswara Nagar
Pampanagar
Kamlanehru Extension
Vinayaka Extension
Pramod Layout
ITI Layout 3rd Phase
Swami Shivanandapuram
New Bamboo Bazaar
Sulthangunta
Shivajinagar
Immadihalli
Siddapura
Nallurhalli
Pattandur Agrahara
Thubarahalli
Ramagondanahalli
Aditya Nagar
Vinatak Nagara
BEL Layout (BEL)
HMT Layout (HMT)
NTI Layout (NTI)
AMS Layout
Narsipura
Vaishnavi Layout
GD Layout
Nanjappa Layout
Durga Parmeshwari Layout
Yelahanka New Town 4th Phase
Yelahanka New Town 5th Phase
Akshaya Nagara
Attur Layout
MP Layout
Santhosh Nagar
RWF West Colony
Badrappa Layout
Udaya Layout
Chikka Bommasandra
Sector A
Ambedkar Colony
LBS Nagar
Somanagar
Anjana Nagar
Madeshwara Layout
Venkateshwara Layout
Vishwaneedam
Venkateshwara Nagar
Ganjana Nagar
Sriganda Nagar
Srinivasa Nagar
Kareemsab Nagar
Anekal Town
Begur
Harlur
Kodathi
Haldenahalli
Chandapura
Yarandahalli
RK Twp
Hennagara
Suryanagar
Thirumagondanahalli
Muthanallur
Chintalamadivala Village
Singena Agrahara
Thirupalya
Chinnapa Layout
Sriram Layout
Devasthanagalu
KBM Layout
Anandnagar
SBM Colony
Chola Nagar
Ayyappa Layout
Guddadahalli
R.T. Nagar
Subramani Nagar
Doddanekundi
Karthik Nagar
Sanjay Nagar
Aswath Nagar
CKB Layout
Belandur
Manjunatha Layout
Marthahalli Village
HAL Central Township
Sector 3
Jawahar Nagar
Adarsh Nagar
Veeraih Nagar
Subramani Apartment
Ullal Upanagara
Sir M Vishweshwaraiah Layout 6th Block
Aditya Nagar
JP Nagar 7th Phase
RBI Layout
Suncity Layout
Nrupathunga Nagar
MS Ramaiah City Layout
Aradhana Layout
Jumbo Sawari Dinne
Royal County
Kembathalli
Koth0or Dinne
Akshya Garden
Infosys Headquarters
""")
from sklearn.cluster import KMeans
import pandas as pd
