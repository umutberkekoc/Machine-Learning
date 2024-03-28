#########################
# K-Means
#########################

# Adımlar
# 1. Küme sayısı belirlenir
# 2. Rastgele k merkez seçilir
# 3. Her gözlem için k merkezlere uzaklıklar hesaplanır
# 4. Her gözlem en yakın olduğu merkeze atanır
# 5. Atama işlemlerinden sonra oluşan kümeler için tekrar
#    merkez hesaplamaları yapılır
# 6. Bu işlem belirlenen bir iterasyon adedince tekrar edilir ve küme için
#    hata kareler toplamlarının toplamının minimum olduğu durumdaki
#    gözlemlerin kümelenme yapısı nihai kümelenme olarak seçilir

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from yellowbrick.cluster import KElbowVisualizer
import warnings

warnings.filterwarnings(action="ignore")
warnings.simplefilter("ignore")

pd.set_option("display.width", 600)
pd.set_option("display.max_columns", None)

df = pd.read_csv("ML Datasets/USArrests.csv", index_col=0)  # read data set and create first column as index

df.head()
print(df.isnull().sum())  # zero nan value
print(df.describe().T)  # zero outlier value
df.nunique()
df.info()

# Normalization
num_cols = [i for i in df.columns if df[i].dtype in ["float64", "int64"]]
mms = MinMaxScaler((0, 1))
for i in num_cols:
    df[i] = mms.fit_transform(df[[i]])

# Creating K-Means Model
kmeans = KMeans(n_clusters=4, random_state=17).fit(df)
# There is no target / dependent variable in unsupervised learning models.
# so, we only write the independent variables (all variables in dataframe) when we fit the model as df instead of X.

print("Parameters-->", kmeans.get_params())  # Parameters of the kmeans
# Parameters--> {'algorithm': 'lloyd', 'copy_x': True, 'init': 'k-means++',
# 'max_iter': 300, 'n_clusters': 4, 'n_init': 'warn',
# 'random_state': 17, 'tol': 0.0001, 'verbose': 0}

print("n_clusters-->", kmeans.n_clusters)    # Number of clusters in kmeans (we set as 4)
# n_clusters--> 4

print("cluster_centers->\n", kmeans.cluster_centers_)  # Cluster centers
#  [[0.1686747  0.11485774 0.34028683 0.12601868]
#  [0.6124498  0.75       0.75423729 0.67980189]
#  [0.30439405 0.32937147 0.70588235 0.31098951]
#  [0.79141566 0.6802226  0.36864407 0.36466408]]

print("labels-->", kmeans.labels_)  # Labels / Clusters dist. / sep. by observations
# labels--> [3 1 1 3 1 1 2 2 1 3 2 0 1 2 0 2 0 3 0 1
# 2 1 0 3 2 0 0 1 0 2 1 1 3 0 2 2 2 2 2 3 0 3 1 2 0 2 2 0 0 2]

print("Inertia / SSD-->", kmeans.inertia_)  # Sum of Squared Distance / Error
# Inertia / SSD--> 3.6834561535859134

#############################################
# Kümelerin Görselleştirilmesi
# (Visualization of Clusters with Scatter Plot)
df = pd.read_csv("ML Datasets/USArrests.csv", index_col=0)
k_means_model = KMeans(n_clusters=2, random_state=17).fit(df)
print(k_means_model.n_clusters)  # 2
print(k_means_model.labels_)
# [1 1 1 1 1 1 0 1 1 1 0 0 1 0 0 0 0 1 0 1 0 1 0 1 0 0 0 1 0 0 1 1 1 0 0 0 0
# 0 0 1 0 1 1 0 0 0 0 0 0 0]

plt.scatter(df.iloc[:, 0], df.iloc[:, 1],
            c=k_means_model.labels_,
            s=50, cmap="viridis")
plt.title("Clusters")
plt.show()


print("Merkezler:", k_means_model.cluster_centers_)
# Merkezler: [[  4.84137931 109.75862069  64.03448276  16.24827586]
#  [ 11.85714286 255.          67.61904762  28.11428571]]


plt.scatter(k_means_model.cluster_centers_[:, 0],
            k_means_model.cluster_centers_[:, 1],
            c="black", s=200, alpha=0.5)
plt.show()


plt.scatter(df.iloc[:, 0], df.iloc[:, 1],
            c=k_means_model.labels_,
            s=50, cmap="viridis")
plt.title("Clusters")
plt.scatter(k_means_model.cluster_centers_[:, 0],
            k_means_model.cluster_centers_[:, 1],
            c="red", s=200, alpha=0.5)
plt.show()


################################################
# Optimum Küme Sayısının Belirlenmesi
# (Determining the Optimal Number of Clustering)
################################################

kmeans = KMeans(random_state=45)
print("n_clusters-->", kmeans.n_clusters)
# Number of clusters in kmeans (8 by default)

SSD = []  # sum of squared distances
for i in range(1, 30):
    kmeans = KMeans(n_clusters=i).fit(df)
    SSD.append(kmeans.inertia_)

print("Sum of Squared Errors-->", SSD)
# Sum of Squared Errors--> [13.184122550256443, 6.596893867946196, 5.010878493006417,
# 3.6834561535859134, 3.1831577316766535, 2.8566826167870354, 2.6175770946580874,
# 2.2970724476648137, 2.1056592993483965, 1.8836422448312036, 1.6709317200626483,
# 1.6615869640326153, 1.3523399835456271, 1.3259862561637004, 1.2295668839307154,
# 1.1754043503853258, 1.0986145959734581, 1.0001476568473509, 0.9350673446891404,
# 0.8579191668633623, 0.816717943575085, 0.781891567869891, 0.701778094075777,
# 0.6434305604016608, 0.5932562512146997, 0.5251879071213519, 0.4618491670316317,
# 0.44083444071161726, 0.39769547492071344]

plt.plot(range(1, 30), SSD, "bx-", color="red")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD\n(SSE/SSR/SSD Values By Different K Values)")
plt.ylabel("Sum of Squared Errors")
plt.title("Optimum Küme Sayısı İçin Elbow Yöntemi\n(Elbow Method to Find Out The Optimal Number of Cluster)")
print(plt.show())


kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
print(elbow.show())

print("Optimum Küme Sayısı-->", elbow.elbow_value_)
# Optimum Küme Sayısı--> 6

#####################################
# Final Cluster'ların Oluşturulması
# Creating of the Final Clusters
#####################################

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)
print("n_clusters-->", kmeans.n_clusters)
print("cluster_centers->\n", kmeans.cluster_centers_)
print("labels-->", kmeans.labels_)
print("Inertia / SSD-->", kmeans.inertia_)
# Inertia / SSD--> 2.955884622294476

clusters = kmeans.labels_
df = pd.read_csv("ML Datasets/USArrests.csv", index_col=0)

df["cluster"] = clusters
df.head()

df["cluster"].value_counts()
print(df[df["cluster"] == 0])
print(df[df["cluster"] == 1])
print(df[df["cluster"] == 2])
print(df[df["cluster"] == 3])
print(df[df["cluster"] == 4])
print(df[df["cluster"] == 5])

print(df.groupby("cluster").agg(["count", "mean", "median"]))

df.to_csv("clusters.csv")