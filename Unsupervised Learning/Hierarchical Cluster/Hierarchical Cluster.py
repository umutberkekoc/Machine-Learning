#########################
# Hiyeraşik Kümeleme (Hierarchical Cluster)
#########################
import warnings

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from warnings import simplefilter, filterwarnings
import warnings

filterwarnings(action="ignore")
warnings.simplefilter("ignore")

# Data Read
df = pd.read_csv("ML Datasets/USArrests.csv", index_col=0)

# Normalization
num_cols = [i for i in df.columns if df[i].dtype in ["float64", "int64"]]

for i in num_cols:
    mms = MinMaxScaler((0, 1))
    df[i] = mms.fit_transform(df[[i]])

df.head()

# Creating Dendrograms
hc_average = linkage(df, method="average")  # method: single, complete, average, median, ward, ....
plt.figure(figsize=(12, 8))
plt.title("Hiyeraşik Kümeleme Dendrogramı (average)")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           leaf_font_size=10)
print(plt.show())


hc_complete = linkage(df, "complete")
plt.figure(figsize=(10, 6))
plt.title("Hiyeraşik Kümeleme Dendrogramı (complete)")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_complete,
           leaf_font_size=10)
print(plt.show())


plt.figure(figsize=(10, 6))
plt.title("Hiyeraşik Kümeleme Dendrogramı (average)")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           leaf_font_size=10,
           truncate_mode="lastp",  # en son p adet göster (küme sayısı)
           p=10,  # p değeri
           show_contracted=True)  # kümelerdeki eleman sayısını getir
print(plt.show())


plt.figure(figsize=(10, 6))
plt.title("Hiyeraşik Kümeleme Dendrogramı (complete)")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_complete,
           leaf_font_size=10,
           truncate_mode="lastp",  # show last number of p ()
           p=10,  # number of clusters (p value)
           show_contracted=True)  # show element number for each cluster
print(plt.show())

#####################################
# Optimum Küme Sayısının Belirlenmesi
# (Determining the Optimal Number of Clustering)
#####################################

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.6, color="red", linestyle="--")
plt.axhline(y=0.5, color="purple", linestyle="--")
plt.show()


plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend2 = dendrogram(hc_complete)
plt.axhline(y=1.1, color="green", linestyle="--")
plt.axhline(y=0.7, color="gray", linestyle="--")
plt.show()

#####################################
# Final Cluster'ların Oluşturulması
# (Creating Final Clusters)
#####################################

cluster = AgglomerativeClustering(n_clusters=5, linkage="average")
clusters = cluster.fit_predict(df)
# model sonunda sadece fit edebilirdik!

df = pd.read_csv("ML Datasets/USArrests.csv", index_col=0)
df["hier_cluster_no"] = clusters
df.head()

df["hier_cluster_no"].value_counts()
print(df[df["hier_cluster_no"] == 0])
print(df[df["hier_cluster_no"] == 1])
print(df[df["hier_cluster_no"] == 2])
print(df[df["hier_cluster_no"] == 3])
print(df[df["hier_cluster_no"] == 4])

