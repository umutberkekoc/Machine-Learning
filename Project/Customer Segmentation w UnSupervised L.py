import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
import datetime as dt

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

pd.set_option("display.width", 700)
pd.set_option("display.max_columns", None)

#############################################################
# Görev 1: Veriyi Hazırlama

# Adım 1: Veri setini okuma
df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()

df.head()
df.isnull().sum()

# Adım 2: Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz
# Tenure (müşterinin yaşı), Recency (en son kaç gün önce alışveriş yaptığı) gibi dğeişkenler oluşturunuz

today = dt.datetime(year=dt.datetime.today().year,
                    month=dt.datetime.today().month,
                    day=dt.datetime.today().day)

df.info()
date_cols = [i for i in df.columns if "date" in i]
# date_cols = df.columns[df.columns.str.contains("date")]

for i in date_cols:
    df[i] = df[i].astype("datetime64[ns]")

df.info()

# Tenure değişkeni oluşturma ve gerekli işlemler
df["Tenure"] = today - df["first_order_date"]
df["Tenure"] = df["Tenure"].astype(str).apply(lambda x: x.split()[0])
df["Tenure"] = df["Tenure"].astype("int64")

# Recency değişkeni oluşturma ve gerekli işlemler
df["Recency"] = today - df["last_order_date"]
df["Recency"] = df["Recency"].astype(str).apply(lambda x: x.split()[0])
df["Recency"] = df["Recency"].astype("int64")
df.head()

# Değişken seçimi
model_df = df.select_dtypes(include=["int64", "float64"])
model_df.head()

#############################################################
# Görev 2: K-Means ile Müşteri Segmentasyonu

# Adım 1: Değişkenleri standartlaştırınız
import scipy.stats as stats

skewness = model_df.skew()
kurtosis = model_df.kurtosis()
print("Skewness:\n", skewness)
print("Kurtosis:\n", kurtosis)

def check_skew(dataframe, column):
    skew = stats.skew(dataframe[column])
    skewtest = stats.skewtest(dataframe[column])
    plt.title("Distribution of " + column)
    sns.distplot(dataframe[column], color="green")
    print("{}'s Skew: {}, : {}".format(column, skew, skewtest))
    return

plt.figure(figsize=(10, 10))
plt.subplot(6, 1, 1)
check_skew(model_df, "order_num_total_ever_online")
plt.subplot(6, 1, 2)
check_skew(model_df, "order_num_total_ever_offline")
plt.subplot(6, 1, 3)
check_skew(model_df, "customer_value_total_ever_offline")
plt.subplot(6, 1, 4)
check_skew(model_df, "customer_value_total_ever_online")
plt.subplot(6, 1, 5)
check_skew(model_df, "Tenure")
plt.subplot(6, 1, 6)
check_skew(model_df, "Recency")
plt.tight_layout()
plt.show()




# Normal dağılımın sağlanamsı için log transformation uygulaması
model_df.head(2)
model_df["order_num_total_ever_offline"] = np.log1p(model_df["order_num_total_ever_offline"])
model_df["order_num_total_ever_online"] = np.log1p(model_df["order_num_total_ever_online"])
model_df["customer_value_total_ever_offline"] = np.log1p(model_df["customer_value_total_ever_offline"])
model_df["customer_value_total_ever_online"] = np.log1p(model_df["customer_value_total_ever_online"])
model_df["Recency"] = np.log1p(model_df["Recency"])
model_df["Tenure"] = np.log1p(model_df["Tenure"])
model_df.head(2)

# Scaling
for i in model_df.columns:
    mms = MinMaxScaler((0, 1))
    model_df[i] = mms.fit_transform(model_df[[i]])

model_df.head()


# Adım 2: Optimum küme sayısını belirleyiniz
kmeans = KMeans()
print("n_clsuters-->", kmeans.n_clusters)
# n_clsuters--> 8
print("Parameters-->", kmeans.get_params())
# Parameters--> {'algorithm': 'lloyd', 'copy_x': True, 'init': 'k-means++',
# 'max_iter': 300, 'n_clusters': 8, 'n_init': 'warn', 'random_state': 42, '
# tol': 0.0001, 'verbose': 0}

SSD = []
for i in range(2, 40):
    kmeans = KMeans(n_clusters=i).fit(model_df)
    SSD.append(kmeans.inertia_)

print("SSD-->\n", SSD)

plt.plot(range(2, 40), SSD, "bx-", color="orange")
plt.xlabel("Farklı K Değerlerine Karşılık SSD\n(SSD Values By Different K Values)")
plt.ylabel("Sum of Squared Errors")
plt.title("Optimum Küme Sayısı İçin Elbow Yöntemi\n(Elbow Method to Find Out The Optimal Number of Cluster)")
print(plt.show())


# 2. Way to determine the optimal value of n_cluster
kmeans = KMeans(random_state=42)
elbow = KElbowVisualizer(kmeans, k=(2, 40)).fit(model_df)
elbow.show()

print("Optimum Küme Sayısı-->", elbow.elbow_value_)
# Optimum Küme Sayısı--> 11
print(elbow.elbow_score_)  # 931


# Adım 3: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz
k_means = KMeans(n_clusters=elbow.elbow_value_,
                 random_state=42).fit(model_df)

print(k_means.labels_)
labels = k_means.labels_

"""final_df = df[["master_id", "order_num_total_ever_online", "order_num_total_ever_offline",
                "customer_value_total_ever_online", "customer_value_total_ever_offline",
               "Recency", "Tenure"]]"""

final_df = model_df.copy()
final_df["master_id"] = df["master_id"]
final_df["Segment"] = labels
final_df.head()


# Adım 4: Her bir segmenti istatistiksel olarak inceleyiniz
final_df.pivot_table(["order_num_total_ever_offline", "order_num_total_ever_online",
                      "customer_value_total_ever_offline", "customer_value_total_ever_online",
                      "Recency", "Tenure"],
                     "Segment",
                     aggfunc=["mean", "min", "max", "count"])

final_df.groupby("Segment").agg({"order_num_total_ever_online": ["mean", "min", "max"],
                                 "order_num_total_ever_offline": ["mean", "min", "max"],
                                 "customer_value_total_ever_online": ["mean", "min", "max"],
                                 "customer_value_total_ever_offline": ["mean", "min", "max"],
                                 "Recency": ["mean", "min", "max"],
                                 "Tenure": ["mean", "min", "max", "count"]})


#############################################################
# Görev 3: Hieararchical Clustering ile Müşteri Segmentasyonu

# Adım 1: Görev 2'de standartlaştırdığınız dataframe'i kullanarak "optimum küme" sayısını belirleyiniz
hc_complete = linkage(model_df, method="complete")
# method: complete, single, average, median,...

plt.figure(figsize=(7, 7))
plt.title("Dendrogram (average)")
dend = dendrogram(hc_complete,
                  truncate_mode="lastp",
                  p=10,
                  show_contracted=True,
                  leaf_font_size=12)
plt.axhline(y=1.2, color="purple", linestyle="dotted")
plt.show()


# Adım 2: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz
hc = AgglomerativeClustering(n_clusters=7)
segments = hc.fit_predict(model_df)

final_df2 = df[["master_id", "order_num_total_ever_online", "order_num_total_ever_offline",
                "customer_value_total_ever_online", "customer_value_total_ever_offline",
               "Recency", "Tenure"]]

final_df2["Segments"] = segments
final_df2.head()
final_df2["Segments"].value_counts()

# Adım 3: Her bir segmenti istatistiksel olarak inceleyiniz

final_df2.groupby("Segments").agg({"order_num_total_ever_online": ["mean", "min", "max"],
                                 "order_num_total_ever_offline": ["mean", "min", "max"],
                                 "customer_value_total_ever_online": ["mean", "min", "max"],
                                 "customer_value_total_ever_offline": ["mean", "min", "max"],
                                 "Recency": ["mean", "min", "max"],
                                 "Tenure": ["mean", "min", "max", "count"]})