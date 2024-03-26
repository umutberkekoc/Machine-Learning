import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score


train_set = pd.read_csv("titanic_train.csv")

# embarked değişkeni nan silme
train_set.dropna(subset="Embarked", inplace=True)

# Age değişkeni nan doldurma

train_set["Age"] = train_set.groupby("Sex")["Age"].transform(lambda x: x.fillna(x.median()))

# Cabin değişkeni nan doldurma
train_set[train_set["Cabin"].isnull()].head(10)

train_set["Cabin"] = (train_set.groupby(["Embarked", "Pclass", "Ticket"])["Cabin"].
                      transform(lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x))

train_set["Cabin"] = (train_set.groupby(["Embarked", "Pclass"])["Cabin"].
                      transform(lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x))


# name droplama
train_set.drop("Name", axis=1, inplace=True)

# Outlier Analizi:
def outlier_threshold(dataframe, variable, q1=0.05, q3=0.95):
    q1 = dataframe[variable].quantile(q1)
    q3 = dataframe[variable].quantile(q3)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    up = q3 + 1.5 * iqr
    return low, up

for i in train_set.columns:
    if train_set[i].dtype in ["int64", "float64"]:
        print(i, ":", outlier_threshold(train_set, i))
    else:
        continue

def check_outlier(dataframe, variable):
    low, up = outlier_threshold(dataframe, variable)
    if dataframe[(dataframe[variable] < low) | (dataframe[variable] > up)].shape[0] > 0:
        return True
    else:
        return False

for i in train_set.columns:
    if train_set[i].dtype in ["int64", "float64"]:
        print(i, ":", check_outlier(train_set, i))
    else:
        continue

def suppress_outliers(dataframe, variable):
    low, up = outlier_threshold(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low), variable] = low
    dataframe.loc[(dataframe[variable] > up), variable] = up

for i in train_set.columns:
    if train_set[i].dtype in ["int64", "float64"]:
        suppress_outliers(train_set, i)
    else:
        continue

# Feature Extraction:
# Age Segment Ayırma
train_set.head()

labels = ["Child", "Teenage", "Adult", "Old"]
bins = [train_set["Age"].min() - 1,
       14, 25, 55,
       train_set["Age"].max() + 1]

train_set["AGE_SEGMENT"] = pd.cut(x=train_set["Age"],
                                  bins=bins,
                                  labels=labels)

# Değişken Yakalama:
def grab_variables(dataframe, num_th=10, car_th=20):
    # Categoric Variables:

    cat_var = [i for i in dataframe.columns if dataframe[i].dtype in ["category", "object", "bool"]]

    cat_but_car = [i for i in dataframe.columns if dataframe[i].dtype in ["category", "object"]
                   and dataframe[i].nunique() > car_th]

    num_but_cat = [i for i in dataframe.columns if dataframe[i].dtype in ["float64", "int64"]
                   and dataframe[i].nunique() < num_th]

    cat_var = cat_var + num_but_cat
    cat_var = [i for i in cat_var if i not in cat_but_car]

    # Numeric Variables
    num_var = [i for i in dataframe.columns if dataframe[i].dtype in ["float64", "int64"]
                   and dataframe[i].nunique() >= num_th]

    print("Toplam Değişken Sayısı:", len(dataframe.columns))
    print("Categoric Değişken Sayısı:", len(cat_var))
    print("Kardinal Değişken Sayısı:", len(cat_but_car))
    print("Numeric Değişken Sayısı:", len(num_var))

    return cat_var, cat_but_car, num_var

cat_var, cat_but_car, num_var = grab_variables(train_set)

# ohe
ohe_cols = [i for i in train_set.columns if 10 > train_set[i].nunique() > 1
            and i != "Survived"]

train_set = pd.get_dummies(data=train_set, columns=ohe_cols,
                           drop_first=True, dtype="int64")

# Ticket değişkeni kaldırma
train_set.drop("Ticket", axis=1, inplace=True)

# Rare Encoding:
print(pd.DataFrame({"COUNT": train_set["Cabin"].value_counts(),
              "RATIO": (train_set["Cabin"].value_counts() / len(train_set)) * 100,
              "TARGET_MEAN": train_set.groupby("Cabin")["Survived"].mean()}))

rare_columns = "Cabin"
tmp = train_set[rare_columns].value_counts() / len(train_set)
istenilen_columns = tmp[tmp > 0.05].index


list_Cabin = [i for i in train_set["Cabin"]]
new_column = []
for i in list_Cabin:
    if i not in istenilen_columns:
         new_column.append("rare")
    else:
        new_column.append(i)

train_set["NEW_CABIN"] = new_column

train_set.drop("Cabin", axis=1, inplace=True)

# cabin için ohe:
train_set = pd.get_dummies(data=train_set, columns=["NEW_CABIN"],
                                        dtype="int64", drop_first=True)

# Standartlaştırma İşlemi:
for i in num_var:
    if i != "PassengerId":
        train_set[i] = RobustScaler().fit_transform(train_set[[i]])


df = train_set.copy()
df.head()
df.isnull().sum()

X = df.drop(["PassengerId", "Survived"], axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=45)

ada_model = AdaBoostClassifier().fit(X_train, y_train)

# Test:
y_pred = ada_model.predict(X_test)
print("*****Classification Report*****\n", classification_report(y_test, y_pred))
print("ROC_AUC:", roc_auc_score(y_test, y_pred))
# Accuracy: 0.76
# Precision: 0.67
# Recall: 0.62
# f1-score: 0.64
# roc_auc: 0.73

# Train:
y_pred_train = ada_model.predict(X_train)
print("*****Classification Report*****\n", classification_report(y_train, y_pred_train))
print("ROC_AUC:", roc_auc_score(y_train, y_pred_train))
# Accuracy: 0.83
# Precision: 0.79
# Recall: 0.78
# f1-score: 0.78
# roc_auc: 0.823

# Hiperparametre Optimizasyonu
ada_model2 = AdaBoostClassifier()
ada_model2.get_params()

params = {"n_estimators": [200, 250, 275, 300, 325, 350],
          "learning_rate": [0.001, 0.01, 0.1, 0.25, 0.5]}

ada_model_CV = GridSearchCV(ada_model2, params, cv=5, n_jobs=-1).fit(X_train, y_train)

print("Best Parameters:", ada_model_CV.best_params_)
# Best Parameters: {'learning_rate': 0.1, 'n_estimators': 325}

ada_model_FINAL = ada_model2.set_params(**ada_model_CV.best_params_).fit(X_train, y_train)

# Test:
y_pred_final = ada_model_FINAL.predict(X_test)
print("*****Classification Report*****\n", classification_report(y_test, y_pred_final))
print("ROC_AUC:", roc_auc_score(y_test, y_pred_final))
# Accuracy: 0.78
# Precision: 0.69
# Recall: 0.66
# f1-score: 0.67
# roc_auc: 0.751

# Train:
y_pred_final_train = ada_model_FINAL.predict(X_train)
print("*****Classification Report*****\n", classification_report(y_train, y_pred_final_train))
print("ROC_AUC:", roc_auc_score(y_train, y_pred_final_train))
# Accuracy: 0.83
# Precision: 0.77
# Recall: 0.78
# f1-score: 0.78
# roc_auc: 0.818




