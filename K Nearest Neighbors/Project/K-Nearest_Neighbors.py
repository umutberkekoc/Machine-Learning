import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier

pd.set_option("display.width", 700)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

df = pd.read_csv("ML Datasets/diabetes.csv")

# 1. Exploraty Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling & Prediction
# 4. Model Evaluation (Holdout- K-FOLD CROSS VALIDATE)
# 5. Hyperparameter Optimization
# 6. Final Model


###########################
# 1. Exploraty Data Analysis
def show_info(dataframe):
    print("********* HEAD **********")
    print(dataframe.head())
    print("********* NA **********")
    print(dataframe.isnull().sum())
    print("********* SHAPE **********")
    print(dataframe.shape)
    print("********* INFO **********")
    print(dataframe.info())
    print("********* COLUMNS **********")
    print(dataframe.columns)
    print("********* DESCRIPTIVE STATS. **********")
    print(dataframe.describe().T)

show_info(df)


def grab_variable(dataframe, num_th=10, car_th=20, var_name=False):
    # Categoric Variables #
    cat_var = [i for i in dataframe.columns if dataframe[i].dtypes in ["object", "category", "bool"]]
    num_but_cat = [i for i in dataframe.columns if dataframe[i].dtypes in ["int64", "float64"]
                   and dataframe[i].nunique() < num_th]
    cat_but_car = [i for i in dataframe.columns if dataframe[i].dtypes in ["category", "object"]
                   and dataframe[i].nunique() > car_th]
    cat_var = cat_var + num_but_cat
    cat_var = [i for i in cat_var if i not in cat_but_car]

    # Numeric Variables #
    num_var = [i for i in dataframe.columns if dataframe[i].dtypes in ["int64", "float64"]
               and dataframe[i].nunique() > num_th]

    print("Observation:", len(dataframe))
    print("Number of Variables:", len(dataframe.columns))
    print("Number of Categoric Variables:", len(cat_var))
    print("Number of Num but Cat Variables:", len(num_but_cat))
    print("Number of Cat but Cardinal Variables:", len(cat_but_car))
    print("Number of Numeric Variables:", len(num_var))

    if var_name:
        print("Categoric Variables:", cat_var)
        print("Numeric But Categoric Variables:", num_but_cat)
        print("Categoric But Cardinal Variables:", cat_but_car)
        print("Numeric Variables:", num_var)
    return cat_var, cat_but_car, num_var

grab_variable(df, var_name=True)
cat_var, cat_but_car, num_var = grab_variable(df)

############################################
# 2. Data Preprocessing & Feature Engineering
# Outliers
def outlier_thresholds(dataframe, variable, quantile1=0.05, quantile3=0.95):
    # you can prefer to setup the quantiles as 0.01 and 0.99
    q1 = dataframe[variable].quantile(quantile1)
    q3 = dataframe[variable].quantile(quantile3)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    return lower_limit, upper_limit

def check_outliers(dataframe, variable):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable] < lower_limit) | (dataframe[variable] > upper_limit)].shape[0] > 0:
        return True
    else:
        return False

def show_outliers(dataframe, variable, head=3):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable] < lower_limit) | (dataframe[variable] > upper_limit)].shape[0] > 0:
        print("Number of Observed Outliers-->", dataframe[(dataframe[variable] < lower_limit) |
                                                           (dataframe[variable] > upper_limit)].shape[0])
        print(dataframe[(dataframe[variable] < lower_limit) |
                        (dataframe[variable] > upper_limit)].head(head))
    else:
        print("There are not any outliers")

def suppress_outliers(dataframe, variable):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable] < lower_limit) | (dataframe[variable] > upper_limit)].shape[0] > 0:
        dataframe.loc[(dataframe[variable] < lower_limit), variable] = lower_limit
        dataframe.loc[(dataframe[variable] > upper_limit), variable] = upper_limit
        print("Outliers Suppressed for " + variable)
    else:
        print("There are not any outlier for " + variable)

def remove_outliers(dataframe, variable):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    dataframe = dataframe[~((dataframe[variable] < lower_limit) | (dataframe[variable] > upper_limit))]
    return dataframe

for i in num_var:  # Creating new df without outliers
    new_df = remove_outliers(df, i)

def boxplot(dataframe, variable):
    sns.boxplot(data=dataframe, x=variable, color="red", whis=(5, 95))
    plt.title("BOXPLOT for " + variable)
    plt.xlabel(variable)
    plt.show()

for i in num_var:
    boxplot(df, i)

for i in num_var:
    print(i, outlier_thresholds(df, i))

for i in num_var:
    print(i, check_outliers(df, i))

for i in num_var:
    print(i,  show_outliers(df, i), end="\n\n")

for i in num_var:
    suppress_outliers(df, i)
    print(i, check_outliers(df, i))

# Missing Values
print(df.isnull().sum())
print(df.describe().T)

# It seems missing values are filled by zeros because it is not possible to become the minimum value of
# some of the variables like Glucose, Insulin, BloodPressure, BMI and SkinThickness.

zero_columns = [i for i in df.columns if df[i].min() == 0 and i in num_var]
zero_columns = [i for i in zero_columns if "Pregnancies" not in i]

for i in zero_columns:
    df[i].replace(0, np.nan, inplace=True)
print(df.isnull().sum())
print(df.describe().T)

def missing_value_table(dataframe):
    miss_var = [i for i in dataframe.columns if dataframe[i].isnull().sum() > 0]
    n_miss = dataframe[miss_var].isnull().sum().sort_values(ascending=False)
    ratio = 100 * (dataframe[miss_var].isnull().sum() / len(dataframe)).sort_values(ascending=False)
    miss_table = pd.concat([n_miss, round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(miss_table, end="\n")

missing_value_table(df)

def deal_with_missings(dataframe):
    nan_var = [i for i in dataframe.columns if dataframe[i].isnull().sum() > 0]
    for i in nan_var:
        print(" INFORMATION OF NAN VARIABLES\n", dataframe[i].describe().T,
              "\n Dtype:\n", dataframe[i].info())
        print(" **** Options ****\n1. Fill NA\n2. Drop NA")
        option = int(input("Enter you choice, 1/2"))
        if option == 1:
            print(" **** Options ****\n1. mean\n2. median\n3. mode\n4. variance\n5. Specific Number")
            option = int(input("Enter you choice, 1/2/3/4"))
            if option == 1:
                dataframe[i].fillna(dataframe[i].mean(), inplace=True)
            elif option == 2:
                dataframe[i].fillna(dataframe[i].median(), inplace=True)
            elif option == 3:
                dataframe[i].fillna(dataframe[i].mode()[0], inplace=True)
            elif option == 4:
                dataframe[i].fillna(dataframe[i].var(), inplace=True)
            elif option == 5:
                dataframe[i].fillna(int(input("Enter a number to filling nan values")), inplace=True)
            else:
                print("Enter a correct option number!!!")
                break
        elif option == 2:
            dataframe.dropna(inplace=True)
        else:
            print("Enter a correct option number!!!")

deal_with_missings(df)
print(df.isnull().sum())
#print(df.describe().T)

for i in num_var:
    print(i, check_outliers(df, i))
    if check_outliers(df, i) == True:
        suppress_outliers(df, i)


cat_var, cat_but_car, num_var = grab_variable(df)
print("Categoric Variables:", cat_var)

def category_summary(dataframe, variable, graph=False):
    print(pd.DataFrame({variable: dataframe[variable].value_counts(),
                        "ratio": dataframe[variable].value_counts() * 100 / len(dataframe)}))
    print("***********************")
    if graph:
        sns.countplot(data=dataframe, x=variable, color="pink")
        plt.grid()
        plt.title("Count Plot for " + variable)
        plt.xlabel(variable)
        plt.ylabel("Value Counts")
        plt.show()

for i in cat_var:
    category_summary(df, i, graph=True)

def numeric_summary(dataframe, num_var, plot=False):
    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    print(dataframe[num_var].describe(quantiles).T, end="\n\n\n")
    if plot:
        sns.histplot(data=dataframe, x=num_var, color="pink", bins=20)
        plt.grid()
        plt.title("Histogram For" + num_var)
        plt.xlabel(num_var)
        plt.show(blocks=True)

for i in num_var:
    numeric_summary(df, i, plot=True)

df.head()
df.corr()

print(df["Pregnancies"].describe().T)

df["Pregnancies_Segments"] = pd.qcut(df["Pregnancies"], q=4, labels=["D", "C", "B", "A"])
df["AGE_PREGNANCIES"] = df["Age"] / df["Pregnancies"]
df.replace([np.inf, -np.inf], np.nan, inplace=True)

df["Glucose"].describe().T  # 0,100,150,max
df["Insulin"].describe().T  # 0,100,250,max


df.loc[(df["Glucose"] < 100) & (df["Insulin"] < 100), "Glucose_Insulin"] = "low"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] < 150)) &
       ((df["Insulin"] >= 100) & (df["Insulin"] < 250)), "Glucose_Insulin"] = "medium"
df.loc[(df["Glucose"] >= 150) & (df["Insulin"] >= 250), "Glucose_Insulin"] = "high"
df.loc[(df["Glucose"] < 100) & ((df["Insulin"] >= 100) & (df["Insulin"] < 250)), "Glucose_Insulin"] = "low_medium"
df.loc[(df["Glucose"] < 100) & (df["Insulin"] >= 250), "Glucose_Insulin"] = "low_high"
df.loc[(df["Glucose"] >= 100) & ((df["Glucose"] < 150) & (df["Insulin"] < 100)), "Glucose_Insulin"] = "medium_low"
df.loc[(df["Glucose"] >= 100) & ((df["Glucose"] < 150) & (df["Insulin"] >= 250)), "Glucose_Insulin"] = "medium_high"
df.loc[(df["Glucose"] >= 150) & (df["Insulin"] < 100), "Glucose_Insulin"] = "high_low"
df.loc[(df["Glucose"] >= 150) & ((df["Insulin"] >= 100) & (df["Insulin"] < 250)), "Glucose_Insulin"] = "high_medium"

df["BMI"].describe().T
bins = [df["BMI"].min(), df["BMI"].describe().T[4],
          df["BMI"].describe().T[6], df["BMI"].max()]

df["BMI_SEGMENT"] = pd.cut(df["BMI"], bins=bins, labels=["Thin", "Normal", "Fat"])

cat_var, cat_but_car, num_var = grab_variable(df)
df.info()

for i in num_var:
    print(i, "<> SkinTickness",  df["SkinThickness"].corr(df[i]))
    print(i, "<> Age", df["Age"].corr(df[i]))
df[["SkinThickness", "BMI"]].head(30)

df["SkinThickness_*_BMI"] = df["SkinThickness"] * df["BMI"]


cat_var, cat_but_car, num_var = grab_variable(df)
print(df.isnull().sum())
deal_with_missings(df)

for i in num_var:
    print(i, check_outliers(df, i))

for i in num_var:
    if check_outliers(df, i) == True:
        suppress_outliers(df, i)
    print(i, check_outliers(df, i))

cat_var, cat_but_car, num_var = grab_variable(df)

def standardization(dataframe):
    print("Standardization Methods:\n1. StandardScaler\n2. RobustScaler\n3. MinMaxScaler")
    type = int(input("Enter the type of standardization"))
    for i in num_var:
        if type == 1:
            ss = StandardScaler()
            dataframe[i] = ss.fit_transform(dataframe[[i]])
        elif type == 2:
            rs = RobustScaler()
            dataframe[i] = rs.fit_transform(dataframe[[i]])
        elif type == 3:
            mms = MinMaxScaler()
            dataframe[i] = mms.fit_transform(dataframe[[i]])
        else:
            print("Enter teh correct type for numeric variable standardization")
            break

standardization(df)
df.head()


cat_var, cat_but_car, num_var = grab_variable(df)
def one_hot_encoder(dataframe, ohe_var, drop_f=True, dummy_na=False):
    dataframe = pd.get_dummies(data=dataframe, columns=ohe_var, dtype="int64",
                               drop_first=drop_f, dummy_na=dummy_na)
    return dataframe

ohe_var = [i for i in df.columns if 10 >= df[i].nunique() > 2]

df = one_hot_encoder(df, ohe_var)
df.head()
#########################
# 3. Modeling & Prediction

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

knn_model = KNeighborsClassifier().fit(X, y)


random_user = X.sample(10)
print("Random Users-->\n", random_user)

row = []
for i in random_user.index:
    if i in df.index:
        row.append(df["Outcome"].loc[i])
print("Real Outcomes\n", row)
print("Predicted Outcomes:\n", knn_model.predict(random_user))


# 4. Model Evaluation

y_pred = knn_model.predict(X)
y_prob = knn_model.predict_proba(X)[:, 1]
print("Real Outcomes:\n", y[0: 10])
print("Estimated Outcomes:\n", y_pred[0: 10])

print(" **** Classification Report ****\n", classification_report(y, y_pred))
print("Roc_AUC_score-->", roc_auc_score(y, y_prob))
# Accuracy: 0.82
# Precision: 0.76
# Recall: 0.69
# f1-score: 0.72
# roc_auc_score: 0.89

# Model Validation - Holdout:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
knn_model = KNeighborsClassifier().fit(X_train, y_train)

# train set
y_pred = knn_model.predict(X_train)
y_prob = knn_model.predict_proba(X_train)[:, 1]

print("*** Classification Report ***\n", classification_report(y_train, y_pred))
print("roc_auc-->", roc_auc_score(y_train, y_prob))
# Accuracy: 0.80
# Precision: 0.75
# Recall: 0.61
# f1-score: 0.67
# roc_auc_score: 0.87

# test set
y_pred = knn_model.predict(X_test)
y_prob = knn_model.predict_proba(X_test)[:, 1]
print("*** Classification Report ***\n", classification_report(y_test, y_pred))
print("roc_auc-->", roc_auc_score(y_test, y_prob))
# Accuracy: 0.73
# Precision: 0.73
# Recall: 0.58
# f1-score: 0.65
# roc_auc_score: 0.80


# Model Validation - K-FOLD CROSS VALIDATION:
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

knn_model = KNeighborsClassifier().fit(X, y)

cv_results = cross_validate(knn_model, X, y, cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

print("Accuracy-->", cv_results["test_accuracy"].mean())
print("Precision-->", cv_results["test_precision"].mean())
print("Recall-->", cv_results["test_recall"].mean())
print("score-->", cv_results["test_f1"].mean())
print("Roc_AUC_score-->", cv_results["test_roc_auc"].mean())
# Accuracy: 0.736
# Precision: 0.647
# Recall: 0.56
# f1-score: 0.595
# roc_auc_score: 0.768


####################################
# 5. Hyperparameter Optimization
knn_model = KNeighborsClassifier().fit(X, y)
print("Parameters:", knn_model.get_params())
# n_neighbors: 5 by default

parameters = {"n_neighbors": range(2, 40)}

knn_best_grid = GridSearchCV(knn_model,            # model
                             parameters,           # parameters
                             cv=5,                 # cross validate
                             n_jobs=-1,            # full performance
                             verbose=1).fit(X, y)  # report

print("Best Parameters:", knn_best_grid.best_params_)
print("Best Score (Accuracy):", knn_best_grid.best_score_)
# n_neighbors:9, score:0.74
# GridSearchCV calculates the best score as "accuracy" score by default
# we can change whenever we want the type of this score like precision, f1 and so on.

knn_best_grid2 = GridSearchCV(knn_model,
                              parameters,
                              cv=5,
                              n_jobs=-1,
                              verbose=True,
                              scoring="roc_auc").fit(X, y)
print("Best Parameters:", knn_best_grid2.best_params_)
print("Best Score (roc_auc):", knn_best_grid2.best_score_)
# n_neighbors:30, score:0.79


####################################
# 6. Final Model
knn_final_model = knn_model.set_params(**knn_best_grid.best_params_).fit(X, y)

cv_results = cross_validate(knn_final_model, X, y, cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

print("Accuracy-->", cv_results["test_accuracy"].mean())
print("Precision-->", cv_results["test_precision"].mean())
print("Recall-->", cv_results["test_recall"].mean())
print("f1-->", cv_results["test_f1"].mean())
print("roc_auc-->", cv_results["test_roc_auc"].mean())
# Accuracy: 0.74
# Precision: 0.65
# Recall: 0.571
# f1-score: 0.606
# roc_auc_score: 0.79

random_user = X.sample(10)
row = []
for i in random_user.index:
    if i in df.index:
        row.append(df["Outcome"].loc[i])
print("Real Outcomes\n", row)
print("Predicted Outcomes:\n", knn_model.predict(random_user))