##################################
# CART (Classification & Regression Tree)
import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from skompiler import skompile
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 600)
warnings.simplefilter(action="ignore", category=Warning)

# 1. Exploratory Data Analysis
# 2. Feature Engineering & Data Preprocessing
# 3. Modeling using CART
# 4. Hyperparameter optimization with GridSearchCV
# 5. Final Model
# 6. Feature Importance
# 7. Analyzing model complexity with Learning Curves (BONUS)
# 8. Visualizing the decision tree
# 9. Extracting decision rules
# 10. Extracting PYTHON/SQL/EXCEL codes of decision rules
# 11. Prediction using Python codes
# 12. Saving and loading model.
df = pd.read_csv("ML Datasets/diabetes.csv")

##############################
# 1. Exploratory Data Analysis
def show_info(dataframe):
    print(" *** HEAD ***")
    print(dataframe.head())
    print(" *** TAIL ***")
    print(dataframe.tail())
    print(" *** SHAPE ***")
    print(dataframe.shape)
    print(" *** COLUMNS ***")
    print(dataframe.columns)
    print(" *** NA ***")
    print(dataframe.isnull().sum())
    print(" *** DESCRIPTIVE STATISTICS ***")
    print(dataframe.describe().T)

show_info(df)

def grab_variables(dataframe, num_th=10, car_th=20):
    # Categoric Variables #
    cat_var = [i for i in dataframe.columns if dataframe[i].dtypes in ["object", "category", "bool"]]
    num_but_cat = [i for i in dataframe.columns if dataframe[i].dtypes in ["int64", "float64"]
                   and dataframe[i].nunique() <= num_th]
    cat_but_car = [i for i in dataframe.columns if dataframe[i].dtypes in ["object", "category"]
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
    print("Number of Cat but Car Variables:", len(cat_but_car))
    print("Number of Numeric Variables:", len(num_var))
    return cat_var, cat_but_car, num_var

grab_variables(df)
cat_var, cat_but_car, num_var = grab_variables(df)

#############################################
# 2. Feature Engineering & Data Preprocessing

def cat_summary(dataframe, variable, plot=False):
    print(pd.DataFrame({variable: dataframe[variable].value_counts(),
                        "Ratio": 100 * dataframe[variable].value_counts() / len(dataframe)}))
    print("***********************")
    if plot:
        sns.countplot(data=dataframe, x=dataframe[variable])
        plt.title("Number of " + variable)
        plt.xlabel(variable)
        plt.grid()
        print(plt.show(block=True))

for i in cat_var:
    cat_summary(df, i, plot=True)

def num_summary(dataframe, variable, plot=False):
    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    print(dataframe[variable].describe(quantiles).T, end="\n\n\n")
    if plot:
        sns.histplot(data=dataframe, x=variable, bins=20, color="green")
        plt.xlabel(variable)
        plt.title("Histogram for " + variable)
        plt.grid()
        plt.show(block=True)
for i in num_var:
    print(num_summary(df, i, plot=True))


# outliers
def outlier_thresholds(dataframe, variable):
    q1 = dataframe[variable].quantile(0.05)
    q3 = dataframe[variable].quantile(0.95)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    return lower_limit, upper_limit

def check_outlier(dataframe, variable):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable] < lower_limit) | (dataframe[variable] > upper_limit)].shape[0] > 0:
        return True
    else:
        return False

def remove_outliers(dataframe, variable):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    df_without_outlier = dataframe[~(dataframe[variable] < lower_limit) | (dataframe[variable] > upper_limit)]
    return df_without_outlier

def suppress_outliers(dataframe, variable):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < lower_limit), variable] = lower_limit
    dataframe.loc[(dataframe[variable] > upper_limit), variable] = upper_limit
    print(variable + " Suppressed")

for i in num_var:
    print(i, outlier_thresholds(df, i))

for i in num_var:
    print(i, check_outlier(df, i))

for i in num_var:
    if check_outlier(df, i) == True:
        suppress_outliers(df, i)
        print(i, check_outlier(df, i))


# missing
print(df.isnull().sum())
print(df.describe().T)

zero_columns = [i for i in df.columns if df[i].min() == 0
                and i not in ["Pregnancies", "Outcome"]]
print(zero_columns)

for i in zero_columns:
    df[i].replace(0, np.nan, inplace=True)

for i in zero_columns:
    if i in num_var:
        df[i].fillna(df[i].median(), inplace=True)
    elif i in cat_var:
        df[i].fillna(df[i].mode()[0], inplace=True)
    else:
        pass


# Encoding
def label_encoder(dataframe, bin_var):
    label_encoer = LabelEncoder()
    dataframe[bin_var] = label_encoer.fit_transform(dataframe[bin_var])
    print("0, 1-->", label_encoer.inverse_transform([0, 1]))

bin_var = [i for i in df.columns if df[i].nunique() == 2
           and i in cat_var]

for i in bin_var:
    label_encoder(df, i)

def one_hot_encoder(dataframe, ohe_var, drop_f=True, dummy=False):
    dataframe = pd.get_dummies(data=dataframe, columns=ohe_var,
                   drop_first=drop_f, dummy_na=dummy, dtype="int64")
    return dataframe

ohe_var = [i for i in df.columns if 10 >= df[i].nunique() > 2]
df = one_hot_encoder(df, ohe_var)

# scaling (Standard/Robust/MinMax)
for i in num_var:
    rs = RobustScaler()
    df[i] = rs.fit_transform(df[[i]])

df.head()

# creating new variables
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

df.head()

df["BMI"].describe().T

bins = [df["BMI"].min(), df["BMI"].describe().T[4],
          df["BMI"].describe().T[6], df["BMI"].max()]

df["BMI_Segment"] = pd.cut(df["BMI"], bins=bins, labels=["Thin", "Normal", "Fat"])

cat_var, cat_but_car, num_var = grab_variables(df)
df.info()

for i in num_var:
    print(i, "<> SkinTickness",  df["SkinThickness"].corr(df[i]))
    print(i, "<> Age", df["Age"].corr(df[i]))
df[["SkinThickness", "BMI"]].head(30)

df["SkinThickness_*_BMI"] = df["SkinThickness"] * df["BMI"]
cat_var, cat_but_car, num_var = grab_variables(df)
print(df.isnull().sum())

for i in num_var:
    print(i, check_outlier(df, i))

for i in num_var:
    if check_outlier(df, i) == True:
        suppress_outliers(df, i)
    print(i, check_outlier(df, i))

cat_var, cat_but_car, num_var = grab_variables(df)
df.head()


bin_var = [i for i in df.columns if df[i].nunique() == 2
           and i in cat_var]

for i in bin_var:
    label_encoder(df, i)


ohe_var = [i for i in df.columns if 10 >= df[i].nunique() > 2]
df = one_hot_encoder(df, ohe_var)

cat_var, cat_but_car, num_var = grab_variables(df)
for i in num_var:
    print(i, check_outlier(df, i))

print(df.isnull().sum())

df["AGE_PREGNANCIES"].fillna(df["AGE_PREGNANCIES"].median(), inplace=True)
df.head()

#######################
# 3. Modeling using CART
X = df.drop(["Outcome"], axis=1)  # Indepednent Variable
y = df["Outcome"]                 # Dependent Variable

cart_model = DecisionTreeClassifier(random_state=123456).fit(X, y)
y_pred = cart_model.predict(X)
y_prob = cart_model.predict_proba(X)[:, 1]

print("Classification Report\n", classification_report(y, y_pred))
print("roc_auc_score-->", roc_auc_score(y, y_prob))
# Accuracy: 1
# Precision: 1
# Recall: 1
# f1: 1
# roc_auc: 1
# (Overfitting!)

# Model Evaluation with Holdout Method
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=123456)
cart_model = DecisionTreeClassifier().fit(X_train, y_train)

# <train set>
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print("Classification Report\n", classification_report(y_train, y_pred))
print("roc_auc_score-->", roc_auc_score(y_train, y_prob))
# Accuracy: 1
# Precision: 1
# Recall: 1
# f1: 1
# roc_auc: 1
# ! warning ! overfitting happened

# <test set>
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print("Classification Report\n", classification_report(y_test, y_pred))
print("ROC_AUC_SCORE-->", roc_auc_score(y_test, y_prob))
# Accuracy: 0.73
# Precision: 0.64
# Recall: 0.60
# f1: 0.62
# roc_auc: 0.702

# Model Evaluation with Cross Validation (CV)
cart_model = DecisionTreeClassifier(random_state=123456).fit(X, y)

cv_results = cross_validate(cart_model,     # Model
                            X, y,           # Independent & Dependent Variables
                            cv=5,           # Cross Validate Value
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

print("Accuracy:", round(cv_results["test_accuracy"].mean(), 4))
print("Precision:", cv_results["test_precision"].mean())
print("Recall:", cv_results["test_recall"].mean())
print("f1:", round(cv_results["test_f1"].mean(), 4))
print("roc_auc:", round(cv_results["test_roc_auc"].mean(), 4))
# Accuracy: 0.705
# Precision: 0.584
# Recall: 0.541
# f1: 0.561
# roc_auc: 0.667

#################################################
# 4. Hyperparameter optimization with GridSearchCV
cart_model = DecisionTreeClassifier(random_state=123456).fit(X, y)
print("Model Parameters:\n", cart_model.get_params())
# ! max_depth: None, min_samples_split: 2 by default

parameters = {"max_depth": range(1, 12),
              "min_samples_split": range(1, 20)}

model_best_grid = GridSearchCV(cart_model,  # Model
                               parameters,  # Model parameters
                               cv=5,        # Cross validate Value
                               n_jobs=-1,   # ful performance of Processor
                               verbose=1).fit(X, y)  # report

print("Best Parameters:\t", model_best_grid.best_params_)
print("Best Score (Accuracy):\t", model_best_grid.best_score_)
# max_depth: 5, min_samples_split: 4
# Accuracy: 0.746
# by defaultly, GridSearchCV calculates the accuracy score as the name
# of the best score. We can change the score type as;

model_best_grid2 = GridSearchCV(cart_model,
                                parameters,
                                cv=5,
                                n_jobs=-1,
                                verbose=True,
                                scoring="roc_auc").fit(X, y)
print("Best Parameters:\t", model_best_grid2.best_params_)
print("Best Score (roc_auc):\t", model_best_grid2.best_score_)
# max_depth: 5, min_samples_split: 5
# roc_auc: 0.783

random = X.sample(1)
print("Random Patient\n", random)
print("Prediction Outcome for Random Patient:\t", model_best_grid.predict(random))
# we can predict the Outcome value for random patient using by the variable which we created
# and named as "model_best_grid". model best grid variable contains the best / optimal
# model with the best parameters thanks to GridSearchCV function.
# But although, we should create the final model.


###############
# 5. Final Model
cart_final_model = cart_model.set_params(**model_best_grid.best_params_).fit(X, y)
print("Final Model Parameters\n", cart_final_model.get_params())  # 1. way (preferred)
cart_final_model = DecisionTreeClassifier(**model_best_grid.best_params_).fit(X, y)
print("Final Model Parameters\n", cart_final_model.get_params())  # 2. way

cv_results = cross_validate(cart_final_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

print("Accuracy:", round(cv_results["test_accuracy"].mean(), 4))
print("Precision:", cv_results["test_precision"].mean())
print("Recall:", cv_results["test_recall"].mean())
print("f1:", round(cv_results["test_f1"].mean(), 4))
print("roc_auc:", round(cv_results["test_roc_auc"].mean(), 4))
# Accuracy: 0.76
# Precision: 0.641
# Recall: 0.631
# f1: 0.634
# roc_auc: 0.782

######################
# 6. Feature Importance
print("Feature Importances\n", cart_final_model.feature_importances_)
print("Feature Names\n", cart_final_model.feature_names_in_)
print("Number of Features-->", cart_final_model.n_features_in_)

def feature_importances_graph_creater(model, num=len(X), save=False):
    feat_imp = pd.DataFrame({"Value": model.feature_importances_,
                             "Name": model.feature_names_in_})
                           # "Name": X.columns  # 2.way
    plt.figure(figsize=(12, 10))
    sns.barplot(x="Value", y="Name", data=feat_imp.sort_values("Value", ascending=False)[0: num])
    plt.title("Feature Importances Bar Graph")
    plt.yticks(rotation=45)
    plt.grid()
    plt.show()
    if save:
        plt.savefig("features.png")

feature_importances_graph_creater(cart_final_model, num=5)


############################################################
# 7. Analyzing model complexity with Learning Curves (BONUS)

train_score, test_score = validation_curve(cart_final_model,
                                           X, y,
                                           cv=5,
                                           param_name="max_depth",
                                           param_range=range(1, 11),
                                           scoring="roc_auc")

mean_train_score = np.mean(train_score, axis=1)
mean_test_score = np.mean(test_score, axis=1)
print("Train Score Mean-->", mean_train_score)
print("Test Score Mean-->", mean_test_score)

plt.plot(mean_train_score,
         label="Training Score", color="blue")

plt.plot(mean_test_score,
         label="Test / Validation Score", color="red")

plt.title("Validation Curve for max_depth")
plt.xlabel("Number of max_depth")
plt.ylabel("ROC_AUC")
plt.grid()
plt.legend(loc="best")
plt.show()
model_best_grid.best_params_

train_score2, test_score2 = validation_curve(cart_final_model, X, y,
                                           cv=5,
                                           param_name="min_samples_split",
                                           param_range=range(2, 20),
                                           scoring="roc_auc")
average_train_score2 = np.mean(train_score2, axis=1)
average_test_score2 = np.mean(test_score2, axis=1)
print("Train Score Mean-->", mean_train_score)
print("Test Score Mean-->", mean_test_score)

plt.plot(average_train_score2, label="Training Score", color="blue")
plt.plot(average_test_score2, label="Validate / Test Score", color="red")
plt.title("Valdiation Curve fot min_samples_split")
plt.xlabel("Number of min_samples_split")
plt.ylabel("ROC_AUC")
plt.legend(loc="best")
plt.grid()
plt.show()

# grafikte max_depth 3 olmalı gibi gözüküyor çünkü 3'ten sonra train ve test set başarı oranları (hataları)
# 3 den sonra ayrışmaya başlıyor ancak biz best_params kısmında max_depth değerini 5 bulduk. önemli olan birbirlerine
# yakın değerlermi, çünkü biz best_params_ metodu ile max_depth ve min_samples_split arasındaki ilişkiye göre bulduk


def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(model, X, y,
                                               param_name=param_name,
                                               param_range=param_range,
                                               scoring=scoring,
                                               cv=cv)
    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color="blue")
    plt.plot(param_range, mean_test_score,
             label="Validation / Test Score", color="green")

    plt.title("Validation Curve for " + type(model).__name__)
    plt.xlabel(" Number of " + param_name)
    plt.ylabel(scoring.upper())
    plt.tight_layout()
    plt.legend(loc="best")
    plt.grid()
    plt.show()

val_curve_params(cart_final_model, X, y, "max_depth", range(1, 11))

cart_val_params = [["max_depth", range(1, 11)], ["min_samples_split", range(2, 20)]]

for i in range(len(cart_val_params)):
    val_curve_params(cart_model, X, y, cart_val_params[i][0],
                     cart_val_params[i][1])


##################################
# 8. Visualizing the decision tree
import graphviz
def tree_graph(model, variable, file_name):
    tree_str = export_graphviz(model, feature_names=variable, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)
tree_graph(model=cart_final_model, variable=X.columns, file_name="cart_final.png")

##############################
# 9. Extracting decision rules
tree_rules = export_text(cart_final_model, feature_names=list(X.columns))
print("*** Tree Rules ***\n", tree_rules)

########################################################
# 10. Extracting PYTHON/EXCEL/SQL codes of decision rules
print(skompile(cart_final_model.predict).to("python/code"))  # for python
print(skompile(cart_final_model.predict).to("excel"))        # for excel
print(skompile(cart_final_model.predict).to("sqlalchemy/sqlite"))  # for sql

###########################################
# 11. Prediction with using the Python codes
def prediction(x):
    return (((((0 if x[0] <= 0.800000011920929 else 1) if x[5] <= -0.14835165441036224
     else 0 if x[6] <= 0.3346405327320099 else 0) if x[5] <=
    1.4395604729652405 else 0 if x[7] <= -0.38235294818878174 else 1) if x[
    7] <= -0.029411764815449715 else 0 if x[5] <= -0.6538461744785309 else
    (0 if x[6] <= 1.1071895360946655 else 0) if x[1] <=
    -0.43209876120090485 else 0 if x[4] <= -2.26086962223053 else 1) if x[1
    ] <= 0.25925925374031067 else (((0 if x[8] <= 7.2058820724487305 else 0
    ) if x[11] <= 0.5 else 0 if x[5] <= -0.7472527623176575 else 1) if x[1] <=
    0.7037037014961243 else 0 if x[7] <= -0.20588235929608345 else 1 if x[7
    ] <= 1.8823529481887817 else 0) if x[5] <= -0.25824177265167236 else ((
    1 if x[2] <= 0.0625 else 0) if x[7] <= 0.08823529444634914 else 1 if x[
    6] <= 0.14901961013674736 else 1) if x[1] <= 1.0 else 0 if x[4] <=
    -7.826086759567261 else 1 if x[5] <= 1.5164835453033447 else 0)

x = [2, 50, 20, 23, 4, 35, 11, 7]
print(prediction(x))
















