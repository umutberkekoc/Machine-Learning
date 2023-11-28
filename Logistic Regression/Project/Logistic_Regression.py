import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, RocCurveDisplay, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

# 1. Exploratory Data Analysis
# 2. Data Pre-processing
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Model Valdiation: Holdout
# 6. Model Valdiation: 10-Fold Cross Validation
# 7. Prediction for A New Observation

pd.set_option("display.width", 700)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

df = pd.read_csv("ML Datasets/diabetes.csv")

##############################
# 1. Exploratory Data Analysis
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

print(df["Outcome"].value_counts())

sns.countplot(data=df, x="Outcome")
plt.grid()
plt.xlabel("Outcome")
plt.title("Number of People by Outcome")
print(plt.show())

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

def hist_for_num_var(dataframe, variable, bins=20):
    import random
    color_list = ["red", "blue", "green", "pink", "purple", "orange"]
    sns.histplot(data=dataframe, x=variable, bins=bins, color=random.choice(color_list))
    plt.grid()
    plt.title("Histogram For " + variable)
    plt.xlabel(variable)
    plt.show(block=True)

for i in num_var:
    hist_for_num_var(df, i)

def target_and_features(dataframe, target, variable):
    print(dataframe.groupby(target).agg({variable: ["mean", "median", "min", "max"]}), end="\n\n\n")

for i in num_var:
    target_and_features(df, "Outcome", i)


########################################
# 2. Data Pre-processing

def threshold_outliers(dataframe, variable, q1=0.05, q3=0.95):
    q1 = dataframe[variable].quantile(q1)
    q3 = dataframe[variable].quantile(q3)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    return lower_limit, upper_limit

def check_outliers(dataframe, variable):
    lower_limit, upper_limit = threshold_outliers(dataframe, variable)
    if dataframe[(dataframe[variable] < lower_limit) | (dataframe[variable] > upper_limit)].shape[0] > 0:
        return True
    else:
        return False

def change_outliers_with_th(dataframe, variable):
    lower_limit, upper_limit = threshold_outliers(dataframe, variable)
    dataframe.loc[(dataframe[variable] < lower_limit), variable] = lower_limit
    dataframe.loc[(dataframe[variable] > upper_limit), variable] = upper_limit
    print("Outliers Changed With Their Thresholds")

def remove_outliers(dataframe, variable):
    lower_limit, upper_limit = threshold_outliers(dataframe, variable)
    dataframe = dataframe[~((dataframe[variable] < lower_limit) | (dataframe[variable] > upper_limit))]
    return dataframe

def show_outliers(dataframe, variable, head=5):
    lower_limit, upper_limit = threshold_outliers(dataframe, variable)
    if dataframe[(dataframe[variable] < lower_limit) | (dataframe[variable] > upper_limit)].shape[0] > 0:
        print("Number of Outlier Observed-->", dataframe[(dataframe[variable] < lower_limit) |
                                                         (dataframe[variable] > upper_limit)].shape[0])
        print(dataframe[(dataframe[variable] < lower_limit) |
                        (dataframe[variable] > upper_limit)].head(head))

# Check Outliers Threshold
for i in num_var:
    print(i, threshold_outliers(df, i))

# Check Outliers
for i in num_var:
    print(i, check_outliers(df, i))

#boxplot shows outliers according to 25 % and 75 % quantiles
'''def box_plot(dataframe, variable):
    sns.boxplot(data=dataframe, x=variable, color="green")
    plt.title("BoxPlot for {}".format(variable))
    plt.show()
for i in num_var:
    box_plot(df, i)'''

for i in num_var:
    show_outliers(df, i)

# Suppress
change_outliers_with_th(df, "Insulin")

# Check Outliers
for i in num_var:
    print(i, check_outliers(df, i))


# Missing Values:
print(df.isnull().sum())
print(df.describe().T)

# It seems missing values are filled by zeros because it is not possible to become the minimum value of
# some of the variables like Glucose, Insulin, BloodPressure, BMI and SkinThickness.

zero_columns = [i for i in df.columns if df[i].min() == 0 and i in num_var]
zero_columns = [i for i in zero_columns if "Pregnancies" not in i]

# changing zeros with NaN
for i in zero_columns:
    df[i].replace(0, np.nan, inplace=True)
print(df.isnull().sum())

def missing_values_table(dataframe):
    miss_var = [i for i in dataframe.columns if dataframe[i].isnull().sum() > 0]
    miss_num = dataframe[miss_var].isnull().sum().sort_values(ascending=False)
    ratio = ((100 * dataframe[miss_var].isnull().sum().sort_values(ascending=False) / len(dataframe)).
             sort_values(ascending=False))
    dataframe = pd.concat([miss_num, ratio], keys=["n_miss", "ratio"], axis=1)
    return dataframe

missing_values_table(df)

# Filling NaN with mean
for i in zero_columns:
    df[i].fillna(df[i].mean(), inplace=True)
print(df.isnull().sum())

for i in num_var:
    print(i, check_outliers(df, i))  # after filling nan values, we should resuppress the outliers

for i in num_var:
    change_outliers_with_th(df, i)
    if check_outliers(df, i) == True:
        print("There is still outliers in:" + i)

# Scale numeric variables
def standardization(dataframe):
    print("Standardization Methods:\n1. StandardScaler\n2.RobustScaler\n3. MinMaxScaler")
    type = int(input("Enter the type of standardization"))
    ask = str(input("Do you want to create new variable, or save changes on the current variable? (yes/no)"))
    if ask == "yes":
        for i in num_var:
            if type == 1:
                ss = StandardScaler()
                dataframe["Scaled_" + i] = ss.fit_transform(dataframe[[i]])
            elif type == 2:
                rs = RobustScaler()
                dataframe["Scaled_" + i] = rs.fit_transform(dataframe[[i]])
            elif type == 3:
                mms = MinMaxScaler()
                dataframe["Scaled_" + i] = mms.fit_transform(dataframe[[i]])
            else:
                print("Enter teh correct type for numeric variable standardization")
                break
    else:
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

########################################
# 3. Model & Prediction

y = df["Outcome"]               # Dependent Variable
X = df.drop("Outcome", axis=1)  # Independent Variable
X.head()

log_model = LogisticRegression().fit(X, y)

print("Intercept (b)-->\t", log_model.intercept_[0])
print("Coefficient / weight (w)-->\t", log_model.coef_[0])

y_predict = log_model.predict(X)

print("*** Real Outcomes ***\n", y[0: 10])
print("*** Estimated Outcomes ***\n", y_predict[0: 10])


########################################
# 4. Model Evaluation
def confusion_matrix_plot(real, guess):
    acc = round(accuracy_score(real, guess), 2)
    cm = confusion_matrix(real, guess)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.title("HeatMap\nAccuracy Score: {}".format(acc))
    plt.xlabel("Predicted Outcomes (y_predict)")
    plt.ylabel("Real Outcomes (y")
    plt.show()
confusion_matrix_plot(y, y_predict)

print("**** Classification Report ****\n", classification_report(y, y_predict))
# Accuracy: 0.78
# Precision: 0.73
# Recall: 0.57
# F1_score: 0.64

y_prob = log_model.predict_proba(X)[:, 1]
print("Roc AUC Score-->", roc_auc_score(y, y_prob))
# Roc AUC Score: 0.845


########################################
# 5. Model Validation - Holdout

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)
print("y_train Shape:", y_train.shape)
print("y_test Shape:", y_test.shape)

log_model = LogisticRegression().fit(X_train, y_train)
print("İntercept (b)-->\t", log_model.intercept_[0])
print("Coefficient / weight (w)\n", log_model.coef_[0])


y_predict = log_model.predict(X_test)
print("Real Outcomes:\n", y[0: 10])
print("Predicted Outcomes:\n", y_predict[0: 10])

def confusion_matrix_plot(real, predict):
    acc = round(accuracy_score(real, predict), 2)
    cm = confusion_matrix(real, predict)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.title("HeatMap\nAccuracy Score: {}".format(acc))
    plt.xlabel("Predicted Outcomes (y)")
    plt.ylabel("Real Outcomes (y_test)")
    plt.show()

confusion_matrix_plot(y_test, y_predict)
print(" *** Classification Report ***\n", classification_report(y_test, y_predict))
# Accuracy: 0.77
# Precision: 0.81
# Recall: 0.51
# F1_score: 0.62

# ROC-AUC Score
y_prob = log_model.predict_proba(X_test)[:, 1]
print("Roc AUC Score-->", roc_auc_score(y_test, y_prob))
# Roc AUC Score: 0.865

RocCurveDisplay.from_estimator(log_model, X_test, y_test)
plt.title("ROC Curve")
plt.plot([0, 1], [0, 1], "r--")
plt.show()


########################################
# 6. Model Validation - K- Fold Cross Validation
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
log_model = LogisticRegression().fit(X, y)

cv_result = cross_validate(log_model, X, y, cv=5,
                           scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

print(cv_result["test_accuracy"].mean())
print(cv_result["test_precision"].mean())
print(cv_result["test_recall"].mean())
print(cv_result["test_f1"].mean())
print(cv_result["test_roc_auc"].mean())
# Accuracy: 0.77
# Precision: 0.726
# Recall: 0.563
# F1_score: 0.633
# roc_auc: 0.838

cv_result2 = cross_validate(log_model, X, y, cv=10,
                           scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

print(cv_result2["test_accuracy"].mean())
print(cv_result2["test_precision"].mean())
print(cv_result2["test_recall"].mean())
print(cv_result2["test_f1"].mean())
print(cv_result2["test_roc_auc"].mean())
# Accuracy: 0.77
# Precision: 0.727
# Recall: 0.556
# F1_score: 0.628
# roc_auc: 0.837

########################################
# 7. Prediction for A New Observation
random_user = X.sample(10)
log_model.predict(random_user)

row = []
for i in random_user.index:
    if i in df.index:
        row.append(df["Outcome"].loc[i])
print("Real Outcomes:\n", row)

print("Predicted Outcomes:\n", log_model.predict(random_user))