import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split

pd.set_option("display.width", 700)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
df = pd.read_csv("ML Datasets/advertising.csv")

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
                   and dataframe[i].nunique(9 > car_th)]
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


# Outliers
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

# Checking Outlier Thresholds
for i in num_var:
    print(i, threshold_outliers(df, i))

# Checking Outliers
for i in num_var:
    print(i, check_outliers(df, i))

# Missing Values
print(df.isnull().sum())

# Modelling
y = df[["sales"]]               # Dependent Variable
X = df.drop("sales", axis=1)    # Independent Variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

mlr_model = LinearRegression().fit(X_train, y_train)

# Prediction
# yi^ = b + w * x
# yi^ --> Predicted Value
# b --> intercept
# w --> weight, coefficient
# x --> Independent variable value

print("Bias (b)-->", mlr_model.intercept_[0])
print("Weight/Coefficient (w)-->", mlr_model.coef_[0])

# What is the prediction salary when,
# TV: 30
# Radio: 10
# Newspaper: 40
print(mlr_model.intercept_[0] + mlr_model.coef_[0][0] * 30 + mlr_model.coef_[0][1] * 10 + mlr_model.coef_[0][2] * 40)

def prediction(model):
    tv = int(input("Enter Expenditure for TV"))
    radio = int(input("Enter Expenditure for radio"))
    newspaper = int(input("Enter Expenditure for newspaper"))
    guess = model.intercept_[0] + model.coef_[0][0] * tv + model.coef_[0][1] * radio + model.coef_[0][2] * newspaper
    print("TV: {}\nRadio: {}\nNewspaper: {}\n Expected Sales: {}".format(tv, radio, newspaper, guess))

prediction(mlr_model)

# Prediction Success for Train Set
y_predict = mlr_model.predict(X_train)

print("MSE-->", mean_squared_error(y_train, y_predict))
print("RMSE-->", np.sqrt(mean_squared_error(y_train, y_predict)))
print("MAE-->", mean_absolute_error(y_train, y_predict))
print("F1_SCORE (R Square)-->", mlr_model.score(X_train, y_train))

# Prediction Success for Test Set
y_predict = mlr_model.predict(X_test)

print("MSE-->", mean_squared_error(y_test, y_predict))
print("RMSE-->", np.sqrt(mean_squared_error(y_test, y_predict)))
print("MAE-->", mean_absolute_error(y_test, y_predict))
print("F1_SCORE (R Square)-->", mlr_model.score(X_train, y_train))

# 10 Fold Cross Validation Root Mean Squared Error (CV RMSE)
print("10 Fold Cross Validation:", np.mean(np.sqrt(-cross_val_score(mlr_model,
                                                                    X, y,
                                                                    cv=10,
                                                                    scoring="neg_mean_squared_error"))))

def K_Fold_Corss_Validation(model, indep_var, dep_var, cv=10):
    return np.mean(np.sqrt(-cross_val_score(model,
                                            indep_var, dep_var,
                                            cv=cv,
                                            scoring="neg_mean_squared_error")))

K_Fold_Corss_Validation(mlr_model, X, y)
