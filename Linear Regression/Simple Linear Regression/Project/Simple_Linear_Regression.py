import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


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


for i in num_var:
    print(i, threshold_outliers(df, i))

for i in num_var:
    print(i, check_outliers(df, i))



# Missing Values
print(df.isnull().sum())


# Modelling

y = df[["sales"]]                                         # Dependent Variable
X = df[[str(input("enter: tv, radio or newspaper"))]]     # Independent Variable

slr_model = LinearRegression().fit(X, y)

y_predict = slr_model.predict(X)

print("Real Values-->\n", y[0: 10])
print("Predictions-->\n", y_predict[0: 10])

print("Bias (b)-->", slr_model.intercept_[0][0])
print("Weight (w)-->", slr_model.coef_[0][0])

# Prediction
# yi^ = b + w * x
# yi^ --> Predicted Value
# b --> intercept
# w --> weight, coefficient
# x --> Independent variable value

# what is the prediction of salary when the expenditure for newspaper is 60
print(slr_model.intercept_[0] + slr_model.coef_[0][0] * 60)
# what is the prediction of salary when the expenditure for newspaper is 100
print(slr_model.intercept_[0] + slr_model.coef_[0][0] * 100)

def prediction_salary(model):
    expenditure = int(input("Enter the Expenditure"))
    prediction = model.intercept_[0] + model.coef_[0][0] * expenditure
    print("Expenditure = {}\tPrediction = {}".format(expenditure, prediction))

prediction_salary(slr_model)

# Visualization of the Model
sns.regplot(data=df, x=X, y=y, color="red", ci=False, scatter_kws={"color": "blue", "s": 10})
plt.title("Model Equation: Sales = {} + Newspaper * {}".format(round(slr_model.intercept_[0], 2),
                                                        round(slr_model.coef_[0][0], 2)))
plt.xlabel("Newspaper Expenditure")
plt.ylabel("Sales")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
print(plt.show())

###################################
# Prediction Success

print("MSE-->", mean_squared_error(y_predict, y))
print("RMSE-->", np.sqrt(mean_squared_error(y_predict, y)))
print("MAE-->", mean_absolute_error(y_predict, y))
print("F1_SCORE (R Square) -->", slr_model.score(X, y))
