import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor

train_set = pd.read_csv("titanic_train.csv")

pd.set_option("display.width", 700)
pd.set_option("display.max_columns", None)

df = pd.read_csv("Housing.csv")
df.head()

df = pd.get_dummies(data=df, columns=[i for i in df.columns if df[i].dtype in ["object", "category"]
                                      and df[i].nunique() > 1], drop_first=True, dtype=int)


# Model
X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=45)

rf_model = RandomForestRegressor(random_state=45).fit(X_train, y_train)

# Test:
y_pred = rf_model.predict(X_test)
print("MSE", mean_squared_error(y_test, y_pred))
print("RMSE", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE", mean_absolute_error(y_test, y_pred))
# MSE: 1348253932696
# RMSE: 1161143
# MAE: 865005

# Train:
y_pred_train = rf_model.predict(X_train)
print("MSE", mean_squared_error(y_train, y_pred_train))
print("RMSE", np.sqrt(mean_squared_error(y_train, y_pred_train)))
print("MAE", mean_absolute_error(y_train, y_pred_train))
# MSE: 195685232106
# RMSE: 442363
# MAE: 313024

# Hiperparametre Optimizasyonu
parameters = {"max_depth": [3, 5, 8, None],
              "max_features": [3, 5, 7, "sqrt"],
              "min_samples_split": [2, 5, 8, 15, 20],
              "n_estimators": [100, 200, 500]}

rf_model2 = RandomForestRegressor()

rf_model_CV = GridSearchCV(rf_model2, parameters, cv=5, n_jobs=-1).fit(X_train, y_train)

print("Best Params:", rf_model_CV.best_params_)
# Best Params: {'max_depth': 8, 'max_features': sqrt, '
# min_samples_split': 5, 'n_estimators': 200}

RF_REG_final = rf_model2.set_params(**rf_model_CV.best_params_).fit(X_train, y_train)

# Test
y_pred = RF_REG_final.predict(X_test)
print("******RMSE******\n", np.sqrt(mean_squared_error(y_test, y_pred)))
print("******MAE******\n", mean_absolute_error(y_test, y_pred))
# RMSE: 1086179
# MAE: 796160

# Train:
y_pred_train = RF_REG_final.predict(X_train)
print("******RMSE******\n", np.sqrt(mean_squared_error(y_train, y_pred_train)))
print("******MAE******\n", mean_absolute_error(y_train, y_pred_train))
# RMSE: 754413
# MAE: 555404
