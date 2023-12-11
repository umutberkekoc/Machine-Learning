# CatBoost
##########################
import warnings
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from catboost import CatBoostClassifier

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
warnings.simplefilter(action="ignore", category=Warning)

df = pd.read_csv("ML Datasets/diabetes.csv")

X = df.drop(["Outcome"], axis=1)  # Independent Variables
y = df["Outcome"]                 # Dependent Variable
df.head()

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

cv_results = cross_validate(catboost_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

print("Accuracy:", round(cv_results["test_accuracy"].mean(), 3))
print("Precision:", round(cv_results["test_precision"].mean(), 3))
print("Recall:", round(cv_results["test_recall"].mean(), 3))
print("f1:", round(cv_results["test_f1"].mean(), 3))
print("roc_auc:", round(cv_results["test_roc_auc"].mean(), 3))
# Accuracy: 0.774
# Preciison: 0.708
# Recall: 0.605
# f1: 0.65
# roc_auc: 0.838

print("Parameters:", catboost_model.get_params())
# Parameters by default;

parameters = {"iterations": [200, 300, 500],
              "learning_rate": [0.01, 0.1],
              "depth": [3, 6]}

best_grid_model = GridSearchCV(catboost_model,
                               parameters,
                               cv=5,
                               n_jobs=-1,
                               verbose=True).fit(X, y)

print("Best Parameters:", best_grid_model.best_params_)
print("Best Accuracy Score:", best_grid_model.best_score_)
# depth: 3
# iterations: 500
# learning_rate: 0.01
# ebst acc: 0.772


catboost_final_model = catboost_model.set_params(**best_grid_model.best_params_, random_state=17).fit(X, y)
cv_results = cross_validate(catboost_final_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

print("Accuracy:", round(cv_results["test_accuracy"].mean(), 3))
print("Precision:", round(cv_results["test_precision"].mean(), 3))
print("Recall:", round(cv_results["test_recall"].mean(), 3))
print("f1:", round(cv_results["test_f1"].mean(), 3))
print("roc_auc:", round(cv_results["test_roc_auc"].mean(), 3))

# Accuracy: 0.772
# Preciison: 0.728
# Recall: 0.564
# f1: 0.632
# roc_auc: 0.842