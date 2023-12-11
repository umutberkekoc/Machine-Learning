# Light Gradient Boosting
##########################
import pandas as pd
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from lightgbm import LGBMClassifier

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 700)
pd.set_option("display.max_rows", None)
warnings.simplefilter(action="ignore", category=Warning)

df = pd.read_csv("ML Datasets/diabetes.csv")

X = df.drop(["Outcome"], axis=1)  # Independent Variables
y = df["Outcome"]                 # Dependent Variables
df.head()

lgbm_model = LGBMClassifier(random_state=17, verbose=-1).fit(X, y)

cv_results = cross_validate(lgbm_model,     # LGBM Model
                            X, y,           # Independent and Dependent Variables
                            cv=5,           # Cross Validate
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

print("Accuracy:", round(cv_results["test_accuracy"].mean(), 3))
print("Precision:", round(cv_results["test_precision"].mean(), 3))
print("Recall:", round(cv_results["test_recall"].mean(), 3))
print("f1:", round(cv_results["test_f1"].mean(), 3))
print("roc_auc:", round(cv_results["test_roc_auc"].mean(), 3))
# Accuracy: 0.747
# Precision: 0.648
# Recall: 0.604
# f1: 0.624
# roc_auc: 0.799

# Hyperparameter Optimization
print("Parameters:", lgbm_model.get_params())
# Parameters by Default,
# learning_rate: 0.1
# n_estimators: 100
# colsample_bytree: 1

parameters = {"learning_rate": [0.01, 0.1],
              "n_estimators": [100, 300, 500, 1000],
              "colsample_bytree": [0.5, 0.7, 1]}

best_grid_model = GridSearchCV(lgbm_model,          # LGBM Model
                               parameters,          # Parameters
                               cv=5,                # Cross Validate
                               n_jobs=-1,           # Full Performance of Processor
                               verbose=1).fit(X, y)

print("Best Parameters:\n", best_grid_model.best_params_)
print("Best Accuracy Score-->", best_grid_model.best_score_)
# Parameters after GridSearch
# learning_rate: 0.01
# n_estimators: 300
# colsample_bytree: 1
# Acuracy Score: 0.764

# # Hyperparameter Optimization with new parameters again
parameters = {"learning_rate": [0.01, 0.02, 0.04],
              "n_estimators": [250, 300, 350, 400],
              "colsample_bytree": [0.8, 0.9, 1]}

best_grid_model = GridSearchCV(lgbm_model,
                               parameters,
                               cv=5,
                               n_jobs=-1,
                               verbose=True).fit(X, y)

print("Best Parameters:\n", best_grid_model.best_params_)
print("Best Accuracy Score-->", best_grid_model.best_score_)
# Parameters after GridSearch
# learning_rate: 0.01
# n_estimators: 300
# colsample_bytree: 1
# Acuracy Score: 0.764


lgbm_final_model = lgbm_model.set_params(**best_grid_model.best_params_).fit(X, y)

cv_results = cross_validate(lgbm_final_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

print("Accuracy:", round(cv_results["test_accuracy"].mean(), 3))
print("Precision:", round(cv_results["test_precision"].mean(), 3))
print("Recall:", round(cv_results["test_recall"].mean(), 3))
print("f1:", round(cv_results["test_f1"].mean(), 3))
print("roc_auc:", round(cv_results["test_roc_auc"].mean(), 3))
# Accuracy: 0.764
# Precision: 0.691
# Recall: 0.597
# f1: 0.637
# roc_auc: 0.815

# # Hyperparameter Optimization for n_estimators (most improtant parameter)
lgbm_model = LGBMClassifier(random_state=17, learning_rate=0.01, colsample_bytree=1).fit(X, y)

parameters = {"n_estimators": [200, 300, 350, 400, 450, 500, 550, 600]}

best_grid_model = GridSearchCV(lgbm_model,
                               parameters,
                               cv=5,
                               n_jobs=-1,
                               verbose=True).fit(X, y)

print("Best Parameters-->", best_grid_model.best_params_)
# n_estimators: 300

def feature_importances_graph_creater(model, num=len(X), save=False):
    feat_imp = pd.DataFrame({"Value": model.feature_importances_,
                             "Feature": X.columns})
    plt.figure(figsize=(12, 10))
    sns.barplot(x="Value", y="Feature", data=feat_imp.sort_values("Value", ascending=False)[0: num])
    plt.title(" Feature Importances ")
    plt.yticks(rotation=45)
    plt.grid()
    plt.show()
    if save:
        plt.savefig("features.png")

feature_importances_graph_creater(lgbm_final_model, num=5)


def val_curve_creater(model, X, y, par_name, par_range, cv=5, scoring="roc_auc"):
    train_score, test_score = validation_curve(model, X, y,
                                               cv=cv,
                                               param_name=par_name,
                                               param_range=par_range,
                                               scoring=scoring)

    average_train_score = np.mean(train_score, axis=1)
    average_test_score = np.mean(test_score, axis=1)

    plt.plot(par_range, average_train_score,
             label="Train Score", color="blue")
    plt.plot(par_range, average_test_score,
             label="Test / Validation Score", color="red")

    plt.grid()
    plt.title("Validation Curve for " + type(model).__name__)
    plt.xlabel(" Number of " + par_name)
    plt.ylabel(scoring.upper())
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show(block=True)

lgbm_parameters = [["learning_rate", [0.01, 0.05, 0.1, 0.2]],
                   ["n_estimators", [100, 300, 500, 800, 1000]],
                   ["colsample_bytree", [0.5, 0.7, 1.0]]]

for i in range(len(lgbm_parameters)):
    val_curve_creater(lgbm_model, X, y,
                      lgbm_parameters[i][0],
                      lgbm_parameters[i][1])
