# Extreme Gradient Boosting
##########################
import pandas as pd
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from xgboost import XGBClassifier

pd.set_option("display.max_columns", None)
warnings.simplefilter(action="ignore", category=Warning)

df = pd.read_csv("ML Datasets/diabetes.csv")

X = df.drop(["Outcome"], axis=1)  # Independent Variables
y = df["Outcome"]                 # Dependent Variables
df.head()

xgb_model = XGBClassifier(random_state=17).fit(X, y)
# random_state is not permanent (shows only the 1 sample for rs 17)

cv_results = cross_validate(xgb_model,  # XGB Model
                            X, y,       # Independent and Dependent variable
                            cv=5,       # Cross Validate
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

print("Accuracy:", round(cv_results["test_accuracy"].mean(), 3))
print("Precision:", round(cv_results["test_precision"].mean(), 3))
print("Recall:", round(cv_results["test_recall"].mean(), 3))
print("f1:", round(cv_results["test_f1"].mean(), 3))
print("roc_auc:", round(cv_results["test_roc_auc"].mean(), 3))
# Accuracy: 0.741
# Precision: 0.636
# Recall: 0.605
# f1: 0.618
# roc_auc: 0.793

print("Parameters:\n", xgb_model.get_params())
# Parameters by Default,
# learning_rate: None
# max_depth: None
# n_estimators: None
# colsample_bytree: None

parameters = {"learning_rate": [0.1, 0.01],
              "max_depth": [5, 8, None],
              "n_estimators": [100, 500, 1000],
              "colsample_bytree": [None, 0.7, 1]}

best_grid_model = GridSearchCV(xgb_model,
                               parameters,
                               cv=5,
                               n_jobs=-1,
                               verbose=1).fit(X, y)

print("Best Parameters:\n", best_grid_model.best_params_)
print("Best Accuracy Score-->", best_grid_model.best_score_)
# Parameters After GridSearchCV,
# learning_rate: 0.1
# max_depth: 5
# n_estimators: 100
# colsample_bytree: 0.7
# Best Accuracy Score: 0.76

xgb_final_model = xgb_model.set_params(**best_grid_model.best_params_).fit(X, y)

cv_results = cross_validate(xgb_final_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

print("Accuracy:", round(cv_results["test_accuracy"].mean(), 3))
print("Precision:", round(cv_results["test_precision"].mean(), 3))
print("Recall:", round(cv_results["test_recall"].mean(), 3))
print("f1:", round(cv_results["test_f1"].mean(), 3))
print("roc_auc:", round(cv_results["test_roc_auc"].mean(), 3))
# Accuracy: 0.76
# Precision: 0.687
# Recall: 0.594
# f1: 0.633
# roc_auc: 0.817

cv_results = cross_validate(xgb_final_model,
                            X, y,
                            cv=10,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

print("Accuracy:", round(cv_results["test_accuracy"].mean(), 3))
print("Precision:", round(cv_results["test_precision"].mean(), 3))
print("Recall:", round(cv_results["test_recall"].mean(), 3))
print("f1:", round(cv_results["test_f1"].mean(), 3))
print("roc_auc:", round(cv_results["test_roc_auc"].mean(), 3))
# Accuracy: 0.745
# Precision: 0.66
# Recall: 0.574
# f1: 0.606
# roc_auc: 0.82

print("Feature Importances-->", xgb_final_model.feature_importances_)
print("Feature Names-->", xgb_final_model.feature_names_in_)

def feat_imp_graph(model, num=len(X.columns), save=False):
    feat_imp = pd.DataFrame({"Value": model.feature_importances_,
                             "Name": X.columns})
    sns.barplot(data=feat_imp.sort_values("Value", ascending=False)[0: num],
                x="Value", y="Name")
    plt.grid()
    plt.title("Feature Importances")
    plt.yticks(rotation=45)
    plt.show()
    if save:
        plt.savefig("feat_imp_graph.png")

feat_imp_graph(xgb_final_model, 3)

# Analyzing model complexity with Learning Curves
def val_curve_creater(model, X, y, param_name, param_range, cv=5, scoring="roc_auc"):
    train_score, test_score = validation_curve(model, X, y,
                                               param_name=param_name,
                                               param_range=param_range,
                                               cv=cv,
                                               scoring=scoring)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color="green")
    plt.plot(param_range, mean_test_score,
             label="Test / Validation Score", color="red")

    plt.title("Validation Curve for " + type(model).__name__)
    plt.xlabel(" Number of " + param_name)
    plt.ylabel(scoring.upper())
    plt.tight_layout()
    plt.legend(loc="best")
    plt.grid()
    plt.show(block=True)

xgb_parameters = [["learning_rate", [0.01, 0.05, 0.1]],
                  ["max_depth", [3, 5, 8, None]],
                  ["n_estimators", [100, 500, 750, 1000]],
                  ["colsample_bytree", [None, 0.7, 1.0]]]

for i in range(len(xgb_parameters)):
    val_curve_creater(xgb_model, X, y,
                      xgb_parameters[i][0],
                      xgb_parameters[i][1])

