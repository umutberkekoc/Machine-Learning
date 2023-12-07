# Gradient Boosting Machines
############################
import warnings
import pandas as pd
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.ensemble import GradientBoostingClassifier

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 700)
warnings.simplefilter(action="ignore", category=Warning)

df = pd.read_csv("ML Datasets/diabetes.csv")

X = df.drop(["Outcome"], axis=1)  # Independent Variables
y = df["Outcome"]                 # Dependent Variables
df.head()

gbm_model = GradientBoostingClassifier(random_state=17).fit(X, y)
# random_state is not permanent (shows only the 1 sample for rs 17)

cv_results = cross_validate(gbm_model,   # Gradient Boosting Model
                            X, y,        # Independent and Dependent Variables
                            cv=5,        # Cross Validate Value
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

print("Accuracy:", round(cv_results["test_accuracy"].mean(), 3))
print("Precision:", round(cv_results["test_precision"].mean(), 3))
print("Recall:", round(cv_results["test_recall"].mean(), 3))
print("f1:", round(cv_results["test_f1"].mean(), 3))
print("roc_auc:", round(cv_results["test_roc_auc"].mean(), 3))
# Accuracy: 0.759
# Precision: 0.675
# Recall: 0.601
# f1: 0.634
# roc_auc: 0.826

print("Parameters:", gbm_model.get_params())
# Parameters by Default,
# learning_rate: 0.1
# max_depth: 3
# n_estimators: 100
# subsample: 1.0

parameters = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 5, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [0.5, 0.7, 1]}
# Default values of parameters should being exists in our parameters that we will check for the optimal

best_grid_model = GridSearchCV(gbm_model,   # GBM Model
                               parameters,  # Parameters
                               cv=5,        # Cross Validate Value
                               n_jobs=-1,   # Full Performance of Processor
                               verbose=True).fit(X, y)  # Report

print("Best Parameters:", best_grid_model.best_params_)
print("Best Accuracy Score:", best_grid_model.best_score_)
# Parameters After GridSearchCV,
# learning_rate: 0.01
# max_depth: 3
# n_estimators: 1000
# subsample: 0.7
# Best Accuracy Score: 0.775

gbm_final_model = gbm_model.set_params(**best_grid_model.best_params_,
                                       random_state=17).fit(X, y)

cv_results = cross_validate(gbm_final_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

print("Accuracy:", round(cv_results["test_accuracy"].mean(), 3))
print("Precision:", round(cv_results["test_precision"].mean(), 3))
print("Recall:", round(cv_results["test_recall"].mean(), 3))
print("f1:", round(cv_results["test_f1"].mean(), 3))
print("roc_auc:", round(cv_results["test_roc_auc"].mean(), 3))
# Accuracy: 0.775
# Precision: 0.695
# Recall: 0.638
# f1: 0.663
# roc_auc: 0.835

cv_results = cross_validate(gbm_final_model,
                            X, y,
                            cv=10,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
print("Accuracy:", round(cv_results["test_accuracy"].mean(), 3))
print("Precision:", round(cv_results["test_precision"].mean(), 3))
print("Recall:", round(cv_results["test_recall"].mean(), 3))
print("f1:", round(cv_results["test_f1"].mean(), 3))
print("roc_auc:", round(cv_results["test_roc_auc"].mean(), 3))
# Accuracy: 0.771
# Precision: 0.703
# Recall: 0.619
# f1: 0.65
# roc_auc: 0.83

print("Feature Importances-->", gbm_final_model.feature_importances_)
def feat_imp_graph(model, num=len(X), save=False):
    feat_imp = pd.DataFrame({"Value": model.feature_importances_,
                             "Feature": model.feature_names_in_})
    sns.barplot(data=feat_imp.sort_values("Value", ascending=False)[0: num],
                x="Value", y="Feature")
    plt.title("Feature Importances")
    plt.xlabel("Value")
    plt.ylabel("Name")
    plt.yticks(rotation=45)
    plt.grid()
    plt.show()
    if save:
        plt.savefig("feat_imp_graph.png")

feat_imp_graph(gbm_final_model)


def validation_curve_creater(model, X, y, name, range, cv=5, scoring="roc_auc"):
    train_score, test_score = validation_curve(model,
                                               X, y,
                                               cv=cv,
                                               param_name=name,
                                               param_range=range,
                                               scoring=scoring)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)
    print("Train Score Mean-->", mean_train_score)
    print("Test Score Mean-->", mean_test_score)

    plt.plot(mean_train_score, label="Training Score", color="green")
    plt.plot(mean_test_score, label="Test / Validation Score", color="red")

    plt.title(" Validation Curve for " + type(model).__name__)
    plt.xlabel("Number of " + name)
    plt.ylabel(scoring.upper())
    plt.tight_layout()
    plt.grid()
    plt.legend(loc="best")
    plt.show(block=True)

gbm_parameters = [["learning_rate", [0.01, 0.05, 0.1]],
                  ["max_depth", [2, 3, 5, 8]],
                  ["n_estimators", [500, 700, 1000]],
                  ["subsample", [0.5, 0.7, 0.9, 1.0]]]

for i in range(len(gbm_parameters)):
    validation_curve_creater(gbm_model, X, y, gbm_parameters[i][0],
                     gbm_parameters[i][1])
