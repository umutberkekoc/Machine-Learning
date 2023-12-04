# Random Forests
##########################
import warnings
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_validate, validation_curve
from sklearn.ensemble import RandomForestClassifier

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 600)
warnings.simplefilter(action="ignore", category=Warning)

df = pd.read_csv("ML Datasets/diabetes.csv")

X = df.drop(["Outcome"], axis=1)  # Independent Variables
y = df["Outcome"]                 # Dependent Variable
df.head()

rf_model = RandomForestClassifier(random_state=17).fit(X, y)
# random_state is not permanent (shows only the 1 sample for rs 17)

# K-fold Cross Validation
cv_results = cross_validate(rf_model,  # Random Forests model
                            X, y,      # Independent and Dependent Variables
                            cv=5,      # Cross Validate Value
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

print("Accuracy:", round(cv_results["test_accuracy"].mean(), 3))
print("Precision:", round(cv_results["test_precision"].mean(), 3))
print("Recall:", round(cv_results["test_recall"].mean(), 3))
print("f1:", round(cv_results["test_f1"].mean(), 3))
print("roc_auc:", round(cv_results["test_roc_auc"].mean(), 3))
# Accuracy: 0.766
# Precision: 0.706
# Recall: 0.582
# f1: 0.635
# roc_auc: 0.828

print("Parameters:\n", rf_model.get_params())
# Parameters by Default,
# max_depth: None
# max_features: sqrt
# min_samples_split: 2
# n_estimators: 100

parameters = {"max_depth": [3, 5, 8, None],
              "max_features": [3, 5, 7, "sqrt"],
              "min_samples_split": [2, 5, 8, 15, 20],
              "n_estimators": [100, 200, 500]}
# Default value of parameters should being exists in our parameters that we will check for the optimal

best_model_grid = GridSearchCV(rf_model,    # Random Forests Model
                               parameters,  # Parameters
                               cv=5,        # Cross Validate Value
                               n_jobs=-1,   # Full Performance of Processor
                               verbose=True).fit(X, y)  # Report

print("Best Parameters:\n", best_model_grid.best_params_)
print("Best Accuracy Score:-->", round(best_model_grid.best_score_, 3))
# Parameters After GridSearchCV,
# max_depth: None
# max_features: 5
# min_samples_split: 8
# n_estimators: 500
# Best Accuracy Score: 0.777

rf_final_model = rf_model.set_params(**best_model_grid.best_params_,
                                     random_state=17).fit(X, y)
# do not need to write random state in real code.

cv_results = cross_validate(rf_final_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

print("Accuracy:", round(cv_results["test_accuracy"].mean(), 3))
print("Precision:", round(cv_results["test_precision"].mean(), 3))
print("Recall:", round(cv_results["test_recall"].mean(), 3))
print("f1:", round(cv_results["test_f1"].mean(), 3))
print("roc_auc:", round(cv_results["test_roc_auc"].mean(), 3))
# Accuracy: 0.777
# Precision: 0.707
# Recall: 0.635
# f1: 0.665
# roc_auc: 0.827

cv_results = cross_validate(rf_final_model,
                            X, y,
                            cv=10,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

print("Accuracy:", round(cv_results["test_accuracy"].mean(), 3))
print("Precision:", round(cv_results["test_precision"].mean(), 3))
print("Recall:", round(cv_results["test_recall"].mean(), 3))
print("f1:", round(cv_results["test_f1"].mean(), 3))
print("roc_auc:", round(cv_results["test_roc_auc"].mean(), 3))
# Accuracy: 0.767
# Precision: 0.692
# Recall: 0.615
# f1: 0.645
# roc_auc: 0.827

# Feature Importances:
print("Feature Importances:\n", rf_final_model.feature_importances_)
def feat_imp_graph(model, num=len(X), save=False):
    feat_imp = pd.DataFrame({"Value": model.feature_importances_,
                             "Feature": model.feature_names_in_})
                           # "Feature": X.columns  # 2.way
    plt.figure(figsize=(10, 10))
    sns.barplot(data=feat_imp.sort_values("Value", ascending=False)[0: num],
                x="Value", y="Feature")
    plt.grid()
    plt.xlabel("Values")
    plt.ylabel("Feature Names")
    plt.yticks(rotation=45)
    plt.show()

    if save:
        plt.savefig("feat_imp_pic.png")

feat_imp_graph(rf_final_model)

# Analyzing Model Complexity with Learning Curves
for i in best_model_grid.best_params_:
    print("Parameter ", i)
param_name = str(input("Enter the parameter name"))
scoring = str(input("Enter the score (roc_auc/f1/accuracy"))
start = int(input("Enter the start point for the range"))
finish = int(input("Enter the finish point for the range"))
step = int(input("Enter the step size for the range"))

train_score, test_score = validation_curve(rf_final_model,
                                           X, y,
                                           cv=5,
                                           param_name=param_name,
                                           param_range=range(start, finish, step),
                                           scoring=scoring)

average_train_score = np.mean(train_score, axis=1)
average_test_score = np.mean(test_score, axis=1)

plt.plot(average_train_score, label="Training Score", color="blue")
plt.plot(average_test_score, label="Test / Validate Score", color="red")
plt.grid()
plt.title("Validation Curve for max_depth")
plt.xlabel("Number of max_depth")
plt.ylabel("ROC_AUC")
plt.legend(loc="best")
plt.show()