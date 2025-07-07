# ==============================================================================
# Logistic Regression (Multi-class Classification)
# Dataset: Iris Flower Dataset
# Objective: Classify Iris flower species based on sepal and petal measurements.
# Species: 0: setosa, 1: versicolor, 2: virginica
# ==============================================================================

# Part 0: Initial Setup and Data Loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris 

# Load the Iris dataset
iris = load_iris(as_frame=True)
df_iris = iris.frame            # Get the DataFrame of features
df_iris['target'] = iris.target # Add the target variable (0: setosa, 1: versicolor, 2: virginica)

print("--- Iris Dataset Loaded ---")
print(df_iris.head())

# --- Iris Dataset Loaded ---
#    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
# 0                5.1               3.5                1.4               0.2       0
# 1                4.9               3.0                1.4               0.2       0
# 2                4.7               3.2                1.3               0.2       0
# 3                4.6               3.1                1.5               0.2       0
# 4                5.0               3.6                1.4               0.2       0

print("\n--- Data Info ---")
df_iris.info()

# --- Data Info ---
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 150 entries, 0 to 149
# Data columns (total 5 columns):
#  #   Column             Non-Null Count  Dtype
# ---  ------             --------------  -----
#  0   sepal length (cm)  150 non-null    float64
#  1   sepal width (cm)   150 non-null    float64
#  2   petal length (cm)  150 non-null    float64
#  3   petal width (cm)   150 non-null    float64
#  4   target             150 non-null    int64
# dtypes: float64(4), int64(1)
# memory usage: 6.0 KB

print("\n--- Data Description ---")
print(df_iris.describe())

# --- Data Description ---
#        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)      target
# count         150.000000        150.000000         150.000000        150.000000  150.000000
# mean            5.843333          3.057333           3.758000          1.199333    1.000000
# std             0.828066          0.435866           1.765298          0.762238    0.819232
# min             4.300000          2.000000           1.000000          0.100000    0.000000
# 25%             5.100000          2.800000           1.600000          0.300000    0.000000
# 50%             5.800000          3.000000           4.350000          1.300000    1.000000
# 75%             6.400000          3.300000           5.100000          1.800000    2.000000
# max             7.900000          4.400000           6.900000          2.500000    2.000000

print("\n--- Target Variable Distribution ---")
print(df_iris['target'].value_counts()) # Check class distribution

# --- Target Variable Distribution ---
# target
# 0    50
# 1    50
# 2    50
# Name: count, dtype: int64

# Description of the features and target names:
# print("\n--- Dataset Features and Target Names Description ---")
print(iris.DESCR)           # Description of the dataset   
print(iris.target_names)    # Returns ['setosa', 'versicolor', 'virginica']

print("\n" + "="*80 + "\n") # Separator

# ==============================================================================
# Exploratory Data Analysis (EDA)
# ==============================================================================

print("--- Exploratory Data Analysis (EDA) ---")

# Check the value_counts() for the target column.
print("\nTarget Variable Distribution (again for Task 1 check):")
print(df_iris['target'].value_counts())
# The dataset is perfectly balanced (50 for each category)

# Generate a seaborn.pairplot for all features (sepal length (cm), sepal width (cm),
# petal length (cm), petal width (cm)) colored by the target variable.
sns.pairplot(df_iris, hue='target')
plt.suptitle("Pairplot of Iris Features by Species", y=1.02)
plt.show()

# Analysis:
# The features that separate the species best are Petal Length vs. Petal Width.
# setosa (class 0) is clearly separated from the other two species for all metrics and is,
# therefore, easiest to separate. Versicolor (1) and Virginica (2) show some overlap,
# particularly in sepal measurements, but are more distinct in petal measurements.

print("\n" + "="*80 + "\n") # Separator

# ==============================================================================
# Data Preparation
# ==============================================================================

print("--- Data Preparation ---")

# Define X (features) and y (target) variables from df_iris.
X = df_iris.drop('target', axis=1)
y = df_iris['target']

# Split the data into training and testing sets using train_test_split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nShape of X_train: {X_train.shape}")
print(f"Shape of X-test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# Shape of X_train: (120, 4)
# Shape of X-test: (30, 4)
# Shape of y_train: (120,)
# Shape of y_test: (30,)

# Apply StandardScaler to the features.
# Instantiate the StandardScaler.
scaler = StandardScaler()
# Fit the scaler only on X_train.
scaler.fit(X_train)
# Transform both X_train and X_test using the fitted scaler, creating X_train_scaled and X_test_scaled.
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print the means and standard deviations of X_train_scaled to confirm they are close to 0 and 1, respectively.
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

print("\n--- Scaled X_train (1st 5 rows) ---")
print(X_train_scaled_df.head())

# --- Scaled X_train (1st 5 rows) ---
#      sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# 8            -1.721568         -0.332101          -1.345722         -1.323276
# 106          -1.124492         -1.227655           0.414505          0.651763
# 76            1.144395         -0.555990           0.584850          0.256755
# 9            -1.124492          0.115676          -1.288941         -1.454945
# 89           -0.408002         -1.227655           0.130598          0.125086

print("\n--- Mean of Scaled X_train (should be close to 0) ---")
print(X_train_scaled_df.mean().head())

# ---Mean of Scaled X_train (should be close to 0) ---
# sepal length (cm)   -1.369275e-16
# sepal width (cm)     9.992007e-16
# petal length (cm)    1.665335e-17
# petal width (cm)     1.702342e-16
# dtype: float64

print("\n--- Standard Deviation of Scaled X_train (should be close to 1) ---")
print(X_train_scaled_df.std().head())

# --- Standard Deviation of Scaled X_train (should be close to 1) ---
# sepal length (cm)    1.004193
# sepal width (cm)     1.004193
# petal length (cm)    1.004193
# petal width (cm)     1.004193
# dtype: float64

print("\n" + "="*80 + "\n") # Separator

# ==============================================================================
# Model Building and Training
# ==============================================================================

print("--- Task 3: Model Building and Training ---")

# Create an instance of LogisticRegression.
# For multi-class problems, sklearn's LogisticRegression handles it automatically.
# It typically uses a "One-vs-Rest" (OvR) strategy by default unless
# 'multi_class' is specified for a multinomial solver.
model_lr = LogisticRegression(random_state=42)  # Not specifying solver or multi_class as sklearn will handle this automatically

# Fit the model using your X_train_scaled and y_train.
model_lr.fit(X_train_scaled, y_train)

print("\n--- Logistic Regression Model Trained ---")
print(f"Number of features: {model_lr.n_features_in_}")
print(f"Model coefficients (for the first class, 1st 4 features): {model_lr.coef_[0, :4]}")
# Number of features: 4
# Model coefficients (for the first class, 1st 4 features): [-1.08894494  1.02420763 -1.79905609 -1.68622819]
print(f"Model intercept (for the first class): {model_lr.intercept_[0]:.4f}")
# Model intercept (for the first class): -0.3056
print(f"Model intercept (for the second class): {model_lr.intercept_[1]:.4f}")
# Model intercept (for the second class): 1.9086
print(f"Model intercept (for the third class): {model_lr.intercept_[2]:.4f}")         
# Model intercept (for the third class): -1.6030

print("\n" + "="*80 + "\n") # Separator

# ==============================================================================
# Prediction
# ==============================================================================

print("--- Prediction ---")

# Use the trained model to make predictions on X_test_scaled.
# Get both the predicted class labels (y_pred) using model.predict().
y_pred = model_lr.predict(X_test_scaled)
print("\n--- Predicted Class Labels (1st 5) ---")
print(y_pred[:5])
# Predicted Class Labels (1st 5) ---
# [0 2 1 1 0]

# And the predicted probabilities (y_prob) using model.predict_proba().
y_prob = model_lr.predict_proba(X_test_scaled)
print("\n--- Predicted Probabilities (1st 5 for all classes) ---")
print(y_prob[:5])

# Predicted Probabilities (1st 5 for all classes) ---
# [[9.78818005e-01 2.11816311e-02 3.63821812e-07]
#  [3.79836951e-03 3.69220168e-01 6.26981463e-01]
#  [1.48799040e-01 8.42474895e-01 8.72606441e-03]
#  [9.54449962e-02 8.94618634e-01 9.93636953e-03]
#  [9.88493051e-01 1.15067767e-02 1.72549650e-07]]

# The columns in y_prob represent the probability of each instance belonging to
# each of the three classes (setosa, versicolor, virginica), in the order of their class labels (0, 1, 2).
# For example, y_prob[0, 0] is the probability of the first sample being class 0 (setosa).

print("\n--- Predicted Probabilities (1st 5 for class 0)")
print(y_prob[:5, 0])    # [0.97881801 0.00379837 0.14879904 0.095445   0.98849305]
print("\n--- Predicted Probabilities (1st 5 for class 1)")
print(y_prob[:5, 1])    # [0.02118163 0.36922017 0.8424749  0.89461863 0.01150678]
print("\n--- Predicted Probabilities (1st 5 for class 2)")
print(y_prob[:5, 2])    # [3.63821812e-07 6.26981463e-01 8.72606441e-03 9.93636953e-03 1.72549650e-07]

y_pred = model_lr.predict(X_test_scaled)
print("\n--- Predicted Class Labels (1st 5) ---")
print(y_pred[:5])           
# Predicted Class Labels (1st 5) ---
# [0 2 1 1 0]

print("\n--- Actual Class Labels (1st 5) ---")
print(y_test.values[:5])
# Actual Class Labels (1st 5) ---
# [0 2 1 1 0]

print("\n" + "="*80 + "\n") # Separator

# ==============================================================================
# Model Evaluation
# ==============================================================================

print("--- Model Evaluation ---")

# Calculate and print the Accuracy Score for your model on the test set.
accuracy = accuracy_score(y_test, y_pred)
print("\n--- Model Accuracy ---")
print(f"Accuracy: {accuracy:.4f}")
# Accuracy: 0.9333

# Print the Confusion Matrix.
cm = confusion_matrix(y_test, y_pred)
print("\n--- Confusion Matrix ---")
print(cm)

# Confusion Matrix ---
# [[10  0  0]
#  [ 0  9  1]
#  [ 0  1  9]]

# Interpretation:
# The confusion matrix has Actual labels on the rows and Predicted labels on the columns.
# Each cell `cm[i, j]` represents the number of instances that actually belong to class `i`
# but were predicted as class `j`.
# For the Iris dataset (0: setosa, 1: versicolor, 2: virginica):
# - `cm[0,0]` = 10: 10 cases were actually setosa and correctly predicted as setosa (True Positive for setosa).
# - `cm[0,1]` = 0: 0 cases were actually setosa but predicted as versicolor (False Positive for versicolor, False Negative for setosa).
# - `cm[0,2]` = 0: 0 cases were actually setosa but predicted as virginica (False Positive for virginica, False Negative for setosa).
# - `cm[1,0]` = 0: 0 cases were actually versicolor but predicted as setosa.
# - `cm[1,1]` = 9: 9 cases were actually versicolor and correctly predicted as versicolor.
# - `cm[1,2]` = 1: 1 case was actually versicolor but predicted as virginica (False Positive for virginica, False Negative for versicolor).
# - `cm[2,0]` = 0: 0 cases were actually virginica but predicted as setosa.
# - `cm[2,1]` = 1: 1 case was actually virginica but predicted as versicolor (False Positive for versicolor, False Negative for virginica).
# - `cm[2,2]` = 9: 9 cases were actually virginica and correctly predicted as virginica.

# Generate and print the Classification Report.
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# Classification Report ---
#               precision    recall  f1-score   support
#
#            0       1.00      1.00      1.00        10
#            1       0.90      0.90      0.90        10
#            2       0.90      0.90      0.90        10
#
#     accuracy                           0.93        30
#    macro avg       0.93      0.93      0.93        30
# weighted avg       0.93      0.93      0.93        30


# Setosa (class 0) is predicted best with perfect precision, recall, and f1-scores (all 1.00).
# This is likely because, as observed in the pairplot, setosa is linearly separable from the other two species.
# The model performs equally well for Versicolor (class 1) and Virginica (class 2), both having
# precision, recall, and f1-scores of 0.90. The model made one error for each of these classes,
# misclassifying one Versicolor as Virginica and one Virginica as Versicolor.
# This aligns with the pairplot showing some overlap between Versicolor and Virginica, making them harder to distinguish perfectly.

print("\n" + "="*80 + "\n") # Separator

# ==============================================================================
# Coefficient Interpretation
# ==============================================================================

print("--- Coefficient Interpretation ---")

# Observe the shape of model.coef_.
print(f"Shape of model coefficients (model_lr.coef_): {model_lr.coef_.shape}")
# Shape of model coefficients (model_lr.coef_): (3, 4)


print(f"\nModel coefficients (model_lr.coef_): \n{model_lr.coef_}")
# Model coefficients (model_lr.coef_):
# [[-1.08894494  1.02420763 -1.79905609 -1.68622819]  # Coefficients for Class 0 (setosa)
#  [ 0.53633654 -0.36048698 -0.20407418 -0.80795703]  # Coefficients for Class 1 (versicolor)
#  [ 0.5526084  -0.66372065  2.00313027  2.49418523]] # Coefficients for Class 2 (virginica)

# Interpretation of model.coef_ shape and meaning:
# - Rows (3): Correspond to the number of classes (setosa, versicolor, virginica).
#   In a One-vs-Rest (OvR) strategy, there's one set of coefficients for each binary
#   classifier (e.g., Class 0 vs. (Class 1 & 2), Class 1 vs. (Class 0 & 2), etc.).
# - Columns (4): Correspond to the number of features in the dataset
#   (sepal length, sepal width, petal length, petal width).

print(f"\nModel intercepts (model_lr.intercept_): {model_lr.intercept_}")
# Model intercepts (model_lr.intercept_): [-0.30560136  0.36647906 -0.0608777 ]
# Interpretation of model.intercept_:
# This is an array of intercepts, one for each of the three binary classifiers trained
# by the OvR strategy.

# Note:
# In multi-class OvR Logistic Regression, interpreting individual coefficients as
# simple "odds ratios" (exp(coef)) is more complex because each set of coefficients
# (each row in `model.coef_`) pertains to a binary classifier that distinguishes
# one specific class against *all other classes combined*, not just against a single
# contrasting class. For example, the first row of coefficients explains the log-odds
# of being 'setosa' versus 'not setosa'. This is different from binary classification
# where a coefficient directly relates to the log-odds of the positive class versus
# the single negative class. Therefore, a direct `exp(coef)` tells you the odds
# of *that specific class* increasing relative to *any other class*, which is a
# less straightforward interpretation than in a simple binary (class A vs. class B) context.

print("\n" + "="*80 + "\n") # Separator
