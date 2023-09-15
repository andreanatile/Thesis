# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# Load your dataset (replace 'data.csv' with your dataset's path)
data = pd.read_csv('data\Training_Datasets\\raw_Training_Dataset.csv')


# Assuming your dataset has a 'label' column with multi-class labels (ok, span, mild, severe)
X = data.drop('Anomaly', axis=1)  # Features
y = data['Anomaly']  # Labels

# Split the data into training and testing sets (e.g., 80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an SVM classifier with an RBF kernel for multi-class classification
clf = svm.SVC(kernel='poly', gamma='scale', C=0.1, decision_function_shape='ovr')  # 'ovr' for one-vs-rest

# Train the SVM model
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# You can also fine-tune hyperparameters using techniques like GridSearchCV
model=SVC()
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 1]
}
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Find the best parameters
best_parameters=grid_search.best_params_
highest_accuracy=grid_search.best_score_
print(highest_accuracy)
print(best_parameters)