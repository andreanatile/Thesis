# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler



# Load your dataset (replace 'data.csv' with your dataset's path)
data = pd.read_csv('data\Training_Datasets\\notchFiltered_Training_Dataset.csv')


# Assuming your dataset has a 'label' column with multi-class labels (ok, span, mild, severe)
X = data.drop('Anomaly', axis=1)  # Features
y = data['Anomaly']  # Labels

# Create an instance of RandomUnderSampler for undersampling "ok"
undersampler = RandomUnderSampler(sampling_strategy={'ok':300,'Anomaly':99}, random_state=None)
X_resampled, y_resampled = undersampler.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize the features (scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an SVM classifier with an RBF kernel for multi-class classification
clf = svm.SVC(kernel='linear', gamma='scale', C=1, decision_function_shape='ovr')  # 'ovr' for one-vs-rest

# Train the SVM model
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#  fine-tune hyperparameters using technique GridSearchCV
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
 