import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Loading dataset
data = pd.read_csv('data\Training_Datasets\\notchFiltered_Training_Dataset.csv')


# Creating dataframe X and y
X = data.drop('Anomaly', axis=1)  # Features
y = data['Anomaly']  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features (scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an SVM classifier 
clf = svm.SVC(kernel='linear', gamma='scale', C=10, decision_function_shape='ovr')

# Training the SVM model
clf.fit(X_train, y_train)

# Predicting on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#  fine-tune hyperparameters using GridSearchCV
model=SVC()
param_grid = {
    'C': [0.1, 1, 10,50,100,200],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 1]
}
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Finding best parameters
best_parameters=grid_search.best_params_
highest_accuracy=grid_search.best_score_
print(highest_accuracy)
print(best_parameters)
