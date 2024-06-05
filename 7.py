import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
url = 'diabetes.csv'
data = pd.read_csv(url)

# Check for missing values
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Split the dataset into features and target variable
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(X_train, y_train, X_test, k=3):
    y_pred = []
    for test_point in X_test:
        distances = []
        for i in range(len(X_train)):
            distance = euclidean_distance(test_point, X_train[i])
            distances.append((distance, y_train.iloc[i]))
        distances.sort(key=lambda x: x[0])
        k_nearest_neighbors = distances[:k]
        k_nearest_labels = [label for (_, label) in k_nearest_neighbors]
        most_common_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
        y_pred.append(most_common_label)
    return np.array(y_pred)

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

def knn_predict_manhattan(X_train, y_train, X_test, k=3):
    y_pred = []
    for test_point in X_test:
        distances = []
        for i in range(len(X_train)):
            distance = manhattan_distance(test_point, X_train[i])
            distances.append((distance, y_train.iloc[i]))
        distances.sort(key=lambda x: x[0])
        k_nearest_neighbors = distances[:k]
        k_nearest_labels = [label for (_, label) in k_nearest_neighbors]
        most_common_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
        y_pred.append(most_common_label)
    return np.array(y_pred)

# Make predictions on the test set with Euclidean distance and k=3
y_pred = knn_predict(X_train, y_train, X_test, k=3)

# Calculate accuracy with Euclidean distance and k=3
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with Euclidean distance and k=3:", accuracy)

# Try different values of K for Euclidean distance
k_values = range(1, 21)
accuracies = []
for k in k_values:
    y_pred = knn_predict(X_train, y_train, X_test, k)
    accuracies.append(accuracy_score(y_test, y_pred))

# Plot the results for Euclidean distance
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.title('K vs. Accuracy (Euclidean Distance)')
plt.xlabel('Number of Neighbors K')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# Make predictions on the test set with Manhattan distance and k=3
y_pred_manhattan = knn_predict_manhattan(X_train, y_train, X_test, k=3)

# Calculate accuracy with Manhattan distance and k=3
accuracy_manhattan = accuracy_score(y_test, y_pred_manhattan)
print("Accuracy with Manhattan distance and k=3:", accuracy_manhattan)
