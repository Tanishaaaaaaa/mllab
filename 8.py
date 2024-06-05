import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore', category=DataConversionWarning)

# Load dataset from local CSV file
path_to_csv = 'diabetes.csv'  # Replace with your local file path

# Read the dataset
data = pd.read_csv(path_to_csv)

# Verify the first few rows of the dataset to understand its structure
print(data.head())

# Verify the columns
print(data.columns)

# Handle missing or malformed data
for col in data.columns:
    if data[col].dtype == 'object':
        # Try converting to numeric, if fails fill with NaN
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Print the data types to verify the conversion
print(data.dtypes)

# Define your columns if necessary
# If the columns are already correct in the CSV file, skip this part
# columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
# data.columns = columns

# Handling missing values (replace 0 with NaN and fill with mean for specific columns)
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    data[col].replace(0, np.nan, inplace=True)
    data[col].fillna(data[col].mean(), inplace=True)

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('Outcome', axis=1), data['Outcome'], test_size=0.2, random_state=42)

# Normalizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class KNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        if self.distance_metric == 'euclidean':
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            
        else:  # manhattan distance
            distances = np.sum(np.abs(self.X_train - x), axis=1)
          
        
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train.iloc[k_indices]
        return Counter(k_nearest_labels).most_common(1)[0][0]

def evaluate_model(k, metric='euclidean'):
    knn = KNNClassifier(k=k, distance_metric=metric)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    return accuracy_score(y_test, y_pred)

# Evaluate KNN with varying k and distance metrics
k_values = range(1, 21)
metrics = ['euclidean', 'manhattan']
results = {metric: [evaluate_model(k, metric) for k in k_values] for metric in metrics}

# Plotting the results

plt.figure(figsize=(12, 6))
for metric, accuracies in results.items():
    plt.plot(k_values, accuracies, label=metric)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('KNN varying k and distance metrics')
plt.legend()
plt.show()
