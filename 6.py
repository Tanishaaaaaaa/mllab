import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import numpy as np
import math

# Load the dataset
df = pd.read_csv('/content/titanic.csv')

# Drop irrelevant columns
df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

# Separate target variable and input features
target = df['Survived']
input = df.drop(['Survived'], axis=1)

# Encode categorical variable 'Sex'
sex_le = LabelEncoder()
input['sex_enc'] = sex_le.fit_transform(input['Sex'])
input.drop(['Sex'], axis=1, inplace=True)

# Fill missing values in 'Age' column with median
median_age = math.floor(input.Age.median())
input.Age = input.Age.fillna(median_age)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(input, target, test_size=0.2)

# Initialize and train the Gaussian Naive Bayes model
model = GaussianNB()
model.fit(x_train, y_train)

# Make predictions
predictions = model.predict(x_test)

# Calculate accuracy
accuracy = model.score(x_test, y_test)
print("Accuracy:", accuracy)
