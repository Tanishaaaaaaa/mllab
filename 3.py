import pandas as pd

# Load the dataset
df = pd.read_csv('/content/diabetes.csv')
df

input = df.drop(['Outcome'], axis=1)
target = df['Outcome']

from sklearn.model_selection import train_test_split

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(input, target, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier

# Creating a Random Forest classifier model
model = RandomForestClassifier(n_estimators=10)
model.fit(x_train, y_train)

# Evaluating the model on the test set
model.score(x_test, y_test)

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Plotting one of the trees from the Random Forest
tree = model.estimators_[0]  # You can change the index to view other trees

plt.figure(figsize=(20, 10))
plot_tree(tree, filled=True, feature_names=input.columns, class_names=['yes', 'no'], precision=2, max_depth=2)
plt.show()
