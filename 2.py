import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the dataset
df = pd.read_csv('/content/salaries.csv')

# Separate input features and target variable
input = df.drop('salary_more_then_100k', axis='columns')
target = df['salary_more_then_100k']

# Label encode categorical variables
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

input['companyEnc'] = le_company.fit_transform(input['company'])
input['jobEnc'] = le_job.fit_transform(input['job'])
input['degreeEnc'] = le_degree.fit_transform(input['degree'])

# Drop original categorical columns
inputs = input.drop(['company', 'job', 'degree'], axis='columns')

# Initialize and train the decision tree model
model = tree.DecisionTreeClassifier(criterion='gini')
model.fit(inputs, target)

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=input.columns, class_names=['No', 'Yes'], filled=True)
plt.title('Decision Tree')
plt.show()
