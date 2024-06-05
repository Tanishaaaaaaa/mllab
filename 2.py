
Open In Colab

import pandas as pd
df=pd.read_csv('/content/salaries.csv')
df

input=df.drop('salary_more_then_100k',axis='columns')
target=df['salary_more_then_100k']

     

input
     

target

from sklearn.preprocessing import LabelEncoder

     

le_company=LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()
     

input['companyEnc']=le_company.fit_transform(input['company'])
input['jobEnc']=le_company.fit_transform(input['job'])
input['degreeEnc']=le_company.fit_transform(input['degree'])

     

input

inputs=input.drop(['company','job','degree'],axis='columns')
     

from sklearn import tree
model=tree.DecisionTreeClassifier(criterion='gini')
#model=tree.DecisionTreeClassifier(criterion='entropy')

     

model.fit(inputs,target)
     


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(model, feature_names=input.columns, class_names=['No', 'Yes'], filled=True)
plt.title('Decision Tree')
plt.show()
     



     
