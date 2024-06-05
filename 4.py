import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import math

df = pd.read_csv('/content/homeprices.csv')

median_bedroom = math.floor(df.bedrooms.median())
df.bedrooms = df.bedrooms.fillna(median_bedroom)

X = df[['area', 'bedrooms', 'age']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the prices on the test data
y_pred = model.predict(X_test)

# Calculate RMSE and R-squared score
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Output RMSE and R2 score
print(f"RMSE: {rmse}")
print(f"R-squared Score: {r2}")

# Visualize the results
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Prices')
plt.show()
