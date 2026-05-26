import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import pickle

# Load Dataset
data = pd.read_csv("house_data.csv")

# Select Features
X = data[['Area', 'No. of Bedrooms', 'CarParking', 'SwimmingPool']]

# Target
y = data['Price']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = r2_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

# Save Model
pickle.dump(model, open("model.pkl", "wb"))

print("\nModel Saved Successfully!")

# Graph
plt.scatter(y_test, predictions)

plt.xlabel("Actual Prices")

plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted House Prices")

plt.show()