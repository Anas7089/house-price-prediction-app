# train_model.py

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

print("Training model...")

# 1. Load the dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)

# For simplicity, we'll use only a few features
features_to_use = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population']
X = X[features_to_use]

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Choose and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluate the model (optional, but good practice)
score = model.score(X_test, y_test)
print(f"Model R^2 score: {score:.4f}")

# 5. Save the trained model to a file
joblib.dump(model, 'model.joblib')

print("Model trained and saved as model.joblib")