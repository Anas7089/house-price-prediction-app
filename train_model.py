# train_model.py (UPDATED)

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import mlflow
import mlflow.sklearn

print("Starting model training with MLflow...")

# Set the experiment name
mlflow.set_experiment("California House Price Prediction")

# Start an MLflow run
with mlflow.start_run():
    # 1. Load the dataset
    housing = fetch_california_housing()
    features_to_use = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population']
    X = pd.DataFrame(housing.data, columns=housing.feature_names)[features_to_use]
    y = pd.Series(housing.target)

    # 2. Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Choose and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 4. Evaluate the model
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    print(f"Model R^2 score: {score:.4f}")

    # 5. Log everything with MLflow
    mlflow.log_param("features", features_to_use)
    mlflow.log_metric("r2_score", score)
    
    # Log the model itself
    mlflow.sklearn.log_model(model, "linear_regression_model")

    print("MLflow run completed and logged.")