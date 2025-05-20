import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os
from automating_donwlonad_and_unzip import fetch_housing_data
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def load_housing_data(housing_path="datasets/housing"):
    fetch_housing_data()
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def prepare_data(housing):
    # Drop rows with missing values
    housing_clean = housing.dropna(axis=0)
    return housing_clean


def batchGradientDescent(X, y, learning_rate=0.01, n_iterations=1000):
    m, n = X.shape  # m = number of samples, n = number of features
    X_b = np.c_[np.ones((m, 1)), X]  # Add bias term → shape: (m, n+1)
    theta = np.random.randn(
        n + 1, 1
    )  # Random initialization of theta → shape: (n+1, 1)

    # Store the cost history for plotting
    cost_history = []

    for iteration in range(n_iterations):
        predictions = X_b.dot(theta)
        errors = predictions - y
        gradients = (2 / m) * X_b.T.dot(errors)
        theta -= learning_rate * gradients

        # Calculate cost and store it
        cost = (1 / (2 * m)) * np.sum(np.square(errors))
        cost_history.append(cost)

        # Print cost every 100 iterations
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Cost = {cost}")

    return theta, cost_history


def predict(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
    return X_b.dot(theta)


def plot_learning_curve(cost_history):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Learning Curve")
    plt.grid(True)
    plt.show()


def plot_prediction_vs_actual(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Prediction vs Actual")
    plt.grid(True)
    plt.show()


def plot_feature_importance(feature_names, theta):
    # Skip theta[0] which is the intercept
    coefficients = theta[1:].flatten()

    plt.figure(figsize=(12, 8))
    plt.barh(feature_names, np.abs(coefficients))
    plt.xlabel("Absolute Coefficient Value")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.grid(True, axis="x")
    plt.tight_layout()
    plt.show()


def main():
    # Load and prepare data
    housing = load_housing_data()
    housing_clean = prepare_data(housing)

    # Convert categorical variable to numeric using one-hot encoding
    housing_clean = pd.get_dummies(
        housing_clean, columns=["ocean_proximity"], drop_first=True
    )

    # Select all features and target variable
    features = [col for col in housing_clean.columns if col != "median_house_value"]
    X = housing_clean[features].values
    y = housing_clean["median_house_value"].values.reshape(-1, 1)

    # Feature scaling is important for gradient descent with multiple features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train the model using batch gradient descent (with more iterations for convergence)
    theta, cost_history = batchGradientDescent(
        X_train, y_train, learning_rate=0.01, n_iterations=1000
    )

    # Make predictions
    y_pred = predict(X_test, theta)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")

    # Plot results
    plot_learning_curve(cost_history)
    plot_prediction_vs_actual(y_test, y_pred)
    plot_feature_importance(features, theta)
    # Print theta
    print("\nModel Parameters (theta):")
    print(f"Intercept: {theta[0][0]}")
    for i, feature in enumerate(features):
        print(f"{feature}: {theta[i+1][0]}")


if __name__ == "__main__":
    main()
