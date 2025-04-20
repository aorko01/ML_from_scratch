import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import kagglehub
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import StandardScaler


def load_iris_data(iris_path="datasets/iris"):
    fetch_iris_data()
    csv_path = os.path.join(iris_path, "iris.data")
    return pd.read_csv(csv_path, header=None)


def load_users_vs_bots_data():
    """Load the users vs bots classification dataset using kagglehub"""
    print("Downloading users vs bots dataset from Kaggle...")
    path = kagglehub.dataset_download("juice0lover/users-vs-bots-classification")
    print("Path to dataset files:", path)

    # Try to locate the CSV file in the downloaded folder
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                print(f"Found CSV file: {file}")
                return pd.read_csv(os.path.join(root, file))

    raise FileNotFoundError(f"Could not find CSV file in downloaded dataset")


def prepare_data(df):
    """Clean and prepare the dataset for training"""
    # Drop rows with missing values
    df_clean = df.dropna()

    # Check for and handle non-numeric columns
    for col in df_clean.columns:
        if df_clean[col].dtype == "object":
            print(f"Converting column {col} to categorical")
            df_clean[col] = pd.Categorical(df_clean[col]).codes

    print(f"Dataset shape after cleaning: {df_clean.shape}")

    # Ensure target variable is binary (0 and 1)
    target_col = df_clean.columns[-1]  # Assuming last column is the target
    if df_clean[target_col].nunique() > 2:
        raise ValueError("Target column should be binary for logistic regression")

    # Ensure target values are 0 and 1
    if set(df_clean[target_col].unique()) != {0, 1}:
        print("Converting target values to 0 and 1")
        df_clean[target_col] = (
            df_clean[target_col] == df_clean[target_col].max()
        ).astype(int)

    return df_clean


def sigmoid(X, theta):
    z = X.dot(theta)
    # Clip values to avoid overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, theta):
    """Compute the cost function for logistic regression"""
    m = len(y)
    h = sigmoid(X, theta)

    # Avoid log(0) by adding a small epsilon
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1 - epsilon)

    cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost


def batchGradientDescent(X, y, learning_rate=0.01, n_iterations=1000, verbose=True):
    m, n = X.shape  # m = number of samples, n = number of features
    X_b = np.c_[np.ones((m, 1)), X]  # Add bias term → shape: (m, n+1)
    theta = np.zeros((n + 1, 1))  # Initialize theta with zeros

    # Store the cost history for plotting
    cost_history = []

    for iteration in range(n_iterations):
        predictions = sigmoid(X_b, theta)
        errors = predictions - y
        gradients = (1 / m) * X_b.T.dot(errors)
        theta = theta - learning_rate * gradients

        # Calculate cost and store it
        cost = compute_cost(X_b, y, theta)
        cost_history.append(cost)

        # Print progress if verbose
        if verbose and iteration % 100 == 0:
            print(f"Iteration {iteration}: Cost = {cost}")

    return theta, cost_history


def predict(X, theta, threshold=0.5):
    """Make binary predictions using the trained model"""
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
    y_prob = sigmoid(X_b, theta)
    return (y_prob >= threshold).astype(int), y_prob


def evaluate_model(y_true, y_pred):
    """Calculate and print performance metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("Model Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, f1, cm


def plot_learning_curve(cost_history):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Learning Curve")
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(cm, classes=["User", "Bot"]):
    """Plot the confusion matrix as a heatmap"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def plot_feature_importance(feature_names, theta):
    # Skip the bias term
    coefficients = theta[1:].flatten()

    # Sort features by importance (absolute value of coefficients)
    sorted_idx = np.argsort(np.abs(coefficients))
    sorted_features = np.array(feature_names)[sorted_idx]
    sorted_coeffs = coefficients[sorted_idx]

    plt.figure(figsize=(12, 8))
    plt.barh(sorted_features, np.abs(sorted_coeffs))
    plt.xlabel("Absolute Coefficient Value")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.grid(True, axis="x")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_test, y_prob):
    """Plot the ROC curve"""
    fpr, tpr, _ = roc_curve(y_test, y_prob.flatten())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def main():
    # Use the users-vs-bots dataset instead of iris
    try:
        print("Loading users vs bots dataset...")
        df = load_users_vs_bots_data()
        print("Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print("First few rows:")
        print(df.head())

        print("\nPreparing data...")
        df_clean = prepare_data(df)

        # Extract features and target
        X = df_clean.iloc[:, :-1].values  # All columns except the last one
        y = df_clean.iloc[:, -1].values.reshape(-1, 1)  # Last column as target

        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")

        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the model using batch gradient descent
        print("\nTraining the logistic regression model...")
        theta, cost_history = batchGradientDescent(
            X_train_scaled, y_train, learning_rate=0.1, n_iterations=1000
        )

        print("\nTraining completed!")

        # Make predictions
        print("\nMaking predictions on the test set...")
        y_pred, y_prob = predict(X_test_scaled, theta)

        # Evaluate the model
        print("\nEvaluating model performance...")
        accuracy, precision, recall, f1, cm = evaluate_model(y_test, y_pred)

        # Plotting
        print("\nGenerating plots...")

        # Plot learning curve
        plot_learning_curve(cost_history)

        # Plot confusion matrix
        plot_confusion_matrix(cm)

        # Plot feature importance
        feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
        if (
            df_clean.shape[1] <= 20
        ):  # Only use actual column names if there aren't too many
            feature_names = df_clean.columns[:-1].tolist()
        plot_feature_importance(feature_names, theta)

        # Plot ROC curve
        plot_roc_curve(y_test, y_prob)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
