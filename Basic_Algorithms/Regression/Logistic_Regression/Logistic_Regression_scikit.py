#!/usr/bin/env python3
# Logistic Regression using scikit-learn with users vs bots classification dataset

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import kagglehub
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
)


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
    return df_clean


# Load the users vs bots dataset
print("Loading users vs bots dataset...")
try:
    df = load_users_vs_bots_data()

    # Display the first few rows of the dataset
    print("First 5 rows of the dataset:")
    print(df.head())

    # Clean and prepare the data
    df_clean = prepare_data(df)

    # Display dataset info
    print("\nDataset Information:")
    print(f"Number of samples: {df_clean.shape[0]}")
    print(f"Number of features: {df_clean.shape[1] - 1}")  # Excluding the target column

    # Assuming the target column is the last column
    target_col = df_clean.columns[-1]
    print(f"Class distribution:\n{df_clean[target_col].value_counts()}")

    # Split the data into features (X) and target (y)
    X = df_clean.iloc[:, :-1].values  # All rows, all columns except the last one
    y = df_clean.iloc[:, -1].values  # All rows, only the last column

    # Split the dataset into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling to normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create and train the logistic regression model
    # For binary classification, we use 'liblinear' solver
    model = LogisticRegression(max_iter=1000, random_state=42, solver="liblinear")
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class

    # Evaluate the model
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, zero_division=0):.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["User", "Bot"],
        yticklabels=["User", "Bot"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
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

    # Feature importance
    if df_clean.shape[1] <= 20:  # Only show for reasonable number of features
        feature_names = df_clean.columns[:-1].tolist()
        coefficients = model.coef_[0]

        # Sort by absolute values
        sorted_idx = np.argsort(np.abs(coefficients))

        plt.figure(figsize=(12, 8))
        plt.barh([feature_names[i] for i in sorted_idx], coefficients[sorted_idx])
        plt.xlabel("Coefficient Value")
        plt.ylabel("Feature")
        plt.title("Feature Importance")
        plt.grid(True, axis="x")
        plt.tight_layout()
        plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
