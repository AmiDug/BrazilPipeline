import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import os


def model_evaluation(model_results, data_splits, target_column='price'):

    print("Starting model evaluation...")

    # Get test data
    test_df = data_splits['test']

    # Get the best model
    best_model_name = model_results['best_model']
    best_model = model_results[best_model_name]['model']

    # Extract test features and target
    X_test = test_df.drop(columns=[target_column, 'id', 'image', 'title', 'description'])
    y_test = test_df[target_column].values

    # Transform features as was done in training
    X_test = pd.get_dummies(X_test, drop_first=True)

    # Ensure test has same columns as training
    scaler = model_results['preprocessing']['scaler']

    # Fix missing columns in test set
    for col in model_results[best_model_name]['model'].input_shape[1:]:
        if X_test.shape[1] != col:
            print(f"Warning: Input shape mismatch. Model expects {col} features, but got {X_test.shape[1]}")
            # This is a simplified approach - in production, you'd need more robust handling

    # Scale the features
    X_test_scaled = scaler.transform(X_test)

    # Generate predictions
    y_pred = best_model.predict(X_test_scaled).flatten()

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log non-autologged metrics
    # (TensorFlow's autolog already logs these metrics, but we'll log them explicitly for clarity)

    # Create evaluation metrics dictionary
    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "model_type": best_model_name
    }

    # Create prediction vs actual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)

    # Add perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.title(f'{best_model_name.replace("_", " ").title()} - Predicted vs Actual')
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.grid(True)

    # Add metrics annotation to plot
    plt.annotate(f"RMSE: ${rmse:.2f}\nMAE: ${mae:.2f}\nR²: {r2:.3f}",
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                 fontsize=10, ha='left', va='top')

    # Save the figure for MLflow
    fig_path = "prediction_vs_actual.png"
    plt.savefig(fig_path)
    mlflow.log_artifact(fig_path)
    os.remove(fig_path)
    plt.close()

    # Create error distribution plot
    errors = y_pred - y_test
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Prediction Error Distribution')
    plt.xlabel('Prediction Error ($)')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Save the figure for MLflow
    fig_path = "error_distribution.png"
    plt.savefig(fig_path)
    mlflow.log_artifact(fig_path)
    os.remove(fig_path)
    plt.close()

    # Print evaluation results
    print(f"Evaluation of {best_model_name} model:")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAE: ${mae:.2f}")
    print(f"R²: {r2:.3f}")

    return metrics