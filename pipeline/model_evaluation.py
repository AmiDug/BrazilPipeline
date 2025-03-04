import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import os


def model_evaluation(model_results, data_splits, target_column='price'):
    print("Starting model evaluation...")

    # Extract test data
    test_df = data_splits['test']

    # Get the best model and preprocessing
    best_model_name = model_results['best_model']
    best_model = model_results[best_model_name]['model']
    scaler = model_results['preprocessing']['scaler']
    features = model_results['preprocessing']['features']

    # FIXED: No log transform
    log_transform = False

    # Prepare test features
    X_test = test_df[features].copy()
    y_test = test_df[target_column].values

    # Scale features
    X_test_scaled = scaler.transform(X_test)

    # Generate predictions
    if best_model_name == 'neural_network':
        y_pred = best_model.predict(X_test_scaled).flatten()
    else:
        y_pred = best_model.predict(X_test_scaled)

    # Calculate metrics on original scale
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calculate MAPE safely
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 0.01))) * 100

    # Log metrics
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_r2", r2)
    mlflow.log_metric("test_mape", mape)

    # Create prediction vs actual scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, s=10)

    # Add perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.title(f'{best_model_name.replace("_", " ").title()} - Predicted vs Actual')
    plt.xlabel('Actual Price (R$)')
    plt.ylabel('Predicted Price (R$)')
    plt.grid(True)

    # Add metrics annotation
    plt.annotate(f"RMSE: R${rmse:.2f}\nMAE: R${mae:.2f}\nR²: {r2:.3f}\nMAPE: {mape:.1f}%",
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
    plt.figure(figsize=(10, 6))
    errors = y_pred - y_test
    sns.histplot(errors, bins=30, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Prediction Error Distribution')
    plt.xlabel('Prediction Error (R$)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()

    # Save the figure for MLflow
    fig_path = "error_distribution.png"
    plt.savefig(fig_path)
    mlflow.log_artifact(fig_path)
    os.remove(fig_path)
    plt.close()

    # Feature importance analysis (for Gradient Boosting)
    if best_model_name == 'gradient_boosting' and 'feature_importance' in model_results['gradient_boosting']:
        feature_importance = model_results['gradient_boosting']['feature_importance']

        plt.figure(figsize=(12, 8))
        plt.barh(feature_importance['feature'][:15], feature_importance['importance'][:15])
        plt.title('Top 15 Feature Importances')
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()

        # Save figure
        fig_path = "feature_importance.png"
        plt.savefig(fig_path)
        mlflow.log_artifact(fig_path)
        os.remove(fig_path)
        plt.close()

        print("\nTop 5 Important Features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

    # Print evaluation results
    print(f"\nEvaluation of {best_model_name} model on test data:")
    print(f"RMSE: R${rmse:.2f}")
    print(f"MAE: R${mae:.2f}")
    print(f"R²: {r2:.3f}")
    print(f"MAPE: {mape:.1f}%")

    # Create summary metrics dictionary
    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mape": float(mape),
        "model_type": best_model_name
    }

    return metrics