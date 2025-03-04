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
    log_transform = model_results['preprocessing']['log_transform']

    # Prepare test features
    X_test = test_df[features].copy()

    # Determine which target to use for evaluation
    if log_transform and 'log_price_target' in test_df.columns:
        y_test_transformed = test_df['log_price_target'].values
        y_test_original = test_df[target_column].values
    else:
        y_test_transformed = test_df[target_column].values
        y_test_original = y_test_transformed

    # Scale features
    X_test_scaled = scaler.transform(X_test)

    # Generate predictions
    if best_model_name == 'ensemble':
        # For ensemble, combine predictions from multiple models
        gb_pred = model_results['gradient_boosting']['model'].predict(X_test_scaled)
        xgb_pred = model_results['xgboost']['model'].predict(X_test_scaled)
        weights = best_model['weights']
        y_pred_transformed = weights[0] * gb_pred + weights[1] * xgb_pred
    elif best_model_name == 'neural_network':
        y_pred_transformed = best_model.predict(X_test_scaled).flatten()
    else:
        y_pred_transformed = best_model.predict(X_test_scaled)

    # If we used log transformation, convert predictions back to original scale
    if log_transform:
        y_pred_original = np.expm1(y_pred_transformed)
    else:
        y_pred_original = y_pred_transformed

    # Calculate metrics on original scale
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    mae = mean_absolute_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)

    # Calculate MAPE safely
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_test_original - y_pred_original) / np.maximum(y_test_original, 0.01))) * 100

    # Log metrics
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_r2", r2)
    mlflow.log_metric("test_mape", mape)

    # Calculate weighted MAPE (gives more weight to higher-priced items)
    weights = y_test_original / np.sum(y_test_original)
    weighted_errors = weights * np.abs((y_test_original - y_pred_original) / np.maximum(y_test_original, 0.01))
    weighted_mape = np.sum(weighted_errors) * 100
    mlflow.log_metric("test_weighted_mape", weighted_mape)

    # Create prediction vs actual scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_original, y_pred_original, alpha=0.5, s=10)

    # Add perfect prediction line
    min_val = min(min(y_test_original), min(y_pred_original))
    max_val = max(max(y_test_original), max(y_pred_original))
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
    errors = y_pred_original - y_test_original
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

    # Feature importance analysis
    if best_model_name in ['gradient_boosting', 'xgboost'] and f'feature_importance' in model_results[best_model_name]:
        feature_importance = model_results[best_model_name]['feature_importance']

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

    # Create price range accuracy analysis
    plt.figure(figsize=(12, 6))

    # Create price buckets
    price_buckets = [0, 50, 100, 200, 500, 1000, float('inf')]
    bucket_labels = ['0-50', '50-100', '100-200', '200-500', '500-1000', '1000+']

    test_df['price_bucket'] = pd.cut(test_df[target_column], bins=price_buckets, labels=bucket_labels)
    test_df['prediction'] = y_pred_original
    test_df['abs_error'] = np.abs(test_df['prediction'] - test_df[target_column])
    test_df['rel_error'] = test_df['abs_error'] / test_df[target_column]

    # Calculate error metrics by price bucket
    bucket_metrics = test_df.groupby('price_bucket').agg({
        'abs_error': 'mean',
        'rel_error': lambda x: np.mean(x) * 100,  # Convert to percentage
        target_column: 'count'
    }).reset_index()

    bucket_metrics.columns = ['Price Range', 'MAE', 'MAPE (%)', 'Count']

    # Plot MAPE by price range
    ax = bucket_metrics.plot(x='Price Range', y='MAPE (%)', kind='bar', figsize=(12, 6))

    # Add count labels on top of bars
    for i, count in enumerate(bucket_metrics['Count']):
        plt.text(i, bucket_metrics['MAPE (%)'][i] + 1, f'n={count}', ha='center')

    plt.title('Error by Price Range')
    plt.ylabel('Mean Absolute Percentage Error (%)')
    plt.grid(axis='y')
    plt.tight_layout()

    # Save figure
    fig_path = "error_by_price_range.png"
    plt.savefig(fig_path)
    mlflow.log_artifact(fig_path)
    os.remove(fig_path)
    plt.close()

    # Print evaluation results
    print(f"\nEvaluation of {best_model_name} model on test data:")
    print(f"RMSE: R${rmse:.2f}")
    print(f"MAE: R${mae:.2f}")
    print(f"R²: {r2:.3f}")
    print(f"MAPE: {mape:.1f}%")
    print(f"Weighted MAPE: {weighted_mape:.1f}%")

    # Create summary metrics dictionary
    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mape": float(mape),
        "weighted_mape": float(weighted_mape),
        "model_type": best_model_name
    }

    return metrics