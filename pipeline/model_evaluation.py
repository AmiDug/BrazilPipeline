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

    # Prepare test features
    X_test = test_df[features].copy()
    y_test = test_df[target_column].values

    # Scale features
    X_test_scaled = scaler.transform(X_test)

    # Generate predictions
    if best_model_name == 'neural_network':
        y_pred = best_model.predict(X_test_scaled).flatten()
    elif best_model_name == 'linear_model':
        y_pred = best_model.predict(X_test_scaled).flatten()
    else:
        raise ValueError(f"Unknown model type: {best_model_name}")

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error

    # Log metrics
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_r2", r2)
    mlflow.log_metric("test_mape", mape)

    # Create prediction vs actual scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)

    # Add perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.title(f'{best_model_name.replace("_", " ").title()} - Predicted vs Actual')
    plt.xlabel('Actual Price (R$)')
    plt.ylabel('Predicted Price (R$)')
    plt.grid(True)

    # Add metrics annotation to plot
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
    errors = y_pred - y_test
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=30, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Prediction Error Distribution')
    plt.xlabel('Prediction Error (R$)')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Save the figure for MLflow
    fig_path = "error_distribution.png"
    plt.savefig(fig_path)
    mlflow.log_artifact(fig_path)
    os.remove(fig_path)
    plt.close()

    # Create error by price range
    plt.figure(figsize=(12, 6))

    # Create price bins
    price_bins = [0, 50, 100, 200, 500, max(y_test)]
    bin_labels = ['0-50', '50-100', '100-200', '200-500', '500+']
    test_df['price_bin'] = pd.cut(test_df[target_column], bins=price_bins, labels=bin_labels)

    # Create temporary test_df with predictions
    temp_df = test_df.copy()
    temp_df['prediction'] = y_pred
    temp_df['abs_error'] = np.abs(temp_df['prediction'] - temp_df[target_column])
    temp_df['rel_error'] = temp_df['abs_error'] / temp_df[target_column] * 100

    # Plot absolute error by price bin
    plt.subplot(1, 2, 1)
    sns.boxplot(x='price_bin', y='abs_error', data=temp_df)
    plt.title('Absolute Error by Price Range')
    plt.xlabel('Price Range (R$)')
    plt.ylabel('Absolute Error (R$)')

    # Plot relative error by price bin
    plt.subplot(1, 2, 2)
    sns.boxplot(x='price_bin', y='rel_error', data=temp_df)
    plt.title('Relative Error (%) by Price Range')
    plt.xlabel('Price Range (R$)')
    plt.ylabel('Relative Error (%)')

    plt.tight_layout()

    # Save the figure for MLflow
    fig_path = "error_by_price_range.png"
    plt.savefig(fig_path)
    mlflow.log_artifact(fig_path)
    os.remove(fig_path)
    plt.close()

    # Analyze performance by product category
    # Get top categories by count
    top_categories = test_df['category'].value_counts().head(10).index.tolist()

    # Calculate metrics by category
    category_metrics = {}
    for category in top_categories:
        cat_mask = test_df['category'] == category
        cat_y_test = y_test[cat_mask]
        cat_y_pred = y_pred[cat_mask]

        if len(cat_y_test) > 0:
            cat_mae = mean_absolute_error(cat_y_test, cat_y_pred)
            cat_rmse = np.sqrt(mean_squared_error(cat_y_test, cat_y_pred))
            cat_r2 = r2_score(cat_y_test, cat_y_pred)

            category_metrics[category] = {
                'count': len(cat_y_test),
                'mae': cat_mae,
                'rmse': cat_rmse,
                'r2': cat_r2
            }

    # Create metrics by category plot
    if category_metrics:
        plt.figure(figsize=(14, 8))

        # Convert to DataFrame for easier plotting
        metrics_df = pd.DataFrame.from_dict(category_metrics, orient='index')

        # Sort by MAE
        metrics_df = metrics_df.sort_values('mae')

        # Plot MAE by category
        plt.subplot(1, 2, 1)
        plt.barh(metrics_df.index, metrics_df['mae'])
        plt.title('MAE by Category')
        plt.xlabel('Mean Absolute Error (R$)')

        # Plot R² by category
        plt.subplot(1, 2, 2)
        plt.barh(metrics_df.index, metrics_df['r2'])
        plt.title('R² by Category')
        plt.xlabel('R² Score')

        plt.tight_layout()

        # Save the figure for MLflow
        fig_path = "metrics_by_category.png"
        plt.savefig(fig_path)
        mlflow.log_artifact(fig_path)
        os.remove(fig_path)
        plt.close()

        # Log best and worst performing categories
        best_cat = metrics_df['mae'].idxmin()
        worst_cat = metrics_df['mae'].idxmax()

        mlflow.log_param("best_category", best_cat)
        mlflow.log_param("worst_category", worst_cat)
        mlflow.log_metric("best_category_mae", metrics_df.loc[best_cat, 'mae'])
        mlflow.log_metric("worst_category_mae", metrics_df.loc[worst_cat, 'mae'])

    # Print evaluation results
    print(f"Evaluation of {best_model_name} model on test data:")
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