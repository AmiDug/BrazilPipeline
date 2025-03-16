import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import mlflow
import xgboost as xgb


def model_evaluation(training_results, data_splits, target_column='price'):
    """Evaluate model performance on test data with compatibility for all GPU-accelerated models"""
    print("Starting model evaluation for GPU-accelerated models...")

    # Extract test data
    test_df = data_splits['test']

    # Get features and target
    features = data_splits.get('features')
    X_test = test_df[features].copy()
    y_test = test_df[target_column].values

    # Get best model
    best_model_name = training_results.get('best_model_name', 'xgboost_gpu')
    model_info = training_results['models'][best_model_name]

    # Get the model and prediction function
    model = model_info.get('model')

    # Use the custom prediction function if available
    if 'predict_fn' in model_info:
        print(f"Using custom prediction function for {best_model_name}")
        y_pred = model_info['predict_fn'](X_test)
    # For XGBoost models, use the DMatrix approach
    elif isinstance(model, xgb.Booster):
        dtest = xgb.DMatrix(X_test)
        y_pred = model.predict(dtest)
    # For all other models, use the standard approach
    else:
        y_pred = model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Calculate MAPE
    differences = np.abs(y_test - y_pred)
    mape = np.mean(differences / y_test) * 100

    print(f"Model: {best_model_name}")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # Create a DataFrame with actual vs predicted values
    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'error': y_test - y_pred,
        'abs_error': np.abs(y_test - y_pred),
        'pct_error': differences / y_test * 100
    })

    # Handle potential NaN or Inf values in pct_error
    results_df['pct_error'] = results_df['pct_error'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Add error analysis by price range
    bins = [0, 50, 100, 200, 500, 1000, float('inf')]
    bin_labels = ['0-50', '50-100', '100-200', '200-500', '500-1000', '1000plus']
    results_df['price_range'] = pd.cut(results_df['actual'], bins=bins, labels=bin_labels)

    # Calculate error statistics by price range
    error_by_range = results_df.groupby('price_range', observed=True).agg({
        'actual': 'count',
        'abs_error': 'mean',
        'pct_error': 'mean'
    }).rename(columns={
        'actual': 'count',
        'abs_error': 'mean_abs_error',
        'pct_error': 'mean_pct_error'
    })

    print("\nError by Price Range:")
    print(error_by_range)

    # Collect all metrics into a dictionary
    metrics = {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mape": mape
    }

    # Log final evaluation metrics and summary
    mlflow.log_metric("final_r2", r2)
    mlflow.log_metric("final_rmse", rmse)
    mlflow.log_metric("final_mae", mae)
    mlflow.log_metric("final_mape", mape)
    mlflow.log_param("selected_model", best_model_name)

    # Log error analysis by price range
    for price_range in error_by_range.index:
        # Replace '+' with 'plus' for MLflow compatibility
        metric_name_safe = str(price_range).replace('+', 'plus')
        mlflow.log_metric(f"mae_{metric_name_safe}", error_by_range.loc[price_range, 'mean_abs_error'])
        mlflow.log_metric(f"mape_{metric_name_safe}", error_by_range.loc[price_range, 'mean_pct_error'])

    # Return evaluation results with enhanced error analysis
    return {
        "metrics": metrics,
        "results_df": results_df,
        "error_by_range": error_by_range,
        "best_model_name": best_model_name
    }