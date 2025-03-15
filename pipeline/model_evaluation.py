import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import mlflow
import xgboost as xgb


def model_evaluation(training_results, data_splits, target_column='price'):
    """Evaluate model performance on test data with XGBoost compatibility"""
    print("Starting model evaluation...")

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

    # Make predictions based on model type
    if isinstance(model, xgb.Booster):
        # For XGBoost models, use the DMatrix approach
        dtest = xgb.DMatrix(X_test)
        y_pred = model.predict(dtest)
    else:
        # For sklearn models, use the standard approach
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

    # Log metrics to MLflow
    try:
        # Check if there's an active run before starting a new one
        active_run = mlflow.active_run()
        if active_run is None:
            mlflow.start_run()

        mlflow.log_metric("r2", r2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)
        mlflow.log_param("model_type", best_model_name)

        # Only end the run if we started it
        if active_run is None:
            mlflow.end_run()
    except Exception as e:
        print(f"Warning: MLflow issue: {e}")

    # Create a DataFrame with actual vs predicted values
    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'error': y_test - y_pred,
        'abs_error': np.abs(y_test - y_pred)
    })

    # Collect all metrics into a dictionary
    metrics = {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mape": mape
    }

    # Return evaluation results
    return {
        "metrics": metrics,  # Add consolidated metrics dictionary
        "results_df": results_df,
        "best_model_name": best_model_name  # Add this for pipeline compatibility
    }