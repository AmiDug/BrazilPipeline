import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
import mlflow


# Helper function to convert DataFrame to DMatrix when needed
def to_dmatrix(df, label=None):
    """Convert DataFrame to XGBoost DMatrix format"""
    if isinstance(df, xgb.DMatrix):
        return df
    if label is not None:
        return xgb.DMatrix(data=df, label=label)
    return xgb.DMatrix(data=df)


def model_training(data_splits, target_column='price'):
    """Train model using XGBoost with GPU acceleration"""
    print("Starting GPU-accelerated model training...")

    # Extract training data
    train_df = data_splits['train']
    features = data_splits.get('features', [col for col in train_df.columns
                                            if col != 'price' and col != 'product_id'])

    # Prepare training data
    X_train = train_df[features].copy()
    y_train = train_df[target_column].values

    # Extract test data
    test_df = data_splits['test']
    X_test = test_df[features].copy()
    y_test = test_df[target_column].values

    print(f"Training data: {X_train.shape[0]} samples, {len(features)} features")

    # Dict to store model results
    model_results = {}

    # Create XGBoost DMatrix objects (optimized data structure)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # XGBoost parameters to approximate GradientBoostingRegressor
    # with n_estimators=200, max_depth=11
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',  # Fast histogram algorithm
        'device': 'cuda',  # Use GPU acceleration (modern syntax)
        'max_depth': 11,  # Same as notebook
        'learning_rate': 0.1,  # Same as notebook
        'min_child_weight': 5,  # Similar to min_samples_split
        'subsample': 1.0,  # Use all data
        'colsample_bytree': 1.0,  # Use all features
        'seed': 100  # Same random seed
    }

    print("Training XGBoost model with GPU acceleration...")

    # Train model
    num_round = 200  # Same as n_estimators in the notebook
    xgb_model = xgb.train(params, dtrain, num_round)

    # Make predictions
    y_pred = xgb_model.predict(dtest)

    # Calculate metrics
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"XGBoost GPU - Test R²: {test_r2:.4f}")
    print(f"XGBoost GPU - Test RMSE: {test_rmse:.4f}")

    # Get feature importance
    importance_scores = xgb_model.get_score(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': list(importance_scores.keys()),
        'importance': list(importance_scores.values())
    }).sort_values('importance', ascending=False)

    print("Top 10 features by importance:")
    print(feature_importance.head(10))

    model_results['xgboost_gpu'] = {
        'model': xgb_model,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'feature_importance': feature_importance
    }

    # Log metrics to MLflow
    try:
        mlflow.log_metric("xgb_r2", test_r2)
        mlflow.log_metric("xgb_rmse", test_rmse)
    except:
        print("Warning: Could not log to MLflow")

    # Create a custom predictor function that handles DataFrame inputs
    def predict_fn(X_df):
        """Prediction function that handles DataFrame input by converting to DMatrix"""
        X_dmatrix = to_dmatrix(X_df)
        return xgb_model.predict(X_dmatrix)

    # Add the predict function to the results
    model_results['xgboost_gpu']['predict_fn'] = predict_fn

    return {
        "models": model_results,
        "best_model_name": "xgboost_gpu",
        "features": features,
        "feature_importance": feature_importance,
        "predict_fn": predict_fn  # Add prediction function to top level
    }