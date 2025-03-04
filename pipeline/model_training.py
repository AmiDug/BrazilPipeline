import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import mlflow


def model_training(data_splits, target_column='price', log_transform=True):
    print("Starting model training...")

    # Extract data
    train_df = data_splits['train']

    # Use log-transformed target if specified
    if log_transform and 'log_price_target' in train_df.columns:
        target = 'log_price_target'
        print(f"Using log-transformed price as target")
    else:
        target = target_column
        print(f"Using original price as target")

    # Get selected features from data_transformation
    if 'selected_features' in data_splits:
        valid_features = data_splits['selected_features']
    else:
        # Fallback to basic features if not provided
        valid_features = [
            'product_weight_g', 'volume_cm3', 'freight_value',
            'description_length', 'image_count', 'density', 'count',
            'category_code', 'freight_value_min', 'freight_value_mean'
        ]
        valid_features = [f for f in valid_features if f in train_df.columns]

    # Exclude target and ID columns from features
    exclude_cols = [target_column, 'id', 'product_id', 'seller_id', 'customer_id', 'order_id', 'log_price_target']
    valid_features = [f for f in valid_features if f not in exclude_cols]

    print(f"Using {len(valid_features)} features: {valid_features}")

    # Select X and y
    X_train = train_df[valid_features].copy()
    y_train = train_df[target].values

    # Log feature selection
    mlflow.log_param("training_features", ", ".join(valid_features))
    mlflow.log_param("target_variable", target)
    mlflow.log_param("log_transform", str(log_transform))

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Create validation split
    val_size = int(0.2 * len(X_train))
    X_val = X_train_scaled[-val_size:]
    y_val = y_train[-val_size:]
    X_train_final = X_train_scaled[:-val_size]
    y_train_final = y_train[:-val_size]

    # ---- Model 1: Ridge Regression ----
    with mlflow.start_run(nested=True, run_name="Ridge_Regression"):
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train_final, y_train_final)

        # Evaluate on validation data
        ridge_val_pred = ridge_model.predict(X_val)
        ridge_val_mse = mean_squared_error(y_val, ridge_val_pred)
        ridge_val_rmse = np.sqrt(ridge_val_mse)
        ridge_val_r2 = r2_score(y_val, ridge_val_pred)

        # Log metrics
        mlflow.log_metric("val_mse", ridge_val_mse)
        mlflow.log_metric("val_rmse", ridge_val_rmse)
        mlflow.log_metric("val_r2", ridge_val_r2)

        print(f"Ridge Regression - Validation RMSE: {ridge_val_rmse:.4f}, R²: {ridge_val_r2:.4f}")

    # ---- Model 2: Gradient Boosting ----
    with mlflow.start_run(nested=True, run_name="Gradient_Boosting"):
        gb_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=10,
            random_state=42
        )

        # Train model
        gb_model.fit(X_train_final, y_train_final)

        # Evaluate on validation data
        gb_val_pred = gb_model.predict(X_val)
        gb_val_mse = mean_squared_error(y_val, gb_val_pred)
        gb_val_rmse = np.sqrt(gb_val_mse)
        gb_val_r2 = r2_score(y_val, gb_val_pred)

        # Log metrics
        mlflow.log_metric("val_mse", gb_val_mse)
        mlflow.log_metric("val_rmse", gb_val_rmse)
        mlflow.log_metric("val_r2", gb_val_r2)

        print(f"Gradient Boosting - Validation RMSE: {gb_val_rmse:.4f}, R²: {gb_val_r2:.4f}")

        # Log feature importances
        feature_importance = pd.DataFrame({
            'feature': valid_features,
            'importance': gb_model.feature_importances_
        }).sort_values('importance', ascending=False)

    # ---- Model 3: XGBoost (New) ----
    with mlflow.start_run(nested=True, run_name="XGBoost"):
        xgb_model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        )

        # Train model
        xgb_model.fit(X_train_final, y_train_final)

        # Evaluate on validation data
        xgb_val_pred = xgb_model.predict(X_val)
        xgb_val_mse = mean_squared_error(y_val, xgb_val_pred)
        xgb_val_rmse = np.sqrt(xgb_val_mse)
        xgb_val_r2 = r2_score(y_val, xgb_val_pred)

        # Log metrics
        mlflow.log_metric("val_mse", xgb_val_mse)
        mlflow.log_metric("val_rmse", xgb_val_rmse)
        mlflow.log_metric("val_r2", xgb_val_r2)

        print(f"XGBoost - Validation RMSE: {xgb_val_rmse:.4f}, R²: {xgb_val_r2:.4f}")

        # Log XGBoost feature importances
        xgb_feature_importance = pd.DataFrame({
            'feature': valid_features,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

    # ---- Model 4: Ensemble of GB and XGB (New) ----
    with mlflow.start_run(nested=True, run_name="Ensemble"):
        # Simple weighted average of predictions
        ensemble_val_pred = 0.4 * gb_val_pred + 0.6 * xgb_val_pred
        ensemble_val_mse = mean_squared_error(y_val, ensemble_val_pred)
        ensemble_val_rmse = np.sqrt(ensemble_val_mse)
        ensemble_val_r2 = r2_score(y_val, ensemble_val_pred)

        # Log metrics
        mlflow.log_metric("val_mse", ensemble_val_mse)
        mlflow.log_metric("val_rmse", ensemble_val_rmse)
        mlflow.log_metric("val_r2", ensemble_val_r2)

        print(f"Ensemble - Validation RMSE: {ensemble_val_rmse:.4f}, R²: {ensemble_val_r2:.4f}")

    # Determine best model
    model_names = ['ridge_regression', 'gradient_boosting', 'xgboost', 'ensemble']
    r2_values = [ridge_val_r2, gb_val_r2, xgb_val_r2, ensemble_val_r2]
    best_model_idx = np.argmax(r2_values)
    best_model_name = model_names[best_model_idx]

    # Define best model object
    if best_model_name == 'ridge_regression':
        best_model = ridge_model
        best_val_rmse = ridge_val_rmse
        best_val_r2 = ridge_val_r2
    elif best_model_name == 'gradient_boosting':
        best_model = gb_model
        best_val_rmse = gb_val_rmse
        best_val_r2 = gb_val_r2
    elif best_model_name == 'xgboost':
        best_model = xgb_model
        best_val_rmse = xgb_val_rmse
        best_val_r2 = xgb_val_r2
    else:  # Ensemble
        # For ensemble, store both models
        best_model = {'gb_model': gb_model, 'xgb_model': xgb_model, 'weights': [0.4, 0.6]}
        best_val_rmse = ensemble_val_rmse
        best_val_r2 = ensemble_val_r2

    mlflow.log_param("best_model", best_model_name)
    mlflow.log_metric("best_model_rmse", best_val_rmse)
    mlflow.log_metric("best_model_r2", best_val_r2)

    print(f"Best model: {best_model_name} with R²: {best_val_r2:.4f} and RMSE: {best_val_rmse:.4f}")

    # Return models and preprocessing info
    return {
        "ridge_regression": {
            "model": ridge_model,
            "rmse": ridge_val_rmse,
            "r2": ridge_val_r2
        },
        "gradient_boosting": {
            "model": gb_model,
            "rmse": gb_val_rmse,
            "r2": gb_val_r2,
            "feature_importance": feature_importance
        },
        "xgboost": {
            "model": xgb_model,
            "rmse": xgb_val_rmse,
            "r2": xgb_val_r2,
            "feature_importance": xgb_feature_importance
        },
        "ensemble": {
            "model": {'gb_model': gb_model, 'xgb_model': xgb_model, 'weights': [0.4, 0.6]},
            "rmse": ensemble_val_rmse,
            "r2": ensemble_val_r2
        },
        "best_model": best_model_name,
        "preprocessing": {
            "scaler": scaler,
            "features": valid_features,
            "log_transform": log_transform
        }
    }