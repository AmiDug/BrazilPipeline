import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import mlflow
import os


def model_training(data_splits, target_column='price', log_transform=False):
    print("Starting model training...")

    # Extract data
    train_df = data_splits['train']

    # FIXED: Remove log transformation to prevent leakage
    target = target_column
    print(f"Using original price as target")

    # Get selected features from data_transformation
    if 'selected_features' in data_splits:
        valid_numeric_features = data_splits['selected_features']
    else:
        # Fallback to basic features if not provided
        valid_numeric_features = [
            'product_weight_g', 'volume_cm3', 'freight_value',
            'description_length', 'image_count', 'density', 'count',
            'category_code'
        ]
        # Only use features that actually exist
        valid_numeric_features = [f for f in valid_numeric_features if f in train_df.columns]

    # FIXED: Remove any price-derived features that might have slipped through
    price_derived_features = [
        'price_min', 'price_max', 'price_variance', 'price_per_gram',
        'price_to_freight_ratio', 'description_price_ratio', 'price_cv',
        'category_avg_price', 'log_price', 'category_mean_price',
        'price_ratio'
    ]
    valid_numeric_features = [f for f in valid_numeric_features if f not in price_derived_features]

    # Exclude target and ID columns from features
    exclude_cols = [target_column, 'id', 'product_id', 'seller_id', 'customer_id', 'order_id']
    valid_numeric_features = [f for f in valid_numeric_features if f not in exclude_cols]

    print(f"Using {len(valid_numeric_features)} features: {valid_numeric_features}")

    # Select X and y
    X_train = train_df[valid_numeric_features].copy()
    y_train = train_df[target].values

    # Log feature selection (using different parameter name)
    mlflow.log_param("training_features", ", ".join(valid_numeric_features))
    mlflow.log_param("target_variable", target)
    mlflow.log_param("log_transform", "False")  # Always false now

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
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
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
            'feature': valid_numeric_features,
            'importance': gb_model.feature_importances_
        }).sort_values('importance', ascending=False)

    # ---- Model 3: Neural Network ----
    with mlflow.start_run(nested=True, run_name="Neural_Network"):
        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # Create neural network with regularization
        input_shape = X_train_final.shape[1]
        inputs = tf.keras.Input(shape=(input_shape,))
        x = layers.Dense(32, activation='relu', kernel_regularizer=l2(0.01))(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(16, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Compile model
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        # Train model
        history = model.fit(
            X_train_final, y_train_final,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )

        # Evaluate on validation data
        nn_val_pred = model.predict(X_val).flatten()
        nn_val_mse = mean_squared_error(y_val, nn_val_pred)
        nn_val_rmse = np.sqrt(nn_val_mse)
        nn_val_r2 = r2_score(y_val, nn_val_pred)

        # Log metrics
        mlflow.log_metric("val_mse", nn_val_mse)
        mlflow.log_metric("val_rmse", nn_val_rmse)
        mlflow.log_metric("val_r2", nn_val_r2)

        print(f"Neural Network - Validation RMSE: {nn_val_rmse:.4f}, R²: {nn_val_r2:.4f}")

    # Determine best model
    model_names = ['ridge_regression', 'gradient_boosting', 'neural_network']
    r2_values = [ridge_val_r2, gb_val_r2, nn_val_r2]
    best_model_idx = np.argmax(r2_values)
    best_model_name = model_names[best_model_idx]

    # Log best model
    if best_model_name == 'ridge_regression':
        best_model = ridge_model
        best_val_rmse = ridge_val_rmse
        best_val_r2 = ridge_val_r2
    elif best_model_name == 'gradient_boosting':
        best_model = gb_model
        best_val_rmse = gb_val_rmse
        best_val_r2 = gb_val_r2
    else:  # Neural Network
        best_model = model
        best_val_rmse = nn_val_rmse
        best_val_r2 = nn_val_r2

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
        "neural_network": {
            "model": model,
            "history": history.history,
            "rmse": nn_val_rmse,
            "r2": nn_val_r2
        },
        "best_model": best_model_name,
        "preprocessing": {
            "scaler": scaler,
            "features": valid_numeric_features,
            "log_transform": False  # Always false now
        }
    }