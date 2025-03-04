import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import mlflow


def model_training(data_splits, target_column='price'):
    """
    Train multiple models on the prepared data and select the best one
    """
    print("Starting model training...")

    # Extract training data
    train_df = data_splits['train']

    # Extract features to use
    if 'features' in data_splits:
        features = data_splits['features']
    else:
        # Default to all columns except price and id
        features = [col for col in train_df.columns if col != 'price' and col != 'product_id']

    # Prepare training data
    X_train = train_df[features].copy()
    y_train = train_df[target_column].values

    # Prepare validation split
    val_size = int(0.2 * len(X_train))
    X_val = X_train.iloc[-val_size:].copy()
    y_val = y_train[-val_size:]
    X_train_final = X_train.iloc[:-val_size].copy()
    y_train_final = y_train[:-val_size]

    print(f"Training data: {X_train_final.shape[0]} samples, {X_train_final.shape[1]} features")
    print(f"Validation data: {X_val.shape[0]} samples")

    # Log feature list
    try:
        mlflow.log_param("feature_count", len(features))
        top_features = min(10, len(features))
        mlflow.log_param("top_features", ", ".join(features[:top_features]))
    except:
        print("Warning: Could not log to MLflow (continuing)")

    # Dictionary to store all models and their performance
    model_results = {}

    # 1. Decision Tree
    with mlflow.start_run(nested=True, run_name="Decision_Tree"):
        print("Training Decision Tree model...")
        dt_model = DecisionTreeRegressor(
            max_depth=8,
            min_samples_split=10,
            random_state=42
        )
        dt_model.fit(X_train_final, y_train_final)

        # Evaluate on validation data
        dt_val_pred = dt_model.predict(X_val)
        dt_val_mse = mean_squared_error(y_val, dt_val_pred)
        dt_val_rmse = np.sqrt(dt_val_mse)
        dt_val_r2 = r2_score(y_val, dt_val_pred)

        # Log metrics
        try:
            mlflow.log_metric("val_mse", dt_val_mse)
            mlflow.log_metric("val_rmse", dt_val_rmse)
            mlflow.log_metric("val_r2", dt_val_r2)
        except:
            pass

        print(f"Decision Tree - Validation RMSE: {dt_val_rmse:.2f}, R²: {dt_val_r2:.2f}")

        # Feature importance
        dt_importance = pd.DataFrame({
            'feature': features,
            'importance': dt_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Decision Tree feature importances:")
        for idx, row in dt_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        # Store model and metrics
        model_results['decision_tree'] = {
            'model': dt_model,
            'val_rmse': dt_val_rmse,
            'val_r2': dt_val_r2,
            'feature_importance': dt_importance
        }

    # 2. Random Forest
    with mlflow.start_run(nested=True, run_name="Random_Forest"):
        print("Training Random Forest model...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_final, y_train_final)

        # Evaluate on validation data
        rf_val_pred = rf_model.predict(X_val)
        rf_val_mse = mean_squared_error(y_val, rf_val_pred)
        rf_val_rmse = np.sqrt(rf_val_mse)
        rf_val_r2 = r2_score(y_val, rf_val_pred)

        # Log metrics
        try:
            mlflow.log_metric("val_mse", rf_val_mse)
            mlflow.log_metric("val_rmse", rf_val_rmse)
            mlflow.log_metric("val_r2", rf_val_r2)
        except:
            pass

        print(f"Random Forest - Validation RMSE: {rf_val_rmse:.2f}, R²: {rf_val_r2:.2f}")

        # Feature importance
        rf_importance = pd.DataFrame({
            'feature': features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Random Forest feature importances:")
        for idx, row in rf_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        # Store model and metrics
        model_results['random_forest'] = {
            'model': rf_model,
            'val_rmse': rf_val_rmse,
            'val_r2': rf_val_r2,
            'feature_importance': rf_importance
        }

    # 3. Gradient Boosting
    with mlflow.start_run(nested=True, run_name="Gradient_Boosting"):
        print("Training Gradient Boosting model...")
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(X_train_final, y_train_final)

        # Evaluate on validation data
        gb_val_pred = gb_model.predict(X_val)
        gb_val_mse = mean_squared_error(y_val, gb_val_pred)
        gb_val_rmse = np.sqrt(gb_val_mse)
        gb_val_r2 = r2_score(y_val, gb_val_pred)

        # Log metrics
        try:
            mlflow.log_metric("val_mse", gb_val_mse)
            mlflow.log_metric("val_rmse", gb_val_rmse)
            mlflow.log_metric("val_r2", gb_val_r2)
        except:
            pass

        print(f"Gradient Boosting - Validation RMSE: {gb_val_rmse:.2f}, R²: {gb_val_r2:.2f}")

        # Feature importance
        gb_importance = pd.DataFrame({
            'feature': features,
            'importance': gb_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Gradient Boosting feature importances:")
        for idx, row in gb_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        # Store model and metrics
        model_results['gradient_boosting'] = {
            'model': gb_model,
            'val_rmse': gb_val_rmse,
            'val_r2': gb_val_r2,
            'feature_importance': gb_importance
        }

    # 4. Neural Network (simplified from previous implementation)
    with mlflow.start_run(nested=True, run_name="Neural_Network"):
        print("Training Neural Network model...")

        # Scale features for neural network
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_final)
        X_val_scaled = scaler.transform(X_val)

        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Build neural network model - FIX: Using tf.keras instead of the 'models' variable
        input_shape = X_train_scaled.shape[1]
        nn_model = tf.keras.Sequential([
            layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001), input_shape=(input_shape,)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
            layers.BatchNormalization(),
            layers.Dense(1)
        ])

        # Compile model
        nn_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        # Train model
        history = nn_model.fit(
            X_train_scaled, y_train_final,
            validation_data=(X_val_scaled, y_val),
            epochs=50,
            batch_size=64,
            callbacks=[early_stop],
            verbose=0
        )

        # Evaluate on validation data
        nn_val_pred = nn_model.predict(X_val_scaled).flatten()
        nn_val_mse = mean_squared_error(y_val, nn_val_pred)
        nn_val_rmse = np.sqrt(nn_val_mse)
        nn_val_r2 = r2_score(y_val, nn_val_pred)

        # Log metrics
        try:
            mlflow.log_metric("val_mse", nn_val_mse)
            mlflow.log_metric("val_rmse", nn_val_rmse)
            mlflow.log_metric("val_r2", nn_val_r2)
            mlflow.log_param("nn_epochs", len(history.history['loss']))
        except:
            pass

        print(f"Neural Network - Validation RMSE: {nn_val_rmse:.2f}, R²: {nn_val_r2:.2f}")

        # Store model and metrics
        model_results['neural_network'] = {
            'model': nn_model,
            'scaler': scaler,  # We need to keep the scaler for prediction
            'val_rmse': nn_val_rmse,
            'val_r2': nn_val_r2,
            'history': history.history
        }

    # Find best model based on validation R²
    best_model_name = max(model_results, key=lambda x: model_results[x]['val_r2'])
    best_model = model_results[best_model_name]

    print(f"\nBest model: {best_model_name} with validation R²: {best_model['val_r2']:.4f}")

    # Log best model
    try:
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_val_rmse", best_model['val_rmse'])
        mlflow.log_metric("best_val_r2", best_model['val_r2'])
    except:
        pass

    # Return trained models and training info
    return {
        "models": model_results,
        "best_model_name": best_model_name,
        "features": features
    }