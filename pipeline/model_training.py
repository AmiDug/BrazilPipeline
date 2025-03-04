import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import mlflow


def model_training(data_splits, target_column='price', log_transform=False):
    """Train models with enhanced neural network architecture"""
    print("Starting model training...")

    # Extract data
    train_df = data_splits['train']

    # Decide whether to use log-transformed target
    if log_transform and 'log_price' in train_df.columns:
        target = 'log_price'
        print(f"Using log-transformed price as target")
    else:
        target = target_column
        print(f"Using original price as target")

    # Get selected features
    if 'selected_features' in data_splits:
        features = data_splits['selected_features']
    else:
        # Fallback to basic features
        features = [col for col in train_df.columns
                    if col not in [target_column, 'log_price', 'id']
                    and train_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]

    print(f"Using {len(features)} features")

    # Select X and y
    X_train = train_df[features].copy()
    y_train = train_df[target].values

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Create validation split
    val_size = int(0.2 * len(X_train))
    X_val = X_train_scaled[-val_size:]
    y_val = y_train[-val_size:]
    X_train_final = X_train_scaled[:-val_size]
    y_train_final = y_train[:-val_size]

    # ENHANCEMENT: Improved neural network architecture
    input_dim = X_train_final.shape[1]

    # Build model with improved architecture from notebooks
    model = Sequential()

    # Input layer
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Hidden layers
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    # Output layer
    model.add(Dense(1, activation='linear'))

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )

    # Callbacks for better training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    ]

    # Train model
    history = model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on validation data
    nn_val_pred = model.predict(X_val).flatten()
    nn_val_mse = mean_squared_error(y_val, nn_val_pred)
    nn_val_rmse = np.sqrt(nn_val_mse)
    nn_val_r2 = r2_score(y_val, nn_val_pred)

    # Log metrics
    mlflow.log_metric("val_rmse", nn_val_rmse)
    mlflow.log_metric("val_r2", nn_val_r2)

    print(f"Neural Network - Validation RMSE: {nn_val_rmse:.4f}, R²: {nn_val_r2:.4f}")

    # Return model results
    return {
        "neural_network": {
            "model": model,
            "history": history.history,
            "rmse": nn_val_rmse,
            "r2": nn_val_r2
        },
        "best_model": "neural_network",
        "preprocessing": {
            "scaler": scaler,
            "features": features,
            "log_transform": log_transform
        }
    }