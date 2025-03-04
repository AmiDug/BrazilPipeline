import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import mlflow
import os


def model_training(data_splits, target_column='price', epochs=15, batch_size=32):
    print("Starting model training...")

    # Extract data
    train_df = data_splits['train']

    # Feature engineering - separate numeric and categorical features
    # Keep only relevant features for pricing model
    selected_numeric_features = [
        'product_weight_g', 'volume_cm3', 'freight_value',
        'description_length', 'image_count', 'density', 'price_per_gram',
        'count'
    ]

    # Try to use category_code if it exists, otherwise ignore
    if 'category_code' in train_df.columns:
        selected_numeric_features.append('category_code')

    # Try to use category_avg_price if it exists, otherwise ignore
    if 'category_avg_price' in train_df.columns:
        selected_numeric_features.append('category_avg_price')

    # Try to use price_ratio if it exists, otherwise ignore
    if 'price_ratio' in train_df.columns:
        selected_numeric_features.append('price_ratio')

    # Keep only numeric features for simplicity
    valid_numeric_features = [col for col in selected_numeric_features if col in train_df.columns]

    print(f"Using features: {valid_numeric_features}")

    # Select X and y
    X_train = train_df[valid_numeric_features].copy()
    y_train = train_df[target_column].values

    # Log feature selection
    mlflow.log_param("selected_features", ", ".join(valid_numeric_features))

    # Scale features for better model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Create validation split from training data
    val_size = int(0.2 * len(X_train))
    X_val = X_train_scaled[-val_size:]
    y_val = y_train[-val_size:]
    X_train_final = X_train_scaled[:-val_size]
    y_train_final = y_train[:-val_size]

    # Create visualization of price distribution
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(y_train_final, bins=30, alpha=0.7)
    plt.title('Training Set Price Distribution')
    plt.xlabel('Price (R$)')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    plt.hist(y_val, bins=30, alpha=0.7)
    plt.title('Validation Set Price Distribution')
    plt.xlabel('Price (R$)')

    # Save and log figure
    price_dist_path = "price_distribution_train_val.png"
    plt.savefig(price_dist_path)
    mlflow.log_artifact(price_dist_path)
    os.remove(price_dist_path)
    plt.close()

    # Log input shape for models
    input_shape = X_train_final.shape[1]
    mlflow.log_param("input_feature_count", input_shape)
    mlflow.log_param("model_train_size", len(X_train_final))
    mlflow.log_param("model_val_size", len(X_val))

    # ---- Model 1: Neural Network with Regularization ----
    # Track separately using nested runs
    with mlflow.start_run(nested=True, run_name="Neural_Network"):
        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # Learning rate reduction callback
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-5
        )

        # Create neural network with regularization
        # Using Functional API to avoid the input_shape warning
        inputs = tf.keras.Input(shape=(input_shape,))
        x = layers.Dense(16, activation='relu', kernel_regularizer=l2(0.01))(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(8, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Compile model
        model.compile(
            optimizer='adam',
            loss='mse',  # Mean Squared Error for regression
            metrics=['mae']  # Mean Absolute Error
        )

        # Train model
        history = model.fit(
            X_train_final, y_train_final,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        # Evaluate on validation data
        val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)

        # Create and save learning curves
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['mae'], label='Train')
        plt.plot(history.history['val_mae'], label='Validation')
        plt.title('Neural Network MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Neural Network Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()

        plt.tight_layout()

        # Save and log figure
        nn_curves_path = "nn_learning_curves.png"
        plt.savefig(nn_curves_path)
        mlflow.log_artifact(nn_curves_path)
        os.remove(nn_curves_path)
        plt.close()

        # Make predictions on validation set
        val_pred = model.predict(X_val, verbose=0).flatten()

        # Create prediction vs actual plot
        plt.figure(figsize=(10, 8))
        plt.scatter(y_val, val_pred, alpha=0.5)

        # Add perfect prediction line
        min_val = min(min(y_val), min(val_pred))
        max_val = max(max(y_val), max(val_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        plt.title('Neural Network: Predicted vs Actual Prices')
        plt.xlabel('Actual Price (R$)')
        plt.ylabel('Predicted Price (R$)')
        plt.grid(True, alpha=0.3)

        # Add metrics annotation
        plt.annotate(f"MAE: {val_mae:.2f}\nMSE: {val_loss:.2f}",
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        # Save and log figure
        nn_pred_path = "nn_predictions.png"
        plt.savefig(nn_pred_path)
        mlflow.log_artifact(nn_pred_path)
        os.remove(nn_pred_path)
        plt.close()

        print(f"Neural Network - Validation MAE: {val_mae:.2f}")

    # ---- Model 2: Linear Model ----
    with mlflow.start_run(nested=True, run_name="Linear_Model"):
        # Create simple linear model
        # Using Functional API to avoid the input_shape warning
        inputs = tf.keras.Input(shape=(input_shape,))
        outputs = layers.Dense(1, kernel_regularizer=l2(0.01))(inputs)
        linear_model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Compile model
        linear_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        # Add early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # Train model
        linear_history = linear_model.fit(
            X_train_final, y_train_final,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )

        # Evaluate on validation data
        linear_val_loss, linear_val_mae = linear_model.evaluate(X_val, y_val, verbose=0)

        # Create and save learning curves
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(linear_history.history['mae'], label='Train')
        plt.plot(linear_history.history['val_mae'], label='Validation')
        plt.title('Linear Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(linear_history.history['loss'], label='Train')
        plt.plot(linear_history.history['val_loss'], label='Validation')
        plt.title('Linear Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()

        plt.tight_layout()

        # Save and log figure
        linear_curves_path = "linear_learning_curves.png"
        plt.savefig(linear_curves_path)
        mlflow.log_artifact(linear_curves_path)
        os.remove(linear_curves_path)
        plt.close()

        # Make predictions on validation set
        linear_val_pred = linear_model.predict(X_val, verbose=0).flatten()

        # Create prediction vs actual plot
        plt.figure(figsize=(10, 8))
        plt.scatter(y_val, linear_val_pred, alpha=0.5)

        # Add perfect prediction line
        min_val = min(min(y_val), min(linear_val_pred))
        max_val = max(max(y_val), max(linear_val_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        plt.title('Linear Model: Predicted vs Actual Prices')
        plt.xlabel('Actual Price (R$)')
        plt.ylabel('Predicted Price (R$)')
        plt.grid(True, alpha=0.3)

        # Add metrics annotation
        plt.annotate(f"MAE: {linear_val_mae:.2f}\nMSE: {linear_val_loss:.2f}",
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        # Save and log figure
        linear_pred_path = "linear_predictions.png"
        plt.savefig(linear_pred_path)
        mlflow.log_artifact(linear_pred_path)
        os.remove(linear_pred_path)
        plt.close()

        print(f"Linear Model - Validation MAE: {linear_val_mae:.2f}")

    # ---- Model Comparison ----
    # Create comparison bar chart
    plt.figure(figsize=(10, 6))
    model_names = ['Neural Network', 'Linear Model']
    mae_values = [val_mae, linear_val_mae]

    plt.bar(model_names, mae_values)
    plt.title('Model Comparison - Validation MAE (lower is better)')
    plt.ylabel('Mean Absolute Error (R$)')

    # Add text labels on bars
    for i, mae in enumerate(mae_values):
        plt.text(i, mae + 0.5, f'R${mae:.2f}', ha='center')

    # Determine best model
    best_model_idx = np.argmin(mae_values)
    best_model_name = model_names[best_model_idx].lower().replace(' ', '_')
    plt.annotate(f'Best Model', xy=(best_model_idx, 0),
                 xytext=(best_model_idx, -5),
                 ha='center', va='top',
                 arrowprops=dict(arrowstyle='->'))

    # Save figure
    comparison_path = "model_comparison.png"
    plt.savefig(comparison_path)
    mlflow.log_artifact(comparison_path)
    os.remove(comparison_path)
    plt.close()

    # Log best model
    mlflow.log_param("best_model", best_model_name)
    mlflow.log_metric("best_model_mae", min(mae_values))

    print(f"Best model: {best_model_name} with MAE: {min(mae_values):.2f}")

    # Return models, history, and preprocessing objects
    return {
        "neural_network": {
            "model": model,
            "history": history.history,
            "mae": val_mae
        },
        "linear_model": {
            "model": linear_model,
            "history": linear_history.history,
            "mae": linear_val_mae
        },
        "best_model": best_model_name,
        "preprocessing": {
            "scaler": scaler,
            "features": valid_numeric_features
        }
    }