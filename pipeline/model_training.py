import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import mlflow
import os


def model_training(data_splits, target_column='price', epochs=10, batch_size=32):

    print("Starting model training...")

    # Extract data
    train_df = data_splits['train']
    test_df = data_splits['test']

    # Split features and target
    X_train = train_df.drop(columns=[target_column, 'id', 'image', 'title', 'description'])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column, 'id', 'image', 'title', 'description'])
    y_test = test_df[target_column]

    # Convert categorical features to one-hot encoding
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)

    # Ensure test set has same columns as train set
    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[X_train.columns]

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Only log non-autologged parameters
    mlflow.log_param("num_features", X_train.shape[1])

    # ---------------------------- Model 1: Neural Network with Regularization ----------------------------
    # Early stopping for both models
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Create neural network with reasonable hyperparameters
    model = models.Sequential([
        layers.Dense(16, activation='relu', kernel_regularizer=l2(0.01),
                     input_shape=(X_train_scaled.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(8, activation='relu', kernel_regularizer=l2(0.01)),
        layers.Dropout(0.3),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the neural network
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop]
    )

    # Evaluate on test data
    test_mse, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)

    # Save learning curves to MLflow
    plt.figure(figsize=(12, 4))
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
    plt.ylabel('Loss')
    plt.legend()

    # Save figure for MLflow
    fig_path = "model1_learning_curves.png"
    plt.savefig(fig_path)
    mlflow.log_artifact(fig_path)
    os.remove(fig_path)
    plt.close()

    print(f"Neural Network - Test MAE: ${test_mae:.2f}")

    # ---------------------------- Model 2: Simple Linear Model ----------------------------
    # Create a simple linear model with L2 regularization
    simple_model = models.Sequential([
        layers.Dense(1, kernel_regularizer=l2(0.01), input_shape=(X_train_scaled.shape[1],))
    ])

    simple_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    simple_history = simple_model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop]
    )

    # Evaluate on test data
    simple_test_mse, simple_test_mae = simple_model.evaluate(X_test_scaled, y_test, verbose=0)

    # Save learning curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(simple_history.history['mae'], label='Train')
    plt.plot(simple_history.history['val_mae'], label='Validation')
    plt.title('Linear Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(simple_history.history['loss'], label='Train')
    plt.plot(simple_history.history['val_loss'], label='Validation')
    plt.title('Linear Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Save figure for MLflow
    fig_path = "model2_learning_curves.png"
    plt.savefig(fig_path)
    mlflow.log_artifact(fig_path)
    os.remove(fig_path)
    plt.close()

    print(f"Linear Model - Test MAE: ${simple_test_mae:.2f}")

    # ---------------------------- Model Comparison ----------------------------
    # Create comparison plot
    plt.figure(figsize=(8, 5))
    models_names = ['Neural Network', 'Linear Model']
    mae_values = [test_mae, simple_test_mae]

    plt.bar(models_names, mae_values)
    plt.title('Model Comparison - Test MAE (lower is better)')
    plt.ylabel('Mean Absolute Error ($)')

    # Add text labels on bars
    for i, mae in enumerate(mae_values):
        plt.text(i, mae + 0.5, f'${mae:.2f}', ha='center')

    # Save comparison figure
    fig_path = "model_comparison.png"
    plt.savefig(fig_path)
    mlflow.log_artifact(fig_path)
    os.remove(fig_path)
    plt.close()

    # Determine best model
    best_model = "neural_network" if test_mae < simple_test_mae else "linear_model"

    # Only log non-autologged parameters
    mlflow.log_param("best_model", best_model)

    return {
        "neural_network": {
            "model": model,
            "history": history.history,
            "mae": test_mae
        },
        "linear_model": {
            "model": simple_model,
            "history": simple_history.history,
            "mae": simple_test_mae
        },
        "best_model": best_model,
        "preprocessing": {
            "scaler": scaler
        }
    }