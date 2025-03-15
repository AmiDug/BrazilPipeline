import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set matplotlib to use a non-GUI backend BEFORE any other imports
import matplotlib

matplotlib.use('Agg')  # Force non-interactive backend

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
import mlflow


def model_training(data_splits, target_column='price'):
    """Train multiple models on the prepared data using cross-validation"""
    print("Starting model training...")

    # Extract training data
    train_df = data_splits['train']
    features = data_splits.get('features', [col for col in train_df.columns
                                            if col != 'price' and col != 'product_id'])

    # Prepare training data
    X_train = train_df[features].copy()
    y_train = train_df[target_column].values

    print(f"Training data: {X_train.shape[0]} samples, {len(features)} features")

    # Set up cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Dict to store models and performance
    model_results = {}

    # 1. Decision Tree with cross-validation
    print("Training Decision Tree model...")
    dt_model = DecisionTreeRegressor(max_depth=8, min_samples_split=10, random_state=42)
    dt_cv_scores = cross_val_score(dt_model, X_train, y_train, cv=kf, scoring='r2')
    dt_model.fit(X_train, y_train)  # Train on full dataset

    # Get feature importance
    dt_importance = pd.DataFrame({
        'feature': features,
        'importance': dt_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"Decision Tree - CV R²: {dt_cv_scores.mean():.4f} ± {dt_cv_scores.std():.4f}")

    model_results['decision_tree'] = {
        'model': dt_model,
        'cv_r2': dt_cv_scores.mean(),
        'cv_r2_std': dt_cv_scores.std(),
        'feature_importance': dt_importance
    }

    # 2. Random Forest with cross-validation
    print("Training Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                     min_samples_split=5, random_state=42, n_jobs=-1)
    rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='r2')
    rf_model.fit(X_train, y_train)  # Train on full dataset

    # Get feature importance
    rf_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"Random Forest - CV R²: {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")

    model_results['random_forest'] = {
        'model': rf_model,
        'cv_r2': rf_cv_scores.mean(),
        'cv_r2_std': rf_cv_scores.std(),
        'feature_importance': rf_importance
    }

    # 3. Gradient Boosting with cross-validation
    print("Training Gradient Boosting model...")
    gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                         learning_rate=0.1, random_state=42)
    gb_cv_scores = cross_val_score(gb_model, X_train, y_train, cv=kf, scoring='r2')
    gb_model.fit(X_train, y_train)  # Train on full dataset

    # Get feature importance
    gb_importance = pd.DataFrame({
        'feature': features,
        'importance': gb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"Gradient Boosting - CV R²: {gb_cv_scores.mean():.4f} ± {gb_cv_scores.std():.4f}")

    model_results['gradient_boosting'] = {
        'model': gb_model,
        'cv_r2': gb_cv_scores.mean(),
        'cv_r2_std': gb_cv_scores.std(),
        'feature_importance': gb_importance
    }

    # 4. Neural Network - prepare validation split here instead of earlier
    print("Training Neural Network model...")
    val_size = int(0.2 * len(X_train))
    X_val = X_train.iloc[-val_size:].copy()
    y_val = y_train[-val_size:]
    X_train_final = X_train.iloc[:-val_size].copy()
    y_train_final = y_train[:-val_size]

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_val_scaled = scaler.transform(X_val)

    # Build model
    nn_model = tf.keras.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Dense(1)
    ])

    # Compile and train
    nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train with minimal verbosity
    history = nn_model.fit(
        X_train_scaled, y_train_final,
        validation_data=(X_val_scaled, y_val),
        epochs=50, batch_size=64, callbacks=[early_stop], verbose=0
    )

    # Evaluate
    nn_val_pred = nn_model.predict(X_val_scaled).flatten()
    nn_val_rmse = np.sqrt(mean_squared_error(y_val, nn_val_pred))
    nn_val_r2 = r2_score(y_val, nn_val_pred)

    print(f"Neural Network - Validation R²: {nn_val_r2:.4f}")

    model_results['neural_network'] = {
        'model': nn_model,
        'scaler': scaler,
        'val_rmse': nn_val_rmse,
        'val_r2': nn_val_r2,
        'history': history.history
    }

    # Find best model
    tree_models = ['decision_tree', 'random_forest', 'gradient_boosting']
    best_tree_model = max(tree_models, key=lambda x: model_results[x]['cv_r2'])

    # Compare best tree model with neural network
    if model_results[best_tree_model]['cv_r2'] > model_results['neural_network']['val_r2']:
        best_model_name = best_tree_model
    else:
        best_model_name = 'neural_network'

    print(f"\nBest model: {best_model_name}")

    return {
        "models": model_results,
        "best_model_name": best_model_name,
        "features": features
    }