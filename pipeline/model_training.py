import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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

    # Prepare validation split (keep this for neural network)
    val_size = int(0.2 * len(X_train))
    X_val = X_train.iloc[-val_size:].copy()
    y_val = y_train[-val_size:]
    X_train_final = X_train.iloc[:-val_size].copy()
    y_train_final = y_train[:-val_size]

    print(f"Training data: {X_train_final.shape[0]} samples, {X_train_final.shape[1]} features")
    print(f"Validation data: {X_val.shape[0]} samples")

    # Set up 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Log feature list
    try:
        mlflow.log_param("feature_count", len(features))
        top_features = min(10, len(features))
        mlflow.log_param("top_features", ", ".join(features[:top_features]))
    except:
        pass

    # Dictionary to store all models and their performance
    model_results = {}

    # 1. Decision Tree with cross-validation
    with mlflow.start_run(nested=True, run_name="Decision_Tree"):
        print("Training Decision Tree model...")
        dt_model = DecisionTreeRegressor(max_depth=8, min_samples_split=10, random_state=42)

        # Get cross-validation scores
        dt_cv_scores = cross_val_score(dt_model, X_train, y_train, cv=kf, scoring='r2')
        dt_cv_mean = dt_cv_scores.mean()
        dt_cv_std = dt_cv_scores.std()

        # Train on full dataset
        dt_model.fit(X_train, y_train)

        # Get feature importance
        dt_importance = pd.DataFrame({
            'feature': features,
            'importance': dt_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"Decision Tree - CV R²: {dt_cv_mean:.4f} ± {dt_cv_std:.4f}")
        try:
            mlflow.log_metric("cv_r2", dt_cv_mean)
        except:
            pass

        # Store model and metrics
        model_results['decision_tree'] = {
            'model': dt_model,
            'cv_r2': dt_cv_mean,
            'cv_r2_std': dt_cv_std,
            'feature_importance': dt_importance
        }

    # 2. Random Forest with cross-validation
    with mlflow.start_run(nested=True, run_name="Random_Forest"):
        print("Training Random Forest model...")
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5,
                                         random_state=42, n_jobs=-1)

        # Get cross-validation scores
        rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='r2')
        rf_cv_mean = rf_cv_scores.mean()
        rf_cv_std = rf_cv_scores.std()

        # Train on full dataset
        rf_model.fit(X_train, y_train)

        # Get feature importance
        rf_importance = pd.DataFrame({
            'feature': features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"Random Forest - CV R²: {rf_cv_mean:.4f} ± {rf_cv_std:.4f}")
        try:
            mlflow.log_metric("cv_r2", rf_cv_mean)
        except:
            pass

        # Store model and metrics
        model_results['random_forest'] = {
            'model': rf_model,
            'cv_r2': rf_cv_mean,
            'cv_r2_std': rf_cv_std,
            'feature_importance': rf_importance
        }

    # 3. Gradient Boosting with cross-validation
    with mlflow.start_run(nested=True, run_name="Gradient_Boosting"):
        print("Training Gradient Boosting model...")
        gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                             random_state=42)

        # Get cross-validation scores
        gb_cv_scores = cross_val_score(gb_model, X_train, y_train, cv=kf, scoring='r2')
        gb_cv_mean = gb_cv_scores.mean()
        gb_cv_std = gb_cv_scores.std()

        # Train on full dataset
        gb_model.fit(X_train, y_train)

        # Get feature importance
        gb_importance = pd.DataFrame({
            'feature': features,
            'importance': gb_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"Gradient Boosting - CV R²: {gb_cv_mean:.4f} ± {gb_cv_std:.4f}")
        try:
            mlflow.log_metric("cv_r2", gb_cv_mean)
        except:
            pass

        # Store model and metrics
        model_results['gradient_boosting'] = {
            'model': gb_model,
            'cv_r2': gb_cv_mean,
            'cv_r2_std': gb_cv_std,
            'feature_importance': gb_importance
        }

    # 4. Neural Network (unchanged)
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

        # Build neural network model
        input_shape = X_train_scaled.shape[1]
        nn_model = tf.keras.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
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

        print(f"Neural Network - Validation RMSE: {nn_val_rmse:.2f}, R²: {nn_val_r2:.2f}")

        # Store model and metrics
        model_results['neural_network'] = {
            'model': nn_model,
            'scaler': scaler,
            'val_rmse': nn_val_rmse,
            'val_r2': nn_val_r2,
            'history': history.history
        }

    # Find best model based on cross-validation R² for tree models, validation R² for neural network
    tree_models = ['decision_tree', 'random_forest', 'gradient_boosting']
    best_tree_model = max(tree_models, key=lambda x: model_results[x]['cv_r2'])

    # Compare best tree model with neural network
    if model_results[best_tree_model]['cv_r2'] > model_results['neural_network']['val_r2']:
        best_model_name = best_tree_model
    else:
        best_model_name = 'neural_network'

    best_model = model_results[best_model_name]

    print(f"\nBest model: {best_model_name}")
    if best_model_name in tree_models:
        print(f"CV R²: {best_model['cv_r2']:.4f} ± {best_model['cv_r2_std']:.4f}")
    else:
        print(f"Validation R²: {best_model['val_r2']:.4f}")

    # Return trained models and training info
    return {
        "models": model_results,
        "best_model_name": best_model_name,
        "features": features
    }