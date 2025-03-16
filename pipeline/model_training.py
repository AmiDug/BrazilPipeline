import os
import tempfile
from pathlib import Path

# Set environment variables before any other imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Set matplotlib to use a non-GUI backend BEFORE any other imports
import matplotlib

matplotlib.use('Agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
import mlflow

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def create_error_by_price_range_plot(y_test, y_pred, save_path=None):
    """Create a bar chart showing prediction error by price range."""
    # Convert to numpy arrays if they're not already
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    # Calculate absolute percentage error
    ape = np.abs((y_test - y_pred) / y_test) * 100

    # Create price range bins
    bins = [0, 50, 100, 200, 500, 1000, float('inf')]
    bin_labels = ['0-50', '50-100', '100-200', '200-500', '500-1000', '1000+']

    # Assign each sample to a bin
    bin_indices = np.digitize(y_test, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_labels) - 1)

    # Calculate mean APE for each bin
    mean_ape_by_bin = []
    counts_by_bin = []

    for i in range(len(bin_labels)):
        mask = (bin_indices == i)
        bin_ape = ape[mask]
        if len(bin_ape) > 0:
            mean_ape_by_bin.append(np.mean(bin_ape))
            counts_by_bin.append(np.sum(mask))
        else:
            mean_ape_by_bin.append(0)
            counts_by_bin.append(0)

    # Create plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(bin_labels, mean_ape_by_bin)

    # Add count annotations
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width() / 2, 5,
                 f'n={counts_by_bin[i]}',
                 ha='center', va='bottom')

    plt.title('Error by Price Range')
    plt.xlabel('Price Range')
    plt.ylabel('Mean Absolute Percentage Error (%)')
    plt.grid(axis='y', alpha=0.3)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.tight_layout()
        plt.show()
        plt.close()


def create_error_distribution_plot(y_test, y_pred, save_path=None):
    """Create a histogram of prediction errors."""
    # Calculate errors
    errors = y_pred - y_test

    plt.figure(figsize=(12, 6))
    sns.histplot(errors, kde=True, bins=30)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Prediction Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.tight_layout()
        plt.show()
        plt.close()


def create_feature_importance_plot(feature_importance_df, save_path=None):
    """Create a horizontal bar chart of feature importances."""
    # Sort by importance and take top 15
    df = feature_importance_df.sort_values('importance', ascending=False).head(15)

    plt.figure(figsize=(12, 8))
    plt.barh(df['feature'], df['importance'])
    plt.title('Top 15 Feature Importances')
    plt.xlabel('Importance')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        plt.close()


def create_predicted_vs_actual_plot(y_test, y_pred, model_name, metrics, save_path=None):
    """Create a scatter plot of predicted vs actual values."""
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, s=5)

    # Add diagonal line (perfect predictions)
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.plot([0, max_val], [0, max_val], 'r--')

    # Add metrics to plot
    metrics_text = f"RMSE: {metrics['test_rmse']:.2f}\nR²: {metrics['test_r2']:.4f}\nMAPE: {metrics['test_mape']:.2f}%"
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    plt.title(f'{model_name} - Predicted vs Actual')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.grid(alpha=0.3)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.tight_layout()
        plt.show()
        plt.close()


def create_customer_states_plot(df, save_path=None):
    """Create a bar chart of top 10 customer states."""
    # Count states
    state_counts = df['customer_state'].value_counts().head(10)

    plt.figure(figsize=(12, 6))
    plt.bar(state_counts.index, state_counts.values)
    plt.title('Top 10 Customer States')
    plt.xlabel('customer_state')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.3)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.tight_layout()
        plt.show()
        plt.close()


def model_training(data_splits, target_column='price'):
    """
    Train multiple models on the prepared data with comprehensive metrics reporting.

    Models included:
    - Decision Tree
    - Random Forest
    - XGBoost with GPU acceleration
    - Neural Network

    All models report R², RMSE, MAE, MSE, and MAPE metrics.
    """
    print("Starting multi-model training with cross-validation...")

    # Extract training data
    train_df = data_splits['train']
    features = data_splits.get('features', [col for col in train_df.columns
                                            if col != target_column and col != 'product_id'])

    # Prepare training data
    X_train = train_df[features].copy()
    y_train = train_df[target_column].values

    # Extract test data
    test_df = data_splits['test']
    X_test = test_df[features].copy()
    y_test = test_df[target_column].values

    print(f"Training data: {X_train.shape[0]} samples, {len(features)} features")

    # Create a temporary directory for visualizations
    artifacts_dir = tempfile.mkdtemp()

    # Dict to store model results
    model_results = {}

    # Set up cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # ----------------------------------------------------
    # 1. Decision Tree
    # ----------------------------------------------------
    print("\nTraining Decision Tree model...")
    dt_model = DecisionTreeRegressor(max_depth=8, min_samples_split=10, random_state=42)
    dt_cv_scores = cross_val_score(dt_model, X_train, y_train, cv=kf, scoring='r2')
    dt_model.fit(X_train, y_train)  # Train on full dataset

    # Get feature importance
    dt_importance = pd.DataFrame({
        'feature': features,
        'importance': dt_model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Test set performance
    dt_pred = dt_model.predict(X_test)
    dt_test_r2 = r2_score(y_test, dt_pred)
    dt_test_mse = mean_squared_error(y_test, dt_pred)
    dt_test_rmse = np.sqrt(dt_test_mse)
    dt_test_mae = mean_absolute_error(y_test, dt_pred)
    dt_test_mape = np.mean(np.abs((y_test - dt_pred) / y_test)) * 100

    print(
        f"Decision Tree - Test R²: {dt_test_r2:.4f}, RMSE: {dt_test_rmse:.2f}, MAE: {dt_test_mae:.2f}, MSE: {dt_test_mse:.2f}, MAPE: {dt_test_mape:.2f}%")

    model_results['decision_tree'] = {
        'model': dt_model,
        'cv_r2': dt_cv_scores.mean(),
        'cv_r2_std': dt_cv_scores.std(),
        'test_r2': dt_test_r2,
        'test_rmse': dt_test_rmse,
        'test_mae': dt_test_mae,
        'test_mse': dt_test_mse,
        'test_mape': dt_test_mape,
        'feature_importance': dt_importance,
        'predictions': dt_pred
    }

    # Create visualizations - just create and save, don't log (autologging will handle)
    dt_pred_vs_actual_path = os.path.join(artifacts_dir, 'dt_predicted_vs_actual.png')
    create_predicted_vs_actual_plot(y_test, dt_pred, 'Decision Tree',
                                    model_results['decision_tree'], dt_pred_vs_actual_path)

    dt_feature_imp_path = os.path.join(artifacts_dir, 'dt_feature_importance.png')
    create_feature_importance_plot(dt_importance, dt_feature_imp_path)

    # Only log visualizations (not handled by autologging)
    mlflow.log_artifact(dt_pred_vs_actual_path, "visualizations")
    mlflow.log_artifact(dt_feature_imp_path, "visualizations")

    # ----------------------------------------------------
    # 2. Random Forest
    # ----------------------------------------------------
    print("\nTraining Random Forest model...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='r2')
    rf_model.fit(X_train, y_train)  # Train on full dataset

    # Get feature importance
    rf_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Test set performance
    rf_pred = rf_model.predict(X_test)
    rf_test_r2 = r2_score(y_test, rf_pred)
    rf_test_mse = mean_squared_error(y_test, rf_pred)
    rf_test_rmse = np.sqrt(rf_test_mse)
    rf_test_mae = mean_absolute_error(y_test, rf_pred)
    rf_test_mape = np.mean(np.abs((y_test - rf_pred) / y_test)) * 100

    print(
        f"Random Forest - Test R²: {rf_test_r2:.4f}, RMSE: {rf_test_rmse:.2f}, MAE: {rf_test_mae:.2f}, MSE: {rf_test_mse:.2f}, MAPE: {rf_test_mape:.2f}%")

    model_results['random_forest'] = {
        'model': rf_model,
        'cv_r2': rf_cv_scores.mean(),
        'cv_r2_std': rf_cv_scores.std(),
        'test_r2': rf_test_r2,
        'test_rmse': rf_test_rmse,
        'test_mae': rf_test_mae,
        'test_mse': rf_test_mse,
        'test_mape': rf_test_mape,
        'feature_importance': rf_importance,
        'predictions': rf_pred
    }

    # Create and log visualizations
    rf_pred_vs_actual_path = os.path.join(artifacts_dir, 'rf_predicted_vs_actual.png')
    create_predicted_vs_actual_plot(y_test, rf_pred, 'Random Forest',
                                    model_results['random_forest'], rf_pred_vs_actual_path)

    rf_feature_imp_path = os.path.join(artifacts_dir, 'rf_feature_importance.png')
    create_feature_importance_plot(rf_importance, rf_feature_imp_path)

    # Only log visualizations
    mlflow.log_artifact(rf_pred_vs_actual_path, "visualizations")
    mlflow.log_artifact(rf_feature_imp_path, "visualizations")

    # ----------------------------------------------------
    # 3. XGBoost with GPU acceleration
    # ----------------------------------------------------
    print("\nTraining XGBoost model with GPU acceleration...")

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',  # Fast histogram algorithm
        'device': 'cuda',  # Use GPU acceleration
        'max_depth': 11,
        'learning_rate': 0.1,
        'min_child_weight': 5,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'seed': 42  # Set seed for reproducibility
    }

    # XGBoost built-in cross-validation
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=200,
        nfold=5,
        stratified=False,
        metrics={'rmse'},
        early_stopping_rounds=20,
        seed=42,
        as_pandas=True
    )

    # Get the optimal number of boosting rounds
    best_round = cv_results.shape[0]
    xgb_cv_rmse = cv_results.iloc[-1]['test-rmse-mean']

    # Train final model with optimal number of rounds
    xgb_model = xgb.train(params, dtrain, num_boost_round=best_round)

    # Make predictions
    xgb_y_pred = xgb_model.predict(dtest)

    # Calculate all metrics for XGBoost
    xgb_test_r2 = r2_score(y_test, xgb_y_pred)
    xgb_test_mse = mean_squared_error(y_test, xgb_y_pred)
    xgb_test_rmse = np.sqrt(xgb_test_mse)
    xgb_test_mae = mean_absolute_error(y_test, xgb_y_pred)
    xgb_test_mape = np.mean(np.abs((y_test - xgb_y_pred) / y_test)) * 100

    # Print metrics
    print(
        f"XGBoost - Test R²: {xgb_test_r2:.4f}, RMSE: {xgb_test_rmse:.2f}, MAE: {xgb_test_mae:.2f}, MSE: {xgb_test_mse:.2f}, MAPE: {xgb_test_mape:.2f}%")
    print(f"XGBoost - Best boosting rounds: {best_round}")

    # Get feature importance
    importance_scores = xgb_model.get_score(importance_type='gain')
    xgb_importance = pd.DataFrame({
        'feature': list(importance_scores.keys()),
        'importance': list(importance_scores.values())
    }).sort_values('importance', ascending=False)

    # Calculate CV R² (not directly provided by xgb.cv)
    xgb_cv_r2 = 1 - (xgb_cv_rmse ** 2 / np.var(y_train))

    # Store results with all metrics
    model_results['xgboost_gpu'] = {
        'model': xgb_model,
        'cv_rmse': xgb_cv_rmse,
        'cv_r2': xgb_cv_r2,
        'test_r2': xgb_test_r2,
        'test_rmse': xgb_test_rmse,
        'test_mae': xgb_test_mae,
        'test_mse': xgb_test_mse,
        'test_mape': xgb_test_mape,
        'best_round': best_round,
        'feature_importance': xgb_importance,
        'predictions': xgb_y_pred
    }

    # Create and log XGBoost visualizations
    xgb_pred_vs_actual_path = os.path.join(artifacts_dir, 'xgb_predicted_vs_actual.png')
    create_predicted_vs_actual_plot(y_test, xgb_y_pred, 'Gradient Boosting',
                                    model_results['xgboost_gpu'], xgb_pred_vs_actual_path)

    xgb_feature_imp_path = os.path.join(artifacts_dir, 'xgb_feature_importance.png')
    create_feature_importance_plot(xgb_importance, xgb_feature_imp_path)

    xgb_error_dist_path = os.path.join(artifacts_dir, 'xgb_error_distribution.png')
    create_error_distribution_plot(y_test, xgb_y_pred, xgb_error_dist_path)

    xgb_error_by_price_path = os.path.join(artifacts_dir, 'xgb_error_by_price_range.png')
    create_error_by_price_range_plot(y_test, xgb_y_pred, xgb_error_by_price_path)

    # Only log custom visualizations
    mlflow.log_artifact(xgb_pred_vs_actual_path, "visualizations")
    mlflow.log_artifact(xgb_feature_imp_path, "visualizations")
    mlflow.log_artifact(xgb_error_dist_path, "visualizations")
    mlflow.log_artifact(xgb_error_by_price_path, "visualizations")

    # Create custom predict function for XGBoost
    def xgb_predict_fn(X_df):
        """Prediction function that handles DataFrame input for XGBoost"""
        if isinstance(X_df, xgb.DMatrix):
            X_dmatrix = X_df
        else:
            X_dmatrix = xgb.DMatrix(data=X_df)
        return xgb_model.predict(X_dmatrix)

    model_results['xgboost_gpu']['predict_fn'] = xgb_predict_fn

    # ----------------------------------------------------
    # 4. Neural Network
    # ----------------------------------------------------
    print("\nTraining Neural Network model...")

    # Set seeds again for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Manual validation split exactly as in original code
    val_size = int(0.2 * len(X_train))
    X_val = X_train.iloc[-val_size:].copy()
    y_val = y_train[-val_size:]
    X_train_final = X_train.iloc[:-val_size].copy()
    y_train_final = y_train[:-val_size]

    # Scale data - make sure to fit only on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

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

    # Compile model
    nn_model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Train with minimal verbosity
    history = nn_model.fit(
        X_train_scaled, y_train_final,
        validation_data=(X_val_scaled, y_val),
        epochs=50,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0
    )

    # Evaluate on validation set (for comparison with original code)
    nn_val_pred = nn_model.predict(X_val_scaled, verbose=0).flatten()
    nn_val_mse = mean_squared_error(y_val, nn_val_pred)
    nn_val_rmse = np.sqrt(nn_val_mse)
    nn_val_r2 = r2_score(y_val, nn_val_pred)
    nn_val_mae = mean_absolute_error(y_val, nn_val_pred)
    nn_val_mape = np.mean(np.abs((y_val - nn_val_pred) / y_val)) * 100

    print(
        f"Neural Network - Validation R²: {nn_val_r2:.4f}, RMSE: {nn_val_rmse:.2f}, MAE: {nn_val_mae:.2f}, MSE: {nn_val_mse:.2f}, MAPE: {nn_val_mape:.2f}%")

    # Also evaluate on test set for comparison with other models
    nn_test_pred = nn_model.predict(X_test_scaled, verbose=0).flatten()
    nn_test_r2 = r2_score(y_test, nn_test_pred)
    nn_test_mse = mean_squared_error(y_test, nn_test_pred)
    nn_test_rmse = np.sqrt(nn_test_mse)
    nn_test_mae = mean_absolute_error(y_test, nn_test_pred)
    nn_test_mape = np.mean(np.abs((y_test - nn_test_pred) / y_test)) * 100

    print(
        f"Neural Network - Test R²: {nn_test_r2:.4f}, RMSE: {nn_test_rmse:.2f}, MAE: {nn_test_mae:.2f}, MSE: {nn_test_mse:.2f}, MAPE: {nn_test_mape:.2f}%")

    model_results['neural_network'] = {
        'model': nn_model,
        'scaler': scaler,
        'val_rmse': nn_val_rmse,
        'val_r2': nn_val_r2,
        'val_mae': nn_val_mae,
        'val_mse': nn_val_mse,
        'val_mape': nn_val_mape,
        'test_r2': nn_test_r2,
        'test_rmse': nn_test_rmse,
        'test_mae': nn_test_mae,
        'test_mse': nn_test_mse,
        'test_mape': nn_test_mape,
        'history': history.history,
        'predictions': nn_test_pred
    }

    # Create and log NN visualizations not handled by autologging
    nn_pred_vs_actual_path = os.path.join(artifacts_dir, 'nn_predicted_vs_actual.png')
    create_predicted_vs_actual_plot(y_test, nn_test_pred, 'Neural Network',
                                    model_results['neural_network'], nn_pred_vs_actual_path)

    # Training history plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Neural Network Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    nn_history_path = os.path.join(artifacts_dir, 'nn_training_history.png')
    plt.savefig(nn_history_path, bbox_inches='tight')
    plt.close()

    # Log visualizations
    mlflow.log_artifact(nn_pred_vs_actual_path, "visualizations")
    mlflow.log_artifact(nn_history_path, "visualizations")

    # ----------------------------------------------------
    # Model Comparison and Selection
    # ----------------------------------------------------

    # Define all models for comparison and logging
    all_models = ['decision_tree', 'random_forest', 'xgboost_gpu', 'neural_network']

    # Find best model based on cross-validation and validation scores
    tree_models = ['decision_tree', 'random_forest', 'xgboost_gpu']
    best_tree_model = max(tree_models, key=lambda x: model_results[x].get('test_r2', 0))

    # Compare best tree model with neural network using validation R² (like original)
    if model_results[best_tree_model]['test_r2'] > model_results['neural_network']['test_r2']:
        best_model_name = best_tree_model
    else:
        best_model_name = 'neural_network'

    print(f"\nBest model: {best_model_name}")

    # Create a summary DataFrame for model comparison
    summary_data = []
    for model_name in all_models:
        model_data = model_results[model_name]
        summary_data.append({
            'Model': model_name,
            'Test R²': model_data.get('test_r2', 0),
            'Test RMSE': model_data.get('test_rmse', 0),
            'Test MAE': model_data.get('test_mae', 0),
            'Test MAPE (%)': model_data.get('test_mape', 0)
        })

    summary_df = pd.DataFrame(summary_data).sort_values('Test R²', ascending=False)
    print("\nModel Performance Comparison:")
    print(summary_df)

    # Print top features from best model
    if 'feature_importance' in model_results[best_model_name]:
        print(f"\nTop 10 Important Features ({best_model_name}):")
        print(model_results[best_model_name]['feature_importance'].head(10))

    # Create and log customer states visualization
    if 'customer_state' in train_df.columns:
        customer_states_path = os.path.join(artifacts_dir, 'top_customer_states.png')
        create_customer_states_plot(train_df, customer_states_path)
        mlflow.log_artifact(customer_states_path, "visualizations")

    # Create summary model comparison visualization
    plt.figure(figsize=(12, 6))
    models = summary_df['Model']
    r2_values = summary_df['Test R²']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    plt.bar(models, r2_values, color=colors)
    plt.title('Model R² Comparison')
    plt.xlabel('Model')
    plt.ylabel('Test R²')
    plt.ylim(0, 1)

    for i, v in enumerate(r2_values):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')

    summary_path = os.path.join(artifacts_dir, 'model_comparison.png')
    plt.savefig(summary_path, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(summary_path, "visualizations")

    # Log summary table as CSV
    summary_csv_path = os.path.join(artifacts_dir, 'model_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    mlflow.log_artifact(summary_csv_path, "artifacts")

    # Log best model name parameter
    mlflow.log_param("best_model", best_model_name)

    return {
        "models": model_results,
        "best_model_name": best_model_name,
        "features": features,
        "feature_importance": model_results.get(best_model_name, {}).get('feature_importance'),
        "summary": summary_df,
        "predict_fn": model_results[best_model_name].get('predict_fn')
    }