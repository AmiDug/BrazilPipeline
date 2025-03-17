# Set environment variables before any other imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU (3060 Ti)
os.environ['XGBOOST_MODEL_FORMAT'] = 'json'  # Force JSON format for XGBoost model saving

# Set matplotlib to use a non-GUI backend BEFORE any other imports
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
import tempfile
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
import mlflow

def create_plot(plot_function, args, kwargs, save_path=None):
    """Generic plotting function to reduce code duplication"""
    # Create plot using provided function
    plot_function(*args, **kwargs)

    # Save or display
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.tight_layout()
        plt.show()
        plt.close()
        return None


def create_error_by_price_range_plot(y_test, y_pred, save_path=None):
    """Create a bar chart showing prediction error by price range."""
    def plot_function(y_test, y_pred):
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

    return create_plot(plot_function, (y_test, y_pred), {}, save_path)


def create_feature_importance_plot(feature_importance_df, save_path=None):
    """Create a horizontal bar chart of feature importances."""
    def plot_function(df):
        # Sort by importance and take top 15
        df = df.sort_values('importance', ascending=False).head(15)

        plt.figure(figsize=(12, 8))
        plt.barh(df['feature'], df['importance'])
        plt.title('Top 15 Feature Importances')
        plt.xlabel('Importance')
        plt.grid(axis='x', alpha=0.3)

    return create_plot(plot_function, (feature_importance_df,), {}, save_path)


def create_predicted_vs_actual_plot(y_test, y_pred, model_name, metrics, save_path=None):
    """Create a scatter plot of predicted vs actual values."""
    def plot_function(y_test, y_pred, model_name, metrics):
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

    return create_plot(plot_function, (y_test, y_pred, model_name, metrics), {}, save_path)


def create_customer_states_plot(df, save_path=None):
    """Create a bar chart of top 10 customer states."""
    def plot_function(df):
        # Count states
        state_counts = df['customer_state'].value_counts().head(10)

        # Create a mapping from state codes to proper state names (if available)
        state_names = {
            'SP': 'São Paulo', 'RJ': 'Rio de Janeiro', 'MG': 'Minas Gerais',
            'RS': 'Rio Grande do Sul', 'PR': 'Paraná', 'SC': 'Santa Catarina',
            'BA': 'Bahia', 'DF': 'Distrito Federal', 'GO': 'Goiás',
            'ES': 'Espírito Santo'
        }

        plt.figure(figsize=(12, 6))
        plt.bar(state_counts.index, state_counts.values)
        plt.title('Top 10 Customer States')
        plt.xlabel('Customer State')
        plt.ylabel('Count')
        plt.grid(axis='y', alpha=0.3)

    return create_plot(plot_function, (df,), {}, save_path)


def evaluate_model(y_true, y_pred):
    """Calculate common regression metrics for model evaluation"""
    metrics = {
        'r2': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    return metrics


def log_visualizations(paths, artifact_dir="visualizations"):
    """Log multiple visualization artifacts to MLflow"""
    for path in paths:
        if path:
            mlflow.log_artifact(path, artifact_dir)


def create_wrapper_predict_fn(model, scaler=None, is_xgboost=False):
    """Create a standardized prediction function"""
    def predict_fn(X_df):
        if is_xgboost:
            if not isinstance(X_df, xgb.DMatrix):
                X_df = xgb.DMatrix(data=X_df)
            return model.predict(X_df)
        elif scaler is not None:
            X_scaled = scaler.transform(X_df)
            return model.predict(X_scaled, verbose=0).flatten()
        else:
            return model.predict(X_df)
    return predict_fn


def model_training(data_splits, target_column='price'):
    """
    Train multiple models on the prepared data with comprehensive metrics reporting.
    GPU acceleration used for XGBoost and Neural Network models.
    All models report R², RMSE, MAE, MSE, and MAPE metrics.
    """
    print("Starting model training with GPU acceleration for XGBoost and Neural Network...")
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Extract training data
    train_df = data_splits['train']
    features = data_splits.get('features', [col for col in train_df.columns
                                            if col != target_column and col != 'product_id'])

    # Prepare training data
    X_train = train_df[features].copy()
    y_train = train_df[target_column].values

    # Convert integer columns to float to avoid MLflow schema warnings
    int_cols = X_train.select_dtypes(include=['int64', 'int32']).columns
    for col in int_cols:
        X_train[col] = X_train[col].astype(float)

    # Extract test data
    test_df = data_splits['test']
    X_test = test_df[features].copy()
    y_test = test_df[target_column].values

    # Also convert test integer columns to float
    for col in int_cols:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(float)

    print(f"Training data: {X_train.shape[0]} samples, {len(features)} features")

    # Create a temporary directory for visualizations
    artifacts_dir = tempfile.mkdtemp()
    visualizations_dir = os.path.join(artifacts_dir, "visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)

    # Dict to store model results
    model_results = {}

    # Set up cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # ----------------------------------------------------
    # 1. Decision Tree
    # ----------------------------------------------------
    print("\nTraining Decision Tree model...")

    # Use separate MLflow run for Decision Tree model
    with mlflow.start_run(run_name="Decision Tree", nested=True):
        # Train standard scikit-learn Decision Tree
        dt_model = DecisionTreeRegressor(
            max_depth=12,
            min_samples_split=6,
            max_features=0.8,
            random_state=42
        )

        # Cross-validation
        dt_cv_scores = cross_val_score(dt_model, X_train, y_train, cv=kf, scoring='r2')
        dt_cv_mean = dt_cv_scores.mean()
        dt_cv_std = dt_cv_scores.std()

        # Train on full dataset with auto-logging enabled
        mlflow.sklearn.autolog()
        dt_model.fit(X_train, y_train)

    # Get feature importance
    dt_importance = pd.DataFrame({
        'feature': features,
        'importance': dt_model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Test set performance
    dt_pred = dt_model.predict(X_test)
    dt_metrics = evaluate_model(y_test, dt_pred)

    print(f"Decision Tree - Test R²: {dt_metrics['r2']:.4f}, RMSE: {dt_metrics['rmse']:.2f}, "
          f"MAE: {dt_metrics['mae']:.2f}, MSE: {dt_metrics['mse']:.2f}, MAPE: {dt_metrics['mape']:.2f}%")

    # Store results
    model_results['decision_tree'] = {
        'model': dt_model,
        'cv_r2': dt_cv_mean,
        'cv_r2_std': dt_cv_std,
        'test_r2': dt_metrics['r2'],
        'test_rmse': dt_metrics['rmse'],
        'test_mae': dt_metrics['mae'],
        'test_mse': dt_metrics['mse'],
        'test_mape': dt_metrics['mape'],
        'feature_importance': dt_importance,
        'predictions': dt_pred,
        'predict_fn': create_wrapper_predict_fn(dt_model)
    }

    # Create visualizations - save to visualizations directory
    dt_pred_vs_actual_path = os.path.join(visualizations_dir, 'dt_predicted_vs_actual.png')
    create_predicted_vs_actual_plot(y_test, dt_pred, 'Decision Tree',
                                    model_results['decision_tree'], dt_pred_vs_actual_path)

    dt_feature_imp_path = os.path.join(visualizations_dir, 'dt_feature_importance.png')
    create_feature_importance_plot(dt_importance, dt_feature_imp_path)

    # Log visualizations
    log_visualizations([dt_pred_vs_actual_path, dt_feature_imp_path])

    # ----------------------------------------------------
    # 2. Random Forest
    # ----------------------------------------------------
    print("\nTraining Random Forest model...")

    # Use separate MLflow run for Random Forest model
    with mlflow.start_run(run_name="Random Forest", nested=True):
        # Train standard scikit-learn Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=4,
            bootstrap=True,
            max_samples=0.9,
            max_features=0.8,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )

        # Cross-validation
        rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='r2')
        rf_cv_mean = rf_cv_scores.mean()
        rf_cv_std = rf_cv_scores.std()

        # Train on full dataset with auto-logging enabled
        mlflow.sklearn.autolog()
        rf_model.fit(X_train, y_train)

    # Get feature importance
    rf_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Test set performance
    rf_pred = rf_model.predict(X_test)
    rf_metrics = evaluate_model(y_test, rf_pred)

    print(f"Random Forest - Test R²: {rf_metrics['r2']:.4f}, RMSE: {rf_metrics['rmse']:.2f}, "
          f"MAE: {rf_metrics['mae']:.2f}, MSE: {rf_metrics['mse']:.2f}, MAPE: {rf_metrics['mape']:.2f}%")

    # Store results
    model_results['random_forest'] = {
        'model': rf_model,
        'cv_r2': rf_cv_mean,
        'cv_r2_std': rf_cv_std,
        'test_r2': rf_metrics['r2'],
        'test_rmse': rf_metrics['rmse'],
        'test_mae': rf_metrics['mae'],
        'test_mse': rf_metrics['mse'],
        'test_mape': rf_metrics['mape'],
        'feature_importance': rf_importance,
        'predictions': rf_pred,
        'predict_fn': create_wrapper_predict_fn(rf_model)
    }

    # Create and log visualizations to the visualizations directory
    rf_pred_vs_actual_path = os.path.join(visualizations_dir, 'rf_predicted_vs_actual.png')
    create_predicted_vs_actual_plot(y_test, rf_pred, 'Random Forest',
                                    model_results['random_forest'], rf_pred_vs_actual_path)

    rf_feature_imp_path = os.path.join(visualizations_dir, 'rf_feature_importance.png')
    create_feature_importance_plot(rf_importance, rf_feature_imp_path)

    # Log visualizations
    log_visualizations([rf_pred_vs_actual_path, rf_feature_imp_path])

    # ----------------------------------------------------
    # 3. XGBoost with enhanced GPU acceleration
    # ----------------------------------------------------
    print("\nTraining XGBoost model with enhanced GPU acceleration...")

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # XGBoost parameters with updated syntax for newer XGBoost versions (2.0+)
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',  # Updated from 'gpu_hist'
        'device': 'cuda',  # Use CUDA instead of 'gpu_id'
        'max_depth': 12,  # Increased from 11
        'learning_rate': 0.05,  # Reduced to allow more boosting rounds
        'min_child_weight': 4,  # Adjusted from 5
        'subsample': 0.85,  # For better generalization
        'colsample_bytree': 0.85,  # For better generalization
        'colsample_bylevel': 0.9,  # Additional regularization
        'lambda': 1.0,  # L2 regularization
        'alpha': 0.5,  # L1 regularization
        'max_bin': 256,  # Increased histogram bins
        'seed': 42  # Set seed for reproducibility
    }

    # XGBoost built-in cross-validation
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=500,  # Increased from 200
        nfold=5,
        stratified=False,
        metrics={'rmse'},
        early_stopping_rounds=25,  # Increased from 20
        seed=42,
        as_pandas=True
    )

    # Get the optimal number of boosting rounds
    best_round = cv_results.shape[0]
    xgb_cv_rmse = cv_results.iloc[-1]['test-rmse-mean']

    # Print information about the cross-validation
    print(f"XGBoost CV RMSE: {xgb_cv_rmse:.4f} at {best_round} rounds")

    # Train final model with optimal number of rounds
    xgb_model = xgb.train(params, dtrain, num_boost_round=best_round)

    # Make predictions
    xgb_y_pred = xgb_model.predict(dtest)

    # Calculate metrics for XGBoost
    xgb_metrics = evaluate_model(y_test, xgb_y_pred)

    # Print metrics
    print(f"XGBoost - Test R²: {xgb_metrics['r2']:.4f}, RMSE: {xgb_metrics['rmse']:.2f}, "
          f"MAE: {xgb_metrics['mae']:.2f}, MSE: {xgb_metrics['mse']:.2f}, MAPE: {xgb_metrics['mape']:.2f}%")
    print(f"XGBoost - Best boosting rounds: {best_round}")

    # Get feature importance
    importance_scores = xgb_model.get_score(importance_type='gain')
    xgb_importance = pd.DataFrame({
        'feature': list(importance_scores.keys()),
        'importance': list(importance_scores.values())
    }).sort_values('importance', ascending=False)

    # Calculate CV R² (not directly provided by xgb.cv)
    xgb_cv_r2 = 1 - (xgb_cv_rmse ** 2 / np.var(y_train))

    # Store results
    model_results['xgboost_gpu'] = {
        'model': xgb_model,
        'cv_rmse': xgb_cv_rmse,
        'cv_r2': xgb_cv_r2,
        'test_r2': xgb_metrics['r2'],
        'test_rmse': xgb_metrics['rmse'],
        'test_mae': xgb_metrics['mae'],
        'test_mse': xgb_metrics['mse'],
        'test_mape': xgb_metrics['mape'],
        'best_round': best_round,
        'feature_importance': xgb_importance,
        'predictions': xgb_y_pred,
        'predict_fn': create_wrapper_predict_fn(xgb_model, is_xgboost=True)
    }

    # Create and log XGBoost visualizations
    xgb_pred_vs_actual_path = os.path.join(visualizations_dir, 'xgb_predicted_vs_actual.png')
    create_predicted_vs_actual_plot(y_test, xgb_y_pred, 'XGBoost',
                                    model_results['xgboost_gpu'], xgb_pred_vs_actual_path)

    xgb_feature_imp_path = os.path.join(visualizations_dir, 'xgb_feature_importance.png')
    create_feature_importance_plot(xgb_importance, xgb_feature_imp_path)

    xgb_error_by_price_path = os.path.join(visualizations_dir, 'xgb_error_by_price_range.png')
    create_error_by_price_range_plot(y_test, xgb_y_pred, xgb_error_by_price_path)

    # Log visualizations
    log_visualizations([xgb_pred_vs_actual_path, xgb_feature_imp_path, xgb_error_by_price_path])

    # ----------------------------------------------------
    # 4. Neural Network with GPU optimization for 3060 Ti
    # ----------------------------------------------------
    print("\nTraining Neural Network model with GPU optimization...")

    # Use separate MLflow run for Neural Network model
    with mlflow.start_run(run_name="Neural Network", nested=True):
        # Set seeds again for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

        # Disable TensorFlow autologging during training
        mlflow.tensorflow.autolog(disable=True)

        # Manual validation split
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
            layers.Dense(128, activation='relu', kernel_regularizer=l2(0.0005)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(96, activation='relu', kernel_regularizer=l2(0.0005)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu', kernel_regularizer=l2(0.0005)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            layers.Dense(32, activation='relu', kernel_regularizer=l2(0.0005)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(1)
        ])

        # Compile model
        nn_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )

        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )

        # Set up MLflow model saving in Keras format
        class MLflowKerasCallback(tf.keras.callbacks.Callback):
            def on_train_end(self, logs=None):
                keras_path = os.path.join(artifacts_dir, "keras_model.keras")
                self.model.save(keras_path)
                mlflow.log_artifact(keras_path, "model")

        mlflow_callback = MLflowKerasCallback()

        # Train model
        history = nn_model.fit(
            X_train_scaled, y_train_final,
            validation_data=(X_val_scaled, y_val),
            epochs=100,
            batch_size=256,
            callbacks=[early_stop, lr_scheduler, mlflow_callback],
            verbose=1
        )

        # Re-enable TensorFlow autologging
        mlflow.tensorflow.autolog(disable=False)

    # Evaluate on validation set
    nn_val_pred = nn_model.predict(X_val_scaled, verbose=0).flatten()
    nn_val_metrics = evaluate_model(y_val, nn_val_pred)

    print(f"Neural Network - Validation R²: {nn_val_metrics['r2']:.4f}, RMSE: {nn_val_metrics['rmse']:.2f}, "
          f"MAE: {nn_val_metrics['mae']:.2f}, MSE: {nn_val_metrics['mse']:.2f}, MAPE: {nn_val_metrics['mape']:.2f}%")

    # Evaluate on test set
    nn_test_pred = nn_model.predict(X_test_scaled, verbose=0).flatten()
    nn_test_metrics = evaluate_model(y_test, nn_test_pred)

    print(f"Neural Network - Test R²: {nn_test_metrics['r2']:.4f}, RMSE: {nn_test_metrics['rmse']:.2f}, "
          f"MAE: {nn_test_metrics['mae']:.2f}, MSE: {nn_test_metrics['mse']:.2f}, MAPE: {nn_test_metrics['mape']:.2f}%")

    # Store results
    model_results['neural_network'] = {
        'model': nn_model,
        'scaler': scaler,
        'val_rmse': nn_val_metrics['rmse'],
        'val_r2': nn_val_metrics['r2'],
        'val_mae': nn_val_metrics['mae'],
        'val_mse': nn_val_metrics['mse'],
        'val_mape': nn_val_metrics['mape'],
        'test_r2': nn_test_metrics['r2'],
        'test_rmse': nn_test_metrics['rmse'],
        'test_mae': nn_test_metrics['mae'],
        'test_mse': nn_test_metrics['mse'],
        'test_mape': nn_test_metrics['mape'],
        'history': history.history,
        'predictions': nn_test_pred,
        'predict_fn': create_wrapper_predict_fn(nn_model, scaler)
    }

    # Create and log NN visualizations
    nn_pred_vs_actual_path = os.path.join(visualizations_dir, 'nn_predicted_vs_actual.png')
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
    nn_history_path = os.path.join(visualizations_dir, 'nn_training_history.png')
    plt.savefig(nn_history_path, bbox_inches='tight')
    plt.close()

    # Log visualizations
    log_visualizations([nn_pred_vs_actual_path, nn_history_path])

    # ----------------------------------------------------
    # Model Comparison and Selection
    # ----------------------------------------------------

    # Define all models for comparison
    all_models = ['decision_tree', 'random_forest', 'xgboost_gpu', 'neural_network']

    # Find best model based on test R²
    best_model_name = max(all_models, key=lambda x: model_results[x]['test_r2'])
    print(f"\nBest model: {best_model_name}")

    # Create summary DataFrame
    summary_data = [{
        'Model': model_name,
        'Test R²': model_results[model_name]['test_r2'],
        'Test RMSE': model_results[model_name]['test_rmse'],
        'Test MAE': model_results[model_name]['test_mae'],
        'Test MAPE (%)': model_results[model_name]['test_mape']
    } for model_name in all_models]

    summary_df = pd.DataFrame(summary_data).sort_values('Test R²', ascending=False)
    print("\nModel Performance Comparison:")
    print(summary_df)

    # Print top features from best model
    if 'feature_importance' in model_results[best_model_name]:
        print(f"\nTop 10 Important Features ({best_model_name}):")
        print(model_results[best_model_name]['feature_importance'].head(10))

    # Create summary visualization
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

    plt.tight_layout()
    summary_path = os.path.join(visualizations_dir, 'model_comparison.png')
    plt.savefig(summary_path, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(summary_path, "visualizations")

    # Log summary table as CSV
    summary_csv_path = os.path.join(artifacts_dir, 'model_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    mlflow.log_artifact(summary_csv_path, "artifacts")

    # Log best model name and GPU parameters
    mlflow.log_param("best_model", best_model_name)
    mlflow.log_param("gpu_acceleration", "True")

    return {
        "models": model_results,
        "best_model_name": best_model_name,
        "features": features,
        "feature_importance": model_results.get(best_model_name, {}).get('feature_importance'),
        "summary": summary_df,
        "predict_fn": model_results[best_model_name].get('predict_fn')
    }
