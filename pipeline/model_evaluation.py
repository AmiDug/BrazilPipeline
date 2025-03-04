import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import os


def model_evaluation(training_results, data_splits, target_column='price'):
    """
    Evaluate models on test data and provide comprehensive metrics
    """
    print("Starting model evaluation...")

    # Extract test data
    test_df = data_splits['test']

    # Extract features
    features = training_results['features']

    # Extract models
    models = training_results['models']
    best_model_name = training_results['best_model_name']

    # Prepare test data
    X_test = test_df[features].copy()
    y_test = test_df[target_column].values

    # Create dictionary to store evaluation metrics
    evaluation_results = {}

    # Evaluate each model
    for model_name, model_info in models.items():
        model = model_info['model']

        # Special handling for neural network (needs scaling)
        if model_name == 'neural_network':
            scaler = model_info['scaler']
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled).flatten()
        else:
            y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Calculate MAPE with handling for zeros
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 0.01))) * 100

        # Store metrics
        evaluation_results[model_name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'predictions': y_pred
        }

        # Log metrics
        try:
            mlflow.log_metric(f"{model_name}_test_rmse", rmse)
            mlflow.log_metric(f"{model_name}_test_r2", r2)
            mlflow.log_metric(f"{model_name}_test_mape", mape)
        except:
            pass

        print(f"{model_name.replace('_', ' ').title()} - Test RMSE: {rmse:.2f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")

    # Determine best model on test data
    best_test_model_name = max(evaluation_results, key=lambda x: evaluation_results[x]['r2'])
    best_test_model = evaluation_results[best_test_model_name]

    print(f"\nBest model on test data: {best_test_model_name} with R²: {best_test_model['r2']:.4f}")

    # Create visualization for best model
    best_model_info = models[best_test_model_name]
    y_pred = best_test_model['predictions']

    # 1. Create prediction vs actual scatter plot
    try:
        plt.figure(figsize=(10, 6))

        # Filter out extreme values for better visualization
        mask = (y_test < np.percentile(y_test, 99)) & (y_pred < np.percentile(y_pred, 99))
        plt.scatter(y_test[mask], y_pred[mask], alpha=0.5, s=10)

        # Add perfect prediction line
        min_val = max(0, min(np.min(y_test[mask]), np.min(y_pred[mask])))
        max_val = max(np.max(y_test[mask]), np.max(y_pred[mask]))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        plt.title(f'{best_test_model_name.replace("_", " ").title()} - Predicted vs Actual')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.grid(True, alpha=0.3)

        # Add metrics annotation
        plt.annotate(
            f"RMSE: {best_test_model['rmse']:.2f}\nR²: {best_test_model['r2']:.4f}\nMAPE: {best_test_model['mape']:.2f}%",
            xy=(0.05, 0.95), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
            fontsize=10, ha='left', va='top')

        # Save figure for MLflow
        pred_vs_actual_path = "pred_vs_actual.png"
        plt.savefig(pred_vs_actual_path)
        try:
            mlflow.log_artifact(pred_vs_actual_path)
        except:
            pass
        os.remove(pred_vs_actual_path)
        plt.close()
    except Exception as e:
        print(f"Warning: Error creating prediction vs actual plot: {e}")

    # 2. Create error distribution plot
    try:
        plt.figure(figsize=(10, 6))
        errors = y_pred - y_test

        # Filter out extreme errors for better visualization
        errors = errors[np.abs(errors) < np.percentile(np.abs(errors), 95)]

        sns.histplot(errors, bins=30, kde=True)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Prediction Error Distribution')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        # Save figure for MLflow
        error_dist_path = "error_distribution.png"
        plt.savefig(error_dist_path)
        try:
            mlflow.log_artifact(error_dist_path)
        except:
            pass
        os.remove(error_dist_path)
        plt.close()
    except Exception as e:
        print(f"Warning: Error creating error distribution plot: {e}")

    # 3. Create feature importance plot
    try:
        if 'feature_importance' in best_model_info:
            feature_importance = best_model_info['feature_importance']

            plt.figure(figsize=(12, 8))
            top_n = min(15, len(feature_importance))
            plt.barh(feature_importance['feature'][:top_n], feature_importance['importance'][:top_n])
            plt.title(f'Top {top_n} Feature Importances')
            plt.xlabel('Importance')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3)

            # Save figure for MLflow
            importance_path = "feature_importance.png"
            plt.savefig(importance_path)
            try:
                mlflow.log_artifact(importance_path)
            except:
                pass
            os.remove(importance_path)
            plt.close()

            # Print top features
            print("\nTop 10 feature importances:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
    except Exception as e:
        print(f"Warning: Error creating feature importance plot: {e}")

    # 4. Create price range accuracy analysis
    try:
        test_with_pred = test_df.copy()
        test_with_pred['prediction'] = y_pred

        # Create price buckets
        price_buckets = [0, 50, 100, 200, 500, 1000, float('inf')]
        bucket_labels = ['0-50', '50-100', '100-200', '200-500', '500-1000', '1000+']

        test_with_pred['price_bucket'] = pd.cut(test_with_pred[target_column],
                                                bins=price_buckets,
                                                labels=bucket_labels)

        # Calculate error metrics by price bucket
        test_with_pred['abs_error'] = np.abs(test_with_pred['prediction'] - test_with_pred[target_column])

        # Safe relative error calculation
        test_with_pred['rel_error'] = test_with_pred['abs_error'] / np.maximum(test_with_pred[target_column], 0.01)

        # Aggregate by price bucket
        bucket_metrics = test_with_pred.groupby('price_bucket').agg({
            'abs_error': 'mean',
            'rel_error': lambda x: np.mean(x) * 100,  # Convert to percentage
            target_column: 'count'
        }).reset_index()

        bucket_metrics.columns = ['Price Range', 'MAE', 'MAPE (%)', 'Count']

        # Create plot
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Price Range', y='MAPE (%)', data=bucket_metrics)

        # Add count annotations
        for i, count in enumerate(bucket_metrics['Count']):
            plt.text(i, bucket_metrics['MAPE (%)'][i] + 1, f'n={count}', ha='center')

        plt.title('Error by Price Range')
        plt.ylabel('Mean Absolute Percentage Error (%)')
        plt.grid(axis='y', alpha=0.3)

        # Save figure for MLflow
        price_error_path = "error_by_price.png"
        plt.savefig(price_error_path)
        try:
            mlflow.log_artifact(price_error_path)
        except:
            pass
        os.remove(price_error_path)
        plt.close()

        # Print summary
        print("\nError by price range:")
        for _, row in bucket_metrics.iterrows():
            print(f"  {row['Price Range']}: MAPE: {row['MAPE (%)']:.2f}%, Count: {row['Count']}")
    except Exception as e:
        print(f"Warning: Error creating price range analysis: {e}")

    # Return evaluation summary
    evaluation_summary = {
        'best_model_name': best_test_model_name,
        'metrics': {
            'rmse': best_test_model['rmse'],
            'mae': best_test_model['mae'],
            'r2': best_test_model['r2'],
            'mape': best_test_model['mape']
        },
        'all_results': evaluation_results
    }

    # Log overall best model metrics
    try:
        mlflow.log_param("best_test_model", best_test_model_name)
        mlflow.log_metric("best_test_rmse", best_test_model['rmse'])
        mlflow.log_metric("best_test_r2", best_test_model['r2'])
        mlflow.log_metric("best_test_mape", best_test_model['mape'])
    except:
        pass

    return evaluation_summary