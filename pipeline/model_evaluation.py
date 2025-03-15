import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import os


def model_evaluation(training_results, data_splits, target_column='price'):
    """Evaluate models on test data and provide comprehensive metrics"""
    print("Starting model evaluation...")

    # Extract data and models
    test_df = data_splits['test']
    features = training_results['features']
    models = training_results['models']
    best_model_name = training_results['best_model_name']

    # Prepare test data
    X_test = test_df[features].copy()
    y_test = test_df[target_column].values

    # Helper function to save plots
    def save_plot(filename):
        try:
            plt.savefig(filename)
            try:
                mlflow.log_artifact(filename)
            except:
                pass
            os.remove(filename)
        except Exception as e:
            print(f"Warning: Error saving plot {filename}: {e}")
        finally:
            plt.close()

    # Helper function for metrics
    def calculate_metrics(y_true, y_pred):
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        # Calculate MAPE with handling for zeros
        with np.errstate(divide='ignore', invalid='ignore'):
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 0.01))) * 100
        return metrics

    # Evaluate each model
    evaluation_results = {}

    for model_name, model_info in models.items():
        model = model_info['model']

        # Special handling for neural network (needs scaling)
        if model_name == 'neural_network':
            y_pred = model.predict(model_info['scaler'].transform(X_test)).flatten()
        else:
            y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred)
        metrics['predictions'] = y_pred
        evaluation_results[model_name] = metrics

        # Log metrics
        try:
            mlflow.log_metrics({
                f"{model_name}_test_rmse": metrics['rmse'],
                f"{model_name}_test_r2": metrics['r2'],
                f"{model_name}_test_mape": metrics['mape']
            })
        except:
            pass

        print(f"{model_name.replace('_', ' ').title()} - Test RMSE: {metrics['rmse']:.2f}, "
              f"R²: {metrics['r2']:.4f}, MAPE: {metrics['mape']:.2f}%")

    # Determine best model on test data
    best_test_model_name = max(evaluation_results, key=lambda x: evaluation_results[x]['r2'])
    best_test_model = evaluation_results[best_test_model_name]
    best_model_info = models[best_test_model_name]
    y_pred = best_test_model['predictions']

    print(f"\nBest model on test data: {best_test_model_name} with R²: {best_test_model['r2']:.4f}")

    # 1. Create prediction vs actual scatter plot
    try:
        plt.figure(figsize=(10, 6))
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
            f"RMSE: {best_test_model['rmse']:.2f}\nR²: {best_test_model['r2']:.4f}\n"
            f"MAPE: {best_test_model['mape']:.2f}%",
            xy=(0.05, 0.95), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
            fontsize=10, ha='left', va='top')

        save_plot("pred_vs_actual.png")
    except Exception as e:
        print(f"Warning: Error creating prediction vs actual plot: {e}")

    # 2. Create error distribution plot
    try:
        plt.figure(figsize=(10, 6))
        errors = y_pred - y_test
        errors = errors[np.abs(errors) < np.percentile(np.abs(errors), 95)]

        sns.histplot(errors, bins=30, kde=True)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Prediction Error Distribution')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        save_plot("error_distribution.png")
    except Exception as e:
        print(f"Warning: Error creating error distribution plot: {e}")

    # 3. Create feature importance plot
    try:
        if 'feature_importance' in best_model_info:
            fi = best_model_info['feature_importance']
            top_n = min(15, len(fi))

            plt.figure(figsize=(12, 8))
            plt.barh(fi['feature'][:top_n], fi['importance'][:top_n])
            plt.title(f'Top {top_n} Feature Importances')
            plt.xlabel('Importance')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3)

            save_plot("feature_importance.png")

            # Print top features
            print("\nTop 10 feature importances:")
            for _, row in fi.head(10).iterrows():
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
                                                bins=price_buckets, labels=bucket_labels)

        # Calculate error metrics
        test_with_pred['abs_error'] = np.abs(test_with_pred['prediction'] - test_with_pred[target_column])
        test_with_pred['rel_error'] = test_with_pred['abs_error'] / np.maximum(test_with_pred[target_column], 0.01)

        # Aggregate by price bucket
        bucket_metrics = test_with_pred.groupby('price_bucket', observed=False).agg({
            'abs_error': 'mean',
            'rel_error': lambda x: np.mean(x) * 100,  # Convert to percentage
            target_column: 'count'
        }).reset_index()

        bucket_metrics.columns = ['Price Range', 'MAE', 'MAPE (%)', 'Count']

        # Create plot
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Price Range', y='MAPE (%)', data=bucket_metrics)

        # Add count annotations
        for i, count in enumerate(bucket_metrics['Count']):
            plt.text(i, bucket_metrics['MAPE (%)'][i] + 1, f'n={count}', ha='center')

        plt.title('Error by Price Range')
        plt.ylabel('Mean Absolute Percentage Error (%)')
        plt.grid(axis='y', alpha=0.3)

        save_plot("error_by_price.png")

        # Print summary
        print("\nError by price range:")
        for _, row in bucket_metrics.iterrows():
            print(f"  {row['Price Range']}: MAPE: {row['MAPE (%)']:.2f}%, Count: {row['Count']}")
    except Exception as e:
        print(f"Warning: Error creating price range analysis: {e}")

    # Return evaluation summary
    evaluation_summary = {
        'best_model_name': best_test_model_name,
        'metrics': {k: best_test_model[k] for k in ['rmse', 'mae', 'r2', 'mape']},
        'all_results': evaluation_results
    }

    # Log overall best model metrics
    try:
        mlflow.log_param("best_test_model", best_test_model_name)
        mlflow.log_metrics({
            "best_test_rmse": best_test_model['rmse'],
            "best_test_r2": best_test_model['r2'],
            "best_test_mape": best_test_model['mape']
        })
    except:
        pass

    return evaluation_summary