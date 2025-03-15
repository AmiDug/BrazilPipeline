import os
# Set matplotlib to use a non-GUI backend BEFORE any other imports
import matplotlib

matplotlib.use('Agg')  # Force non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow


def model_evaluation(training_results, data_splits, target_column='price'):
    """Evaluate models on test data"""
    print("Starting model evaluation...")

    test_df = data_splits['test']
    features = training_results['features']
    models = training_results['models']

    X_test = test_df[features].copy()
    y_test = test_df[target_column].values

    evaluation_results = {}

    # Evaluate each model
    for model_name, model_info in models.items():
        model = model_info['model']

        # Neural network needs scaling
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
        mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 0.01))) * 100

        evaluation_results[model_name] = {
            'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
            'mape': mape, 'predictions': y_pred
        }

        print(f"{model_name} - R²: {r2:.4f}, RMSE: {rmse:.2f}")

        # Log metrics
        try:
            mlflow.log_metrics({
                f"{model_name}_test_rmse": rmse,
                f"{model_name}_test_r2": r2
            })
        except:
            pass

    # Find best model
    best_model_name = max(evaluation_results, key=lambda x: evaluation_results[x]['r2'])
    best_model = evaluation_results[best_model_name]
    y_pred = best_model['predictions']

    print(f"\nBest model: {best_model_name} with R²: {best_model['r2']:.4f}")

    # Create prediction vs actual scatter plot
    try:
        plt.figure(figsize=(10, 6))
        mask = (y_test < np.percentile(y_test, 99)) & (y_pred < np.percentile(y_pred, 99))
        plt.scatter(y_test[mask], y_pred[mask], alpha=0.5, s=10)

        # Perfect prediction line
        min_val = min(np.min(y_test[mask]), np.min(y_pred[mask]))
        max_val = max(np.max(y_test[mask]), np.max(y_pred[mask]))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        plt.title(f'{best_model_name} - Predicted vs Actual')
        plt.xlabel('Actual');
        plt.ylabel('Predicted')
        plt.grid(True, alpha=0.3)

        # Save plot
        plt.savefig("pred_vs_actual.png")
        try:
            mlflow.log_artifact("pred_vs_actual.png")
        except:
            pass
        os.remove("pred_vs_actual.png")
        plt.close()
    except Exception as e:
        print(f"Warning: Plot error: {e}")

    # Feature importance plot
    try:
        if 'feature_importance' in models[best_model_name]:
            fi = models[best_model_name]['feature_importance']
            top_n = min(10, len(fi))

            plt.figure(figsize=(12, 6))
            plt.barh(fi['feature'][:top_n], fi['importance'][:top_n])
            plt.title('Top Feature Importances')
            plt.gca().invert_yaxis()

            plt.savefig("feature_importance.png")
            try:
                mlflow.log_artifact("feature_importance.png")
            except:
                pass
            os.remove("feature_importance.png")
            plt.close()
    except Exception as e:
        print(f"Warning: Plot error: {e}")

    # Price error analysis
    try:
        test_with_pred = test_df.copy()
        test_with_pred['prediction'] = y_pred

        # Create buckets
        bins = [0, 50, 100, 200, 500, 1000, float('inf')]
        labels = ['0-50', '50-100', '100-200', '200-500', '500-1000', '1000+']

        test_with_pred['price_bucket'] = pd.cut(test_with_pred[target_column], bins=bins, labels=labels)
        test_with_pred['abs_error'] = np.abs(test_with_pred['prediction'] - test_with_pred[target_column])
        test_with_pred['rel_error'] = test_with_pred['abs_error'] / np.maximum(test_with_pred[target_column], 0.01)

        # Group by price bucket - fix observed parameter
        metrics = test_with_pred.groupby('price_bucket', observed=False).agg({
            'abs_error': 'mean',
            'rel_error': lambda x: np.mean(x) * 100
        }).reset_index()
    except Exception as e:
        print(f"Warning: Error analysis error: {e}")

    # Return summary
    return {
        'best_model_name': best_model_name,
        'metrics': {k: best_model[k] for k in ['rmse', 'mae', 'r2', 'mape']},
        'all_results': evaluation_results
    }