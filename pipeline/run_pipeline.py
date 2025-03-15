import mlflow
import sys
import traceback
from data_ingestion import data_ingestion
from data_validation import data_validation
from data_transformation import data_transformation
from model_training import model_training
from model_evaluation import model_evaluation


def run_pipeline():
    """Run the complete Olist price prediction pipeline"""
    # Set up MLflow tracking
    try:
        mlflow.set_tracking_uri("http://127.0.0.1:8080")
        mlflow.set_experiment("Olist-Price-Prediction-Simplified")
    except Exception as e:
        print(f"Warning: MLflow setup error (continuing without tracking): {e}")

    def print_step(step_num, step_name):
        """Helper function to print step headers"""
        print(f"\n{'=' * 50}\nSTEP {step_num}: {step_name}\n{'=' * 50}")

    def run_step(step_num, step_name, func, critical=True, **kwargs):
        """Run a pipeline step with standardized error handling"""
        print_step(step_num, step_name)
        try:
            return func(**kwargs)
        except Exception as e:
            error_type = "Error" if critical else "Warning"
            print(f"{error_type}: {step_name} failed: {e}")
            if critical:
                print(traceback.format_exc())
                return None
            return kwargs.get('default_return')

    try:
        with mlflow.start_run() as run:
            # Step 1: Data Ingestion
            df = run_step(1, "DATA INGESTION", data_ingestion)
            if df is None or len(df) == 0:
                print("Error: Data ingestion failed. Pipeline cannot continue.")
                return None

            # Step 2: Data Validation
            _, _, df = run_step(2, "DATA VALIDATION", data_validation,
                                critical=False, default_return=(False, {}, df), df=df)

            # Step 3: Data Transformation
            data_splits = run_step(3, "DATA TRANSFORMATION", data_transformation, df=df)
            if data_splits is None:
                return None

            # Step 4: Model Training
            training_results = run_step(4, "MODEL TRAINING", model_training,
                                        data_splits=data_splits, target_column='price')
            if training_results is None:
                return None

            # Step 5: Model Evaluation
            evaluation_results = run_step(5, "MODEL EVALUATION", model_evaluation,
                                          training_results=training_results,
                                          data_splits=data_splits, target_column='price')
            if evaluation_results is None:
                return None

            # Print final summary
            metrics = evaluation_results['metrics']
            print(f"\n{'=' * 50}\nPIPELINE COMPLETED SUCCESSFULLY\n{'=' * 50}")
            print(f"Dataset: Olist E-commerce Dataset")
            print(f"Best model: {evaluation_results['best_model_name']}")
            print(f"RMSE: R${metrics['rmse']:.2f} | MAE: R${metrics['mae']:.2f}")
            print(f"R²: {metrics['r2']:.4f} | MAPE: {metrics['mape']:.2f}%")
            print('=' * 50)

            return {
                "training_results": training_results,
                "evaluation_results": evaluation_results
            }
    except Exception as e:
        print(f"Critical error in pipeline: {e}")
        print(traceback.format_exc())
        return None


if __name__ == "__main__":
    results = run_pipeline()
    if results is None:
        sys.exit(1)