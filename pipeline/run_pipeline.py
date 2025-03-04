import mlflow
import sys
import traceback
from data_ingestion import data_ingestion
from data_validation import data_validation
from data_transformation import data_transformation
from model_training import model_training
from model_evaluation import model_evaluation


def run_pipeline():
    """
    Run the complete Olist price prediction pipeline
    """
    # Set up MLflow tracking
    try:
        mlflow.set_tracking_uri("http://127.0.0.1:8080")
        mlflow.set_experiment("Olist-Price-Prediction-Simplified")
    except Exception as e:
        print(f"Warning: MLflow setup error (continuing without tracking): {e}")

    try:
        with mlflow.start_run() as run:
            # Step 1: Data ingestion - load Olist dataset from Kaggle
            print("\n" + "=" * 50)
            print("STEP 1: DATA INGESTION")
            print("=" * 50)
            df = data_ingestion()

            if df is None or len(df) == 0:
                print("Error: Data ingestion failed. Pipeline cannot continue.")
                return None

            # Step 2: Data validation - check data quality
            print("\n" + "=" * 50)
            print("STEP 2: DATA VALIDATION")
            print("=" * 50)
            try:
                validation_passed, validation_results, df = data_validation(df)
                if not validation_passed:
                    print("Data validation failed with warnings")
                    print("Proceeding with pipeline despite validation issues...")
            except Exception as e:
                print(f"Warning: Data validation error: {e}")
                print("Proceeding with pipeline despite validation issues...")

            # Step 3: Data transformation - prepare features and split data
            print("\n" + "=" * 50)
            print("STEP 3: DATA TRANSFORMATION")
            print("=" * 50)
            try:
                data_splits = data_transformation(df)
            except Exception as e:
                print(f"Error: Data transformation failed: {e}")
                print(traceback.format_exc())
                return None

            # Step 4: Model training - train multiple models
            print("\n" + "=" * 50)
            print("STEP 4: MODEL TRAINING")
            print("=" * 50)
            try:
                training_results = model_training(
                    data_splits=data_splits,
                    target_column='price'
                )
            except Exception as e:
                print(f"Error: Model training failed: {e}")
                print(traceback.format_exc())
                return None

            # Step 5: Model evaluation - evaluate on test data
            print("\n" + "=" * 50)
            print("STEP 5: MODEL EVALUATION")
            print("=" * 50)
            try:
                evaluation_results = model_evaluation(
                    training_results=training_results,
                    data_splits=data_splits,
                    target_column='price'
                )
            except Exception as e:
                print(f"Error: Model evaluation failed: {e}")
                print(traceback.format_exc())
                return None

            # Print final pipeline summary
            best_model_name = evaluation_results['best_model_name']
            metrics = evaluation_results['metrics']

            print("\n" + "=" * 50)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 50)
            print(f"Dataset: Olist E-commerce Dataset")
            print(f"Best model: {best_model_name}")
            print(f"RMSE: R${metrics['rmse']:.2f}")
            print(f"MAE: R${metrics['mae']:.2f}")
            print(f"R²: {metrics['r2']:.4f}")
            print(f"MAPE: {metrics['mape']:.2f}%")
            print("=" * 50)

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