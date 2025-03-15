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
        mlflow.set_experiment("Olist-Price-Prediction-CV")
    except Exception as e:
        print(f"Warning: MLflow setup error: {e}")

    try:
        with mlflow.start_run() as run:
            # Step 1: Data ingestion
            print("\n" + "=" * 50 + "\nSTEP 1: DATA INGESTION\n" + "=" * 50)
            df = data_ingestion()
            if df is None or len(df) == 0:
                print("Error: Data ingestion failed.")
                return None

            # Step 2: Data validation
            print("\n" + "=" * 50 + "\nSTEP 2: DATA VALIDATION\n" + "=" * 50)
            try:
                _, _, df = data_validation(df)
            except Exception as e:
                print(f"Warning: Data validation error: {e}")

            # Step 3: Data transformation
            print("\n" + "=" * 50 + "\nSTEP 3: DATA TRANSFORMATION\n" + "=" * 50)
            data_splits = data_transformation(df)
            if data_splits is None:
                return None

            # Step 4: Model training
            print("\n" + "=" * 50 + "\nSTEP 4: MODEL TRAINING\n" + "=" * 50)
            training_results = model_training(data_splits=data_splits, target_column='price')
            if training_results is None:
                return None

            # Step 5: Model evaluation
            print("\n" + "=" * 50 + "\nSTEP 5: MODEL EVALUATION\n" + "=" * 50)
            evaluation_results = model_evaluation(
                training_results=training_results,
                data_splits=data_splits,
                target_column='price'
            )
            if evaluation_results is None:
                return None

            # Print final summary
            best_model_name = evaluation_results['best_model_name']
            metrics = evaluation_results['metrics']
            print("\n" + "=" * 50 + "\nPIPELINE COMPLETED SUCCESSFULLY\n" + "=" * 50)
            print(f"Best model: {best_model_name}")
            print(f"R²: {metrics['r2']:.4f} | RMSE: R${metrics['rmse']:.2f}")
            print("=" * 50)

            return {"training_results": training_results, "evaluation_results": evaluation_results}
    except Exception as e:
        print(f"Critical error: {e}")
        print(traceback.format_exc())
        return None


if __name__ == "__main__":
    results = run_pipeline()
    if results is None:
        sys.exit(1)