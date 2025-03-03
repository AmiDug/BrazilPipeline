import mlflow
from data_ingestion import data_ingestion
from data_validation import data_validation
from data_transformation import data_transformation
from model_training import model_training
from model_evaluation import model_evaluation


def run_pipeline():

    # Set up MLflow tracking
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("HMCrawler")
    mlflow.autolog()

    with mlflow.start_run() as run:
        # Step 1: Data ingestion - loads data from database
        df = data_ingestion()

        # Step 2: Data validation - checks data quality
        validation_passed, validation_results, df = data_validation(df)

        # Only proceed if validation passes
        if not validation_passed:
            print("Data validation failed. Pipeline halted.")
            print(f"Validation issues: {validation_results}")
            return None

        # Step 3: Data transformation - splits data
        data_splits = data_transformation(df)

        # Step 4: Model training - trains and compares models
        model_results = model_training(data_splits)

        # Step 5: Model evaluation - evaluates on test data
        eval_metrics = model_evaluation(
            model_results=model_results,
            data_splits=data_splits,
            target_column='price'
        )

        # Print final pipeline summary
        best_model_name = model_results['best_model']
        print("\n" + "=" * 50)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print(f"Best model: {best_model_name}")
        print(f"RMSE: ${eval_metrics['rmse']:.2f}")
        print(f"MAE: ${eval_metrics['mae']:.2f}")
        print(f"R²: {eval_metrics['r2']:.3f}")
        print("=" * 50)

        return {
            "model_results": model_results,
            "evaluation_metrics": eval_metrics
        }

if __name__ == "__main__":
    run_pipeline()