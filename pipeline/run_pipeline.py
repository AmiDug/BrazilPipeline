import mlflow
from data_ingestion import data_ingestion
from data_validation import data_validation
from data_transformation import data_transformation
from model_training import model_training
from model_evaluation import model_evaluation


def run_pipeline(log_transform=True):
    """
    Run the complete Olist product price prediction pipeline

    Args:
        log_transform (bool): Whether to use log-transformed price as target
    """
    # Set up MLflow tracking
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("Olist-Product-Price-Prediction-Improved")
    mlflow.xgboost.autolog()
    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:
        # Step 1: Data ingestion - load Olist dataset
        print("\n" + "=" * 50)
        print("STEP 1: DATA INGESTION")
        print("=" * 50)
        df = data_ingestion()

        # Step 2: Data validation - check data quality
        print("\n" + "=" * 50)
        print("STEP 2: DATA VALIDATION")
        print("=" * 50)
        validation_passed, validation_results, df = data_validation(df)

        # Continue even if validation has warnings
        if not validation_passed:
            print("Data validation failed with warnings")
            print(f"Validation issues: {validation_results}")
            print("Proceeding with pipeline despite validation issues...")

        # Step 3: Data transformation - clean and split data
        print("\n" + "=" * 50)
        print("STEP 3: DATA TRANSFORMATION")
        print("=" * 50)
        data_splits = data_transformation(df)

        # Step 4: Model training - train multiple models
        print("\n" + "=" * 50)
        print("STEP 4: MODEL TRAINING")
        print("=" * 50)
        model_results = model_training(
            data_splits=data_splits,
            target_column='price',
            log_transform=log_transform
        )

        # Step 5: Model evaluation - evaluate on test data
        print("\n" + "=" * 50)
        print("STEP 5: MODEL EVALUATION")
        print("=" * 50)
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
        print(f"Dataset: Olist Brazilian E-commerce")
        print(f"Task: Product Price Prediction")
        print(f"Target transformation: {'Log-transform' if log_transform else 'None'}")
        print(f"Best model: {best_model_name}")
        print(f"RMSE: R${eval_metrics['rmse']:.2f}")
        print(f"MAE: R${eval_metrics['mae']:.2f}")
        print(f"R²: {eval_metrics['r2']:.3f}")
        print(f"MAPE: {eval_metrics['mape']:.1f}%")
        print("=" * 50)

        return {
            "model_results": model_results,
            "evaluation_metrics": eval_metrics
        }


if __name__ == "__main__":
    run_pipeline(log_transform=True)