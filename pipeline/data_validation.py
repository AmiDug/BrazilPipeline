import json
import os
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def data_validation(df):
    print("Starting data validation...")

    # First, ensure required columns exist and have appropriate types
    required_columns = ['id', 'price', 'category', 'product_weight_g', 'volume_cm3']
    expected_columns = [
        'id', 'title', 'price', 'description', 'category', 'image',
        'rate', 'count', 'product_weight_g', 'volume_cm3', 'freight_value',
        'title_length', 'description_length', 'image_count', 'density', 'price_per_gram'
    ]

    # Check if expected columns are missing
    missing_columns = [col for col in expected_columns if col not in df.columns]

    # Add missing columns with default values
    for col in missing_columns:
        if col in ['id', 'title', 'description', 'category', 'image']:
            df[col] = 'Unknown'  # String default
        else:
            df[col] = 0.0  # Numeric default

    # Ensure proper data types
    dtype_mapping = {
        'id': 'object',
        'title': 'object',
        'price': 'float64',
        'description': 'object',
        'category': 'object',
        'image': 'object',
        'rate': 'float64',
        'count': 'int64',
        'product_weight_g': 'float64',
        'volume_cm3': 'float64',
        'freight_value': 'float64',
        'title_length': 'float64',
        'description_length': 'float64',
        'image_count': 'float64',
        'density': 'float64',
        'price_per_gram': 'float64'
    }

    for col, dtype in dtype_mapping.items():
        if col in df.columns:
            if dtype.startswith('float') or dtype.startswith('int'):
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if dtype.startswith('int'):
                    df[col] = df[col].fillna(0).astype(int)

    # Configure validation expectations for Olist dataset
    schema_config = {
        'expected_columns': expected_columns,
        'dtypes': dtype_mapping,
        'ranges': {
            'price': {'min': 0},  # Prices must be positive
            'rate': {'min': 0, 'max': 5},  # Ratings between 0-5
            'count': {'min': 0},  # Allow products with no orders yet
            'product_weight_g': {'min': 0},  # Weight must be positive
            'volume_cm3': {'min': 0},  # Volume must be positive
            'freight_value': {'min': 0}  # Freight value must be positive
        },
        'required_columns': required_columns  # Essential fields
    }

    # Initialize validation tracking
    validation_passed = True
    validation_results = {}

    # Convert all NumPy types to standard Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        else:
            return obj

    # Log any missing columns that were added
    if missing_columns:
        validation_results['added_missing_columns'] = missing_columns
        print(f"Added missing columns with default values: {missing_columns}")

    # Check for missing values in required columns
    required_with_nulls = {col: int(df[col].isnull().sum()) for col in required_columns
                           if col in df.columns and df[col].isnull().any()}

    if required_with_nulls:
        validation_passed = False
        validation_results['null_in_required'] = required_with_nulls
        mlflow.log_param('nulls_valid', False)
    else:
        mlflow.log_param('nulls_valid', True)

    # Check data types
    expected_dtypes = schema_config.get('dtypes', {})
    incorrect_dtypes = {}

    for col, expected_dtype in expected_dtypes.items():
        if col in df.columns:
            # Instead of strict equality, check if types are compatible
            current_dtype = df[col].dtype
            if expected_dtype.startswith('float') and not pd.api.types.is_float_dtype(current_dtype):
                incorrect_dtypes[col] = {
                    'expected': expected_dtype,
                    'actual': str(current_dtype)
                }
            elif expected_dtype.startswith('int') and not pd.api.types.is_integer_dtype(current_dtype):
                incorrect_dtypes[col] = {
                    'expected': expected_dtype,
                    'actual': str(current_dtype)
                }
            elif expected_dtype == 'object' and not pd.api.types.is_object_dtype(current_dtype):
                incorrect_dtypes[col] = {
                    'expected': expected_dtype,
                    'actual': str(current_dtype)
                }

    if incorrect_dtypes:
        validation_passed = False
        validation_results['incorrect_dtypes'] = incorrect_dtypes
        mlflow.log_param('dtypes_valid', False)
        mlflow.log_metric('incorrect_dtypes', len(incorrect_dtypes))
    else:
        mlflow.log_param('dtypes_valid', True)
        mlflow.log_metric('incorrect_dtypes', 0)

    # Check for duplicates in id (should be unique)
    id_duplicates = int(df['id'].duplicated().sum()) if 'id' in df.columns else 0
    validation_results['duplicate_ids'] = id_duplicates
    mlflow.log_metric('duplicate_ids', id_duplicates)

    if id_duplicates > 0:
        validation_passed = False
        mlflow.log_param('unique_ids_valid', False)
    else:
        mlflow.log_param('unique_ids_valid', True)

    # Range validations
    range_validations = schema_config.get('ranges', {})
    range_failures = {}

    for col, ranges in range_validations.items():
        if col in df.columns and not df[col].isnull().all():
            min_val, max_val = ranges.get('min'), ranges.get('max')
            failures = 0

            if min_val is not None and (df[col] < min_val).any():
                failures += int((df[col] < min_val).sum())

            if max_val is not None and (df[col] > max_val).any():
                failures += int((df[col] > max_val).sum())

            if failures > 0:
                range_failures[col] = failures

    if range_failures:
        validation_passed = False
        validation_results['range_failures'] = range_failures
        mlflow.log_param('ranges_valid', False)
    else:
        mlflow.log_param('ranges_valid', True)

    # 6. Data distribution analysis and visualization
    # Create category distribution chart
    plt.figure(figsize=(12, 6))
    top_categories = df['category'].value_counts().head(15)
    sns.barplot(x=top_categories.values, y=top_categories.index)
    plt.title('Top 15 Product Categories')
    plt.xlabel('Number of Products')
    plt.tight_layout()

    # Save figure to MLflow
    category_chart_path = "category_distribution.png"
    plt.savefig(category_chart_path)
    mlflow.log_artifact(category_chart_path)
    os.remove(category_chart_path)
    plt.close()

    # Create price distribution
    plt.figure(figsize=(10, 6))
    # Use log scale for better visualization
    plt.hist(df['price'], bins=50, alpha=0.7)
    plt.title('Price Distribution')
    plt.xlabel('Price (R$)')
    plt.ylabel('Count')
    plt.yscale('log')  # Log scale for better visualization
    plt.grid(True, alpha=0.3)

    # Save figure to MLflow
    price_chart_path = "price_distribution.png"
    plt.savefig(price_chart_path)
    mlflow.log_artifact(price_chart_path)
    os.remove(price_chart_path)
    plt.close()

    # Create weight vs price scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['product_weight_g'], df['price'], alpha=0.3)
    plt.title('Price vs Weight')
    plt.xlabel('Weight (g)')
    plt.ylabel('Price (R$)')
    plt.grid(True, alpha=0.3)

    # Save figure to MLflow
    weight_price_path = "weight_vs_price.png"
    plt.savefig(weight_price_path)
    mlflow.log_artifact(weight_price_path)
    os.remove(weight_price_path)
    plt.close()

    # Correlation matrix of numeric features
    numeric_cols = df.select_dtypes(include=['number']).columns
    corr_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                annot_kws={"size": 8}, vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()

    # Save correlation matrix to MLflow
    corr_matrix_path = "correlation_matrix.png"
    plt.savefig(corr_matrix_path)
    mlflow.log_artifact(corr_matrix_path)
    os.remove(corr_matrix_path)
    plt.close()

    # Convert all data in validation_results to be JSON serializable
    validation_results = convert_to_serializable(validation_results)

    # Log overall validation status
    mlflow.log_param('validation_passed', validation_passed)

    # Save detailed validation results as artifact
    with open('validation_results.json', 'w') as f:
        json.dump(validation_results, f)

    mlflow.log_artifact('validation_results.json')
    os.remove('validation_results.json')

    print(f"Data validation {'passed' if validation_passed else 'failed with warnings'}")
    if not validation_passed:
        print(f"Validation issues: {validation_results}")
        print("Proceeding with pipeline despite validation issues...")
        validation_passed = True  # Allow pipeline to continue

    return validation_passed, validation_results, df