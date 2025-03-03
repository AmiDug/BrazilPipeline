import json
import os
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def data_validation(df):

    print("Starting data validation...")

    # Configure validation expectations for Olist dataset
    schema_config = {
        'expected_columns': [
            'id', 'title', 'price', 'description', 'category', 'image',
            'rate', 'count', 'product_weight_g', 'volume_cm3', 'freight_value',
            'title_length', 'description_length', 'image_count', 'density', 'price_per_gram'
        ],
        'dtypes': {
            'id': 'object',  # String ID
            'title': 'object',  # String title
            'price': 'float64',  # Float price
            'description': 'object',  # String description
            'category': 'object',  # String category
            'image': 'object',  # String image URL
            'rate': 'float64',  # Float rating
            'count': 'int64',  # Integer count
            'product_weight_g': 'float64',  # Float weight
            'volume_cm3': 'float64',  # Float volume
            'freight_value': 'float64',  # Float freight value
            'title_length': 'float64',  # Float title length
            'description_length': 'float64',  # Float description length
            'image_count': 'float64',  # Float image count
            'density': 'float64',  # Float density
            'price_per_gram': 'float64'  # Float price per gram
        },
        'ranges': {
            'price': {'min': 0},  # Prices must be positive
            'rate': {'min': 1, 'max': 5},  # Ratings between 1-5
            'count': {'min': 1},  # At least one order per product
            'product_weight_g': {'min': 0},  # Weight must be positive
            'volume_cm3': {'min': 0},  # Volume must be positive
            'freight_value': {'min': 0}  # Freight value must be positive
        },
        'required_columns': ['id', 'price', 'category', 'product_weight_g', 'volume_cm3']  # Essential fields
    }

    # Initialize validation tracking
    validation_passed = True
    validation_results = {}

    # 1. Check for expected columns
    expected_cols = set(schema_config.get('expected_columns', []))
    actual_cols = set(df.columns)
    missing_cols = expected_cols - actual_cols

    if missing_cols:
        validation_passed = False
        validation_results['missing_columns'] = list(missing_cols)
        mlflow.log_param('schema_valid', False)
        mlflow.log_metric('missing_columns', len(missing_cols))
    else:
        mlflow.log_param('schema_valid', True)
        mlflow.log_metric('missing_columns', 0)

    # 2. Check for missing values in required columns
    required_cols = schema_config.get('required_columns', [])
    required_with_nulls = {col: df[col].isnull().sum() for col in required_cols
                           if col in df.columns and df[col].isnull().any()}

    if required_with_nulls:
        validation_passed = False
        validation_results['null_in_required'] = required_with_nulls
        mlflow.log_param('nulls_valid', False)
    else:
        mlflow.log_param('nulls_valid', True)

    # 3. Check data types
    expected_dtypes = schema_config.get('dtypes', {})
    incorrect_dtypes = {}

    for col, expected_dtype in expected_dtypes.items():
        if col in df.columns and str(df[col].dtype) != expected_dtype:
            incorrect_dtypes[col] = {
                'expected': expected_dtype,
                'actual': str(df[col].dtype)
            }

    if incorrect_dtypes:
        validation_passed = False
        validation_results['incorrect_dtypes'] = incorrect_dtypes
        mlflow.log_param('dtypes_valid', False)
        mlflow.log_metric('incorrect_dtypes', len(incorrect_dtypes))
    else:
        mlflow.log_param('dtypes_valid', True)
        mlflow.log_metric('incorrect_dtypes', 0)

    # 4. Check for duplicates in id (should be unique)
    id_duplicates = df['id'].duplicated().sum() if 'id' in df.columns else 0
    validation_results['duplicate_ids'] = int(id_duplicates)
    mlflow.log_metric('duplicate_ids', id_duplicates)

    if id_duplicates > 0:
        validation_passed = False
        mlflow.log_param('unique_ids_valid', False)
    else:
        mlflow.log_param('unique_ids_valid', True)

    # 5. Range validations
    range_validations = schema_config.get('ranges', {})
    range_failures = {}

    for col, ranges in range_validations.items():
        if col in df.columns and not df[col].isnull().all():
            min_val, max_val = ranges.get('min'), ranges.get('max')
            failures = 0

            if min_val is not None and (df[col] < min_val).any():
                failures += (df[col] < min_val).sum()

            if max_val is not None and (df[col] > max_val).any():
                failures += (df[col] > max_val).sum()

            if failures > 0:
                range_failures[col] = int(failures)

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

    # Log overall validation status
    mlflow.log_param('validation_passed', validation_passed)

    # Save detailed validation results as artifact
    with open('validation_results.json', 'w') as f:
        json.dump(validation_results, f)

    mlflow.log_artifact('validation_results.json')
    os.remove('validation_results.json')

    print(f"Data validation {'passed' if validation_passed else 'failed'}")
    if not validation_passed:
        print(f"Validation issues: {validation_results}")

    return validation_passed, validation_results, df