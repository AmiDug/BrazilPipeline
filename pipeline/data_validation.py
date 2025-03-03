import json
import os
import mlflow

def data_validation(df):
    schema_config = {
        'expected_columns': ['id', 'title', 'price', 'description', 'category', 'image', 'rate', 'count'],
        'dtypes': {
            'id': 'int64',
            'title': 'object',
            'price': 'float64',
            'description': 'object',
            'category': 'object',
            'image': 'object',
            'rate': 'float64',
            'count': 'int64'
        },
        'ranges': {
            'price': {'min': 0},  # Prices shouldn't be negative
            'rate': {'min': 0, 'max': 5},  # Assuming rating is 0-5
            'count': {'min': 0}  # Count shouldn't be negative
        },
        'required_columns': ['id', 'title', 'price', 'category']  # Essential fields
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

    # Log overall validation status
    mlflow.log_param('validation_passed', validation_passed)

    # Save detailed validation results as artifact
    with open('validation_results.json', 'w') as f:
        json.dump(validation_results, f)

    mlflow.log_artifact('validation_results.json')
    os.remove('validation_results.json')

    return validation_passed, validation_results, df