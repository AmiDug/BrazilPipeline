import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow


def data_transformation(df, test_size=0.2, random_state=42):
    """
    Transform the data by creating features, handling missing values,
    and preparing for model training
    """
    print("Starting data transformation...")

    # Create a copy to avoid modifying the original
    df_clean = df.copy()

    # 1. Handle missing values
    print("Handling missing values...")
    # Fill missing categorical values
    categorical_cols = ['product_category_name_english', 'customer_state', 'seller_state',
                        'customer_city', 'seller_city', 'payment_type']

    for col in categorical_cols:
        if col in df_clean.columns:
            missing_count = df_clean[col].isnull().sum()
            if missing_count > 0:
                print(f"Filling {missing_count} missing {col} values with 'unknown'")
                df_clean[col] = df_clean[col].fillna('unknown')

    # Fill missing numeric values
    numeric_cols = ['product_weight_g', 'product_length_cm', 'product_height_cm',
                    'product_width_cm', 'product_name_lenght', 'product_description_lenght',
                    'product_photos_qty', 'payment_installments', 'review_score']

    for col in numeric_cols:
        if col in df_clean.columns and df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(0)

    # 2. Feature engineering
    print("Creating features...")

    # Calculate volume
    if all(col in df_clean.columns for col in
           ['product_length_cm', 'product_height_cm', 'product_width_cm']):
        df_clean['volume_cm3'] = (
                df_clean['product_length_cm'] *
                df_clean['product_height_cm'] *
                df_clean['product_width_cm']
        )

    # Calculate freight to weight ratio
    if all(col in df_clean.columns for col in ['freight_value', 'product_weight_g']):
        df_clean['freight_weight_ratio'] = df_clean['freight_value'] / df_clean['product_weight_g'].replace(0, 1)

    # State and city match
    if all(col in df_clean.columns for col in ['customer_state', 'seller_state']):
        df_clean['same_state'] = (df_clean['customer_state'] == df_clean['seller_state']).astype(int)

    # 3. Encode categorical features
    print("Encoding categorical features...")

    # State frequency encoding
    for state_col in ['customer_state', 'seller_state']:
        if state_col in df_clean.columns:
            # Create frequency encoding
            state_counts = df_clean[state_col].value_counts(normalize=True)
            df_clean[f'{state_col}_freq'] = df_clean[state_col].map(state_counts)

            # One-hot encode top states only
            top_states = state_counts[state_counts > 0.03].index.tolist()  # States with >3% frequency
            for state in top_states:
                df_clean[f'{state_col}_{state}'] = (df_clean[state_col] == state).astype(int)

    # Payment type and product category encoding
    for col in ['payment_type', 'product_category_name_english']:
        if col in df_clean.columns:
            # Only keep most frequent values, group others
            value_counts = df_clean[col].value_counts(normalize=True)
            top_values = value_counts[value_counts >= 0.03].index.tolist()  # Values with >3% frequency

            # One-hot encode only top values
            for val in top_values:
                df_clean[f'{col}_{val}'] = (df_clean[col] == val).astype(int)

    # City encoding - focus on top cities only
    for city_col in ['customer_city', 'seller_city']:
        if city_col in df_clean.columns:
            # Get top cities
            city_counts = df_clean[city_col].value_counts(normalize=True)
            top_cities = city_counts[city_counts > 0.05].index.tolist()  # Cities with >5% frequency

            # Create binary features for top cities
            for city in top_cities:
                df_clean[f'{city_col}_{city}'] = (df_clean[city_col] == city).astype(int)

    # 4. Handle outliers
    print("Handling outliers...")

    # Cap numeric features at 99th percentile
    numeric_features = ['product_weight_g', 'product_length_cm', 'product_height_cm',
                        'product_width_cm', 'freight_value', 'volume_cm3']

    for col in numeric_features:
        if col in df_clean.columns:
            cap_value = df_clean[col].quantile(0.99)
            df_clean[col] = df_clean[col].clip(upper=cap_value)

    # Handle price outliers
    if 'price' in df_clean.columns:
        q_low = df_clean['price'].quantile(0.001)
        q_high = df_clean['price'].quantile(0.999)
        df_clean = df_clean[(df_clean['price'] >= q_low) & (df_clean['price'] <= q_high)]
        print(f"Removed {len(df) - len(df_clean)} price outliers")

    # 5. Drop original categorical columns and unnecessary columns
    print("Finalizing features...")

    columns_to_drop = [
        'customer_state', 'seller_state', 'payment_type',
        'product_category_name_english', 'customer_city', 'seller_city',
        'product_category_name', 'order_id', 'customer_id', 'order_item_id',
        # Don't drop product_id as it might be needed for identification
    ]

    # Only drop columns that exist
    columns_to_drop = [col for col in columns_to_drop if col in df_clean.columns]
    df_clean = df_clean.drop(columns_to_drop, axis=1)

    # 6. Remove any infinity/NaN values
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.fillna(0)

    # 7. Select features for modeling
    # Keep only numeric and boolean columns, plus id for reference
    feature_cols = [col for col in df_clean.columns
                    if col != 'price' and (pd.api.types.is_numeric_dtype(df_clean[col]) or
                                           pd.api.types.is_bool_dtype(df_clean[col]))]

    # Add product_id if needed for identification
    if 'product_id' in df_clean.columns:
        target_and_id = ['price', 'product_id']
    else:
        target_and_id = ['price']

    # Final selected columns
    selected_columns = target_and_id + feature_cols
    df_features = df_clean[selected_columns].copy()

    print(f"Final dataset shape: {df_features.shape}")
    print(f"Selected {len(feature_cols)} features for modeling")

    # 8. Split data into train and test sets
    features = df_features.drop(['price'], axis=1)
    target = df_features['price']

    # For identification, we might need to separate product_id
    if 'product_id' in features.columns:
        product_id = features['product_id']
        features = features.drop(['product_id'], axis=1)
        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
            features, target, product_id, test_size=test_size, random_state=random_state
        )
        # Reattach product_id
        X_train['product_id'] = id_train
        X_test['product_id'] = id_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )

    # Create train and test dataframes
    train_df = X_train.copy()
    train_df['price'] = y_train

    test_df = X_test.copy()
    test_df['price'] = y_test

    # Log transformation parameters
    try:
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("test_size", len(test_df))
        mlflow.log_param("test_fraction", test_size)
    except:
        print("Warning: Could not log to MLflow (continuing)")

    print(f"Split data into {len(train_df)} training samples and {len(test_df)} test samples")

    return {
        "train": train_df,
        "test": test_df,
        "features": feature_cols,
        "id_column": 'product_id' if 'product_id' in df_features.columns else None
    }