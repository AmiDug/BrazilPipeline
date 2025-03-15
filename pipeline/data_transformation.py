import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow


def data_transformation(df, test_size=0.2, random_state=42):
    """Transform data by creating features, handling missing values, and preparing for model training"""
    print("Starting data transformation...")
    df_clean = df.copy()

    # 1. Handle missing values
    # Fill categorical values
    cat_cols = ['product_category_name_english', 'customer_state', 'seller_state',
                'customer_city', 'seller_city', 'payment_type']
    for col in [c for c in cat_cols if c in df_clean.columns]:
        if (missing := df_clean[col].isnull().sum()) > 0:
            print(f"Filling {missing} missing {col} values with 'unknown'")
            df_clean[col] = df_clean[col].fillna('unknown')

    # Fill numeric values
    num_cols = ['product_weight_g', 'product_length_cm', 'product_height_cm',
                'product_width_cm', 'product_name_lenght', 'product_description_lenght',
                'product_photos_qty', 'payment_installments', 'review_score']
    for col in [c for c in num_cols if c in df_clean.columns]:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(0)

    # 2. Feature engineering
    print("Creating features...")

    # Helper function to create features if columns exist
    def create_if_exists(condition, col_name, calculation):
        if all(col in df_clean.columns for col in condition):
            df_clean[col_name] = calculation

    # Calculate volume and ratios
    create_if_exists(
        ['product_length_cm', 'product_height_cm', 'product_width_cm'],
        'volume_cm3',
        df_clean['product_length_cm'] * df_clean['product_height_cm'] * df_clean['product_width_cm']
    )

    create_if_exists(
        ['freight_value', 'product_weight_g'],
        'freight_weight_ratio',
        df_clean['freight_value'] / df_clean['product_weight_g'].replace(0, 1)
    )

    create_if_exists(
        ['customer_state', 'seller_state'],
        'same_state',
        (df_clean['customer_state'] == df_clean['seller_state']).astype(int)
    )

    # 3. Encode categorical features
    print("Encoding categorical features...")

    # Helper function for frequency and one-hot encoding
    def encode_categorical(col, threshold, prefix=None):
        if col not in df_clean.columns:
            return

        # Create frequency encoding
        counts = df_clean[col].value_counts(normalize=True)
        df_clean[f'{col}_freq'] = df_clean[col].map(counts)

        # One-hot encode top values
        top_values = counts[counts > threshold].index.tolist()
        prefix = prefix or col
        for val in top_values:
            df_clean[f'{prefix}_{val}'] = (df_clean[col] == val).astype(int)

    # Encode states, payment types, categories and cities
    for state_col in ['customer_state', 'seller_state']:
        encode_categorical(state_col, 0.03)

    for col in ['payment_type', 'product_category_name_english']:
        encode_categorical(col, 0.03)

    for city_col in ['customer_city', 'seller_city']:
        encode_categorical(city_col, 0.05)

    # 4. Handle outliers
    print("Handling outliers...")

    # Cap numeric features
    for col in [c for c in ['product_weight_g', 'product_length_cm', 'product_height_cm',
                            'product_width_cm', 'freight_value', 'volume_cm3'] if c in df_clean.columns]:
        cap_value = df_clean[col].quantile(0.99)
        df_clean[col] = df_clean[col].clip(upper=cap_value)

    # Handle price outliers
    if 'price' in df_clean.columns:
        q_low, q_high = df_clean['price'].quantile([0.001, 0.999])
        rows_before = len(df_clean)
        df_clean = df_clean[(df_clean['price'] >= q_low) & (df_clean['price'] <= q_high)]
        print(f"Removed {rows_before - len(df_clean)} price outliers")

    # 5. Drop original categorical columns
    cols_to_drop = [col for col in ['customer_state', 'seller_state', 'payment_type',
                                    'product_category_name_english', 'customer_city', 'seller_city',
                                    'product_category_name', 'order_id', 'customer_id', 'order_item_id']
                    if col in df_clean.columns]

    df_clean = df_clean.drop(cols_to_drop, axis=1)

    # 6. Clean up any remaining issues
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 7. Select features for modeling
    feature_cols = [col for col in df_clean.columns
                    if col != 'price' and (pd.api.types.is_numeric_dtype(df_clean[col]) or
                                           pd.api.types.is_bool_dtype(df_clean[col]))]

    # Add product_id if needed
    target_and_id = ['price']
    if 'product_id' in df_clean.columns:
        target_and_id.append('product_id')

    # Final selected columns
    selected_columns = target_and_id + feature_cols
    df_features = df_clean[selected_columns].copy()

    print(f"Final dataset: {df_features.shape}, {len(feature_cols)} features for modeling")

    # 8. Split data
    features = df_features.drop(['price'], axis=1)
    target = df_features['price']

    # Handle ID column if present
    id_col = None
    if 'product_id' in features.columns:
        id_col = features['product_id']
        features = features.drop(['product_id'], axis=1)
        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
            features, target, id_col, test_size=test_size, random_state=random_state
        )
        X_train['product_id'] = id_train
        X_test['product_id'] = id_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )

    # Create output dataframes
    train_df = X_train.copy()
    train_df['price'] = y_train
    test_df = X_test.copy()
    test_df['price'] = y_test

    # Log parameters
    try:
        mlflow.log_params({
            "num_features": len(feature_cols),
            "train_size": len(train_df),
            "test_size": len(test_df),
            "test_fraction": test_size
        })
    except:
        print("Warning: Could not log to MLflow")

    print(f"Split data: {len(train_df)} training samples, {len(test_df)} test samples")

    return {
        "train": train_df,
        "test": test_df,
        "features": feature_cols,
        "id_column": 'product_id' if 'product_id' in df_features.columns else None
    }