import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow

def data_transformation(df, test_size=0.3, random_state=19):
    """Transform data using notebook's approach for higher accuracy"""
    print("Starting data transformation...")
    df_clean = df.copy()

    # 1. Handle missing values
    print("Handling missing values...")
    # Fill missing categorical values
    cat_cols = ['product_category_name_english', 'customer_state', 'seller_state',
                'customer_city', 'seller_city', 'payment_type']
    for col in cat_cols:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna('unknown')

    # Fill missing numeric values
    num_cols = ['product_weight_g', 'product_length_cm', 'product_height_cm',
                'product_width_cm', 'product_name_lenght', 'product_description_lenght',
                'product_photos_qty', 'payment_installments', 'review_score']
    for col in num_cols:
        if col in df_clean.columns and df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(0)

    # 2. Handle outliers with capping
    print("Handling outliers...")
    numeric_features = ['product_weight_g', 'product_length_cm', 'product_height_cm',
                       'product_width_cm', 'freight_value']
    for col in numeric_features:
        if col in df_clean.columns:
            cap_value = df_clean[col].quantile(0.995)
            df_clean[col] = df_clean[col].clip(upper=cap_value)

    # Handle price outliers
    if 'price' in df_clean.columns:
        q_low, q_high = df_clean['price'].quantile([0.001, 0.999])
        rows_before = len(df_clean)
        df_clean = df_clean[(df_clean['price'] >= q_low) & (df_clean['price'] <= q_high)]
        print(f"Removed {rows_before - len(df_clean)} price outliers")

    # 3. Create basic volume feature (same as notebook)
    if all(col in df_clean.columns for col in ['product_length_cm', 'product_height_cm', 'product_width_cm']):
        df_clean['volume_cm3'] = df_clean['product_length_cm'] * df_clean['product_height_cm'] * df_clean['product_width_cm']

    # 4. Label Encoding for categorical variables (like the notebook)
    print("Applying Label Encoding to categorical variables...")
    le = LabelEncoder()
    for col in cat_cols:
        if col in df_clean.columns:
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            print(f"Label encoded {col}")

    # 5. Drop non-modeling columns
    print("Finalizing features...")
    drop_cols = ['product_category_name', 'order_id', 'customer_id', 'order_item_id',
                 'payment_sequential', 'payment_type', 'order_status']  # Same as notebook
    cols_to_drop = [col for col in drop_cols if col in df_clean.columns]
    df_clean = df_clean.drop(cols_to_drop, axis=1)

    # 6. Remove any infinity/NaN values
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 7. Select numeric and boolean columns for modeling (similar to notebook)
    feature_cols = [col for col in df_clean.columns
                   if col != 'price' and (pd.api.types.is_numeric_dtype(df_clean[col]) or
                                         pd.api.types.is_bool_dtype(df_clean[col]))]

    # Add product_id if needed for identification
    target_and_id = ['price']
    if 'product_id' in df_clean.columns:
        target_and_id.append('product_id')

    # 8. Create final dataset
    selected_columns = target_and_id + feature_cols
    df_features = df_clean[selected_columns].copy()
    print(f"Final dataset shape: {df_features.shape}, {len(feature_cols)} features")

    # 9. Split data using notebook's parameters
    features = df_features.drop(['price'], axis=1)
    target = df_features['price']

    # Handle product_id in splitting
    if 'product_id' in features.columns:
        product_id = features['product_id']
        features = features.drop(['product_id'], axis=1)
        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
            features, target, product_id, test_size=test_size, random_state=random_state
        )
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

    # Log parameters
    try:
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("test_size", len(test_df))
    except:
        print("Warning: Could not log to MLflow")

    print(f"Split data: {len(train_df)} training samples, {len(test_df)} test samples")

    return {
        "train": train_df,
        "test": test_df,
        "features": feature_cols,
        "id_column": 'product_id' if 'product_id' in df_features.columns else None
    }