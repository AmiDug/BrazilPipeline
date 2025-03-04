import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import mlflow
import os


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km"""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371 * c  # Earth radius in kilometers


def get_geo_coordinates(df_geolocation, zip_code):
    """Get average coordinates for a zip code"""
    location_data = df_geolocation[df_geolocation['geolocation_zip_code_prefix'] == zip_code]
    if len(location_data) > 0:
        return location_data['geolocation_lat'].mean(), location_data['geolocation_lng'].mean()
    return 0, 0  # Default if no coordinates found


def data_transformation(df, df_geolocation=None, test_size=0.2, random_state=42):
    """Enhanced data transformation with geographical and text features"""
    print("Starting data transformation...")

    # Create a copy to avoid modifying the original
    df_clean = df.copy()

    # Basic cleaning: Replace inf with NaN and fill NaN with median values
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df_clean.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Handle missing categories
    if 'category' in df_clean.columns and df_clean['category'].isnull().any():
        print(f"Filling {df_clean['category'].isnull().sum()} missing category values with 'Unknown'")
        df_clean['category'] = df_clean['category'].fillna('Unknown')

    # Remove price-derived features to prevent data leakage
    leaky_features = ['price_min', 'price_max', 'price_variance', 'price_per_gram',
                      'price_to_freight_ratio', 'log_price', 'category_mean_price']
    leaky_features = [f for f in leaky_features if f in df_clean.columns]
    if leaky_features:
        print(f"Removing {len(leaky_features)} price-derived features")
        df_clean = df_clean.drop(leaky_features, axis=1)

    # ENHANCEMENT 1: Add geographic distance feature if geolocation data is available
    if df_geolocation is not None and 'customer_zip_code_prefix' in df_clean.columns and 'seller_zip_code_prefix' in df_clean.columns:
        print("Adding customer-seller distance feature...")
        df_clean['distance_customer_seller'] = df_clean.apply(
            lambda row: haversine_distance(
                *get_geo_coordinates(df_geolocation, row['customer_zip_code_prefix']),
                *get_geo_coordinates(df_geolocation, row['seller_zip_code_prefix'])
            ), axis=1
        )

    # ENHANCEMENT 2: Add text features
    if 'product_name_lenght' in df_clean.columns:
        print("Adding text features...")
        df_clean['product_name_lenght'] = df_clean['product_name_lenght'].fillna(0)

    if 'product_description_lenght' in df_clean.columns:
        df_clean['product_description_lenght'] = df_clean['product_description_lenght'].fillna(0)
        if 'price' in df_clean.columns:
            df_clean['description_price_ratio'] = df_clean['product_description_lenght'] / df_clean['price'].replace(0,
                                                                                                                     np.nan)
            df_clean['description_price_ratio'] = df_clean['description_price_ratio'].fillna(0)

    # ENHANCEMENT 3: Add temporal features
    if 'order_purchase_timestamp' in df_clean.columns:
        print("Adding temporal features...")
        df_clean['order_purchase_timestamp'] = pd.to_datetime(df_clean['order_purchase_timestamp'])
        df_clean['order_purchase_year'] = df_clean['order_purchase_timestamp'].dt.year
        df_clean['order_purchase_month'] = df_clean['order_purchase_timestamp'].dt.month
        df_clean['order_purchase_day'] = df_clean['order_purchase_timestamp'].dt.day
        df_clean['order_purchase_weekday'] = df_clean['order_purchase_timestamp'].dt.weekday
        df_clean['is_holiday_season'] = df_clean['order_purchase_month'].isin([11, 12]).astype(int)

    # Create interaction features
    if all(col in df_clean.columns for col in ['product_weight_g', 'volume_cm3']):
        df_clean['weight_volume_ratio'] = df_clean['product_weight_g'] / df_clean['volume_cm3'].replace(0, np.nan)
        df_clean['weight_volume_ratio'] = df_clean['weight_volume_ratio'].fillna(0)

    # Remove outliers from price
    if 'price' in df_clean.columns:
        q_low = df_clean['price'].quantile(0.005)
        q_high = df_clean['price'].quantile(0.995)
        df_clean = df_clean[(df_clean['price'] >= q_low) & (df_clean['price'] <= q_high)]

    # FIX: Identify categorical and numerical columns
    categorical_cols = [col for col in df_clean.columns if
                        df_clean[col].dtype == 'object' and col != 'id']
    numerical_cols = [col for col in df_clean.columns if
                      df_clean[col].dtype in ['int64', 'float64'] and
                      col != 'price' and col != 'id']

    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")

    # Handle categorical features
    for col in categorical_cols:
        # Convert to category codes
        df_clean[f'{col}_code'] = df_clean[col].astype('category').cat.codes

        # Add frequency encoding for large cardinality categories
        value_counts = df_clean[col].value_counts(normalize=True)
        df_clean[f'{col}_freq'] = df_clean[col].map(value_counts)

    # FIX: Drop original categorical columns after encoding
    df_clean = df_clean.drop(categorical_cols, axis=1)

    # Check for any remaining NaN values
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(0)

    # Split data
    train_df, test_df = train_test_split(df_clean, test_size=test_size, random_state=random_state)

    # Log split sizes
    mlflow.log_param("train_samples", len(train_df))
    mlflow.log_param("test_samples", len(test_df))

    # Return data
    return {
        "train": train_df,
        "test": test_df,
        "selected_features": [col for col in df_clean.columns if col != 'price' and col != 'id']
    }