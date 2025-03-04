import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import os
import matplotlib.pyplot as plt
import seaborn as sns


def data_transformation(df, test_size=0.2, random_state=42):
    print("Starting data transformation...")

    # Create a copy to avoid modifying the original
    df_clean = df.copy()

    # Basic cleaning: Replace inf with NaN and fill NaN with median values
    print("Cleaning data: replacing inf values and filling NaN values...")
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

    for col in numeric_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Handle missing categories
    if 'category' in df_clean.columns and df_clean['category'].isnull().any():
        missing_categories = df_clean['category'].isnull().sum()
        print(f"Filling {missing_categories} missing category values with 'Unknown'")
        df_clean['category'] = df_clean['category'].fillna('Unknown')

    # Remove ALL price-derived features to prevent data leakage
    price_derived_features = [
        'price_min', 'price_max', 'price_variance', 'price_per_gram',
        'price_to_freight_ratio', 'description_price_ratio', 'price_cv',
        'category_avg_price', 'log_price', 'category_mean_price',
        'price_ratio'
    ]

    # Only remove features that actually exist in the dataframe
    features_to_remove = [f for f in price_derived_features if f in df_clean.columns]
    if features_to_remove:
        print(f"Removing {len(features_to_remove)} price-derived features: {features_to_remove}")
        df_clean = df_clean.drop(features_to_remove, axis=1)

    # Handle categorical features
    categorical_cols = ['category', 'customer_state', 'seller_state']
    for col in categorical_cols:
        if col in df_clean.columns:
            # Create robust category codes that handle new categories at prediction time
            df_clean[f'{col}_code'] = pd.Categorical(df_clean[col]).codes

            # Add frequency encoding
            value_counts = df_clean[col].value_counts(normalize=True)
            df_clean[f'{col}_freq'] = df_clean[col].map(value_counts)

    # Create safe interaction features
    if all(col in df_clean.columns for col in ['product_weight_g', 'volume_cm3']):
        df_clean['weight_volume_ratio'] = df_clean['product_weight_g'] / df_clean['volume_cm3'].replace(0, np.nan)
        df_clean['weight_volume_ratio'] = df_clean['weight_volume_ratio'].fillna(0)

    # Remove outliers from target column only
    q_low = df_clean['price'].quantile(0.005)
    q_high = df_clean['price'].quantile(0.995)
    df_clean = df_clean[(df_clean['price'] >= q_low) & (df_clean['price'] <= q_high)]

    # Log outlier removal stats
    mlflow.log_metric("rows_after_outlier_removal", len(df_clean))

    # Find legitimate features (exclude price-derived features)
    target = 'price'
    correlations = {}
    for col in numeric_cols:
        if col != target and col not in price_derived_features:
            corr = df_clean[col].corr(df_clean[target])
            if not np.isnan(corr):
                correlations[col] = abs(corr)

    correlations = {k: v for k, v in sorted(correlations.items(),
                                            key=lambda item: item[1],
                                            reverse=True)}

    # Select top features (limit to 15 for simplicity)
    top_features = list(correlations.keys())[:15]

    # Core features to always include if present (excluding price-derived features)
    core_features = ['product_weight_g', 'volume_cm3', 'freight_value',
                     'density', 'count', 'category_code', 'category_freq',
                     'description_length', 'weight_volume_ratio',
                     'seller_order_count']

    # Ensure core features are included
    selected_features = list(set(top_features +
                                 [f for f in core_features if f in df_clean.columns]))

    print(f"Selected {len(selected_features)} legitimate features:")
    for i, feature in enumerate(selected_features):
        corr_value = correlations.get(feature, float('nan'))
        if not np.isnan(corr_value):
            print(f"  {i + 1}. {feature} (correlation: {corr_value:.3f})")
        else:
            print(f"  {i + 1}. {feature}")

    # Final check for any remaining NaN values
    for col in df_clean[selected_features].columns:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(0)

    # Log transformation parameters
    mlflow.log_param("transformation_features", ", ".join(selected_features))
    mlflow.log_param("test_size_fraction", test_size)

    # Create train-test split
    train_df, test_df = train_test_split(
        df_clean, test_size=test_size, random_state=random_state
    )

    # Return a dictionary with train/test data and selected features
    return {
        "train": train_df,
        "test": test_df,
        "selected_features": selected_features,
        "categorical_cols": [col for col in categorical_cols if col in df_clean.columns]
    }