import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import os
from category_encoders import TargetEncoder


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

    # Handle missing categories more effectively
    if 'category' in df_clean.columns and df_clean['category'].isnull().any():
        missing_categories = df_clean['category'].isnull().sum()
        print(f"Filling {missing_categories} missing category values with 'Unknown'")
        df_clean['category'] = df_clean['category'].fillna('Unknown')

    # Remove price-derived features to prevent data leakage
    price_derived_features = [
        'price_min', 'price_max', 'price_variance', 'price_per_gram',
        'price_to_freight_ratio', 'description_price_ratio', 'log_price'
    ]
    features_to_remove = [f for f in price_derived_features if f in df_clean.columns]
    if features_to_remove:
        print(f"Removing {len(features_to_remove)} price-derived features: {features_to_remove}")
        df_clean = df_clean.drop(features_to_remove, axis=1)

    # IMPROVEMENT 1: Create log-transformed target for better distribution handling
    df_clean['log_price_target'] = np.log1p(df_clean['price'])

    # IMPROVEMENT 2: Better outlier handling based on product weight and dimensions
    for col in ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']:
        if col in df_clean.columns:
            q_low = df_clean[col].quantile(0.001)
            q_high = df_clean[col].quantile(0.999)
            df_clean.loc[df_clean[col] < q_low, col] = q_low
            df_clean.loc[df_clean[col] > q_high, col] = q_high

    # IMPROVEMENT 3: Create more effective interaction features
    if all(col in df_clean.columns for col in ['product_weight_g', 'volume_cm3']):
        # Weight density feature
        df_clean['weight_volume_ratio'] = df_clean['product_weight_g'] / df_clean['volume_cm3'].replace(0, 1)

    # IMPROVEMENT 4: Calculate freight value to weight ratio
    if all(col in df_clean.columns for col in ['freight_value', 'product_weight_g']):
        df_clean['freight_weight_ratio'] = df_clean['freight_value'] / df_clean['product_weight_g'].replace(0, 1)

    # IMPROVEMENT 5: Create size-price related features
    if 'volume_cm3' in df_clean.columns:
        df_clean['size_category'] = pd.qcut(df_clean['volume_cm3'], 5, labels=False, duplicates='drop')

    # Remove outliers from target column only
    q_low = df_clean['price'].quantile(0.001)
    q_high = df_clean['price'].quantile(0.999)
    df_clean = df_clean[(df_clean['price'] >= q_low) & (df_clean['price'] <= q_high)]

    # Select features based on correlation with target
    target = 'log_price_target'  # Use log-transformed target for correlation analysis
    correlations = {}
    for col in numeric_cols:
        if col != target and col != 'price' and col not in price_derived_features:
            corr = df_clean[col].corr(df_clean[target])
            if not np.isnan(corr):
                correlations[col] = abs(corr)

    correlations = {k: v for k, v in sorted(correlations.items(),
                                            key=lambda item: item[1],
                                            reverse=True)}

    # Select top correlated features
    top_features = list(correlations.keys())[:20]

    # Important features to always include
    core_features = [
        'product_weight_g', 'volume_cm3', 'freight_value',
        'density', 'count', 'category_code', 'category_freq',
        'description_length', 'weight_volume_ratio',
        'freight_weight_ratio', 'product_length_cm',
        'product_height_cm', 'product_width_cm',
        'freight_value_min', 'freight_value_mean', 'freight_value_max'
    ]

    # Ensure core features are included if they exist
    selected_features = list(set(top_features +
                                 [f for f in core_features if f in df_clean.columns]))

    print(f"Selected {len(selected_features)} legitimate features:")
    for i, feature in enumerate(selected_features):
        corr_value = correlations.get(feature, float('nan'))
        if not np.isnan(corr_value):
            print(f"  {i + 1}. {feature} (correlation: {corr_value:.3f})")
        else:
            print(f"  {i + 1}. {feature}")

    # IMPROVEMENT 6: Create train-test split with stratification on price range
    # This ensures good representation of rare expensive items
    df_clean['price_strata'] = pd.qcut(df_clean['price'], 10, labels=False, duplicates='drop')

    # Split data with stratification
    train_df, test_df = train_test_split(
        df_clean, test_size=test_size, random_state=random_state,
        stratify=df_clean['price_strata']
    )

    # Drop the temporary stratification column
    train_df = train_df.drop('price_strata', axis=1)
    test_df = test_df.drop('price_strata', axis=1)

    # IMPROVEMENT 7: Apply target encoding to categorical columns
    categorical_cols = ['category']
    categorical_encoders = {}

    for col in categorical_cols:
        if col in train_df.columns:
            encoder = TargetEncoder(cols=[col])
            train_df[f'{col}_target_enc'] = encoder.fit_transform(train_df[col], train_df['price'])
            test_df[f'{col}_target_enc'] = encoder.transform(test_df[col])
            categorical_encoders[col] = encoder

            # Add encoded column to selected features
            selected_features.append(f'{col}_target_enc')

    # Final check for any remaining NaN values
    for col in selected_features:
        if col in train_df.columns and train_df[col].isnull().any():
            train_df[col] = train_df[col].fillna(0)
        if col in test_df.columns and test_df[col].isnull().any():
            test_df[col] = test_df[col].fillna(0)

    # Log transformation parameters
    mlflow.log_param("transformation_features", ", ".join(selected_features))
    mlflow.log_param("test_size_fraction", test_size)
    mlflow.log_param("log_transform_target", "True")

    # Return data, selected features, and encoders
    return {
        "train": train_df,
        "test": test_df,
        "selected_features": selected_features,
        "categorical_cols": [col for col in categorical_cols if col in df_clean.columns],
        "categorical_encoders": categorical_encoders,
        "log_transform": True
    }