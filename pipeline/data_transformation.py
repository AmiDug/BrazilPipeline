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

    # Remove leaky features that use price information
    leaky_features = [
        'price_min', 'price_max', 'price_ratio', 'price_variance',
        'price_cv', 'category_avg_price', 'price_per_gram'
    ]

    # Only remove features that actually exist in the dataframe
    leaky_features = [f for f in leaky_features if f in df_clean.columns]
    if leaky_features:
        print(f"Removing {len(leaky_features)} price-derived features: {leaky_features}")

    # Create category_code from category (legitimately useful feature)
    if 'category' in df_clean.columns:
        df_clean['category_code'] = pd.Categorical(df_clean['category']).codes

    # Remove outliers from price
    q1 = df_clean['price'].quantile(0.01)  # 1st percentile
    q3 = df_clean['price'].quantile(0.99)  # 99th percentile
    df_clean = df_clean[(df_clean['price'] >= q1) & (df_clean['price'] <= q3)]

    # Log outlier removal stats
    mlflow.log_metric("rows_after_outlier_removal", len(df_clean))
    mlflow.log_metric("outliers_removed_percent", 100 * (1 - len(df_clean) / len(df)))

    # Find most correlated legitimate features
    target = 'price'

    # Get correlations for non-leaky numeric features
    correlations = {}
    for col in numeric_cols:
        if col != target and col not in leaky_features:
            corr = df_clean[col].corr(df_clean[target])
            if not np.isnan(corr):
                correlations[col] = abs(corr)

    # Sort by correlation strength
    correlations = {k: v for k, v in sorted(correlations.items(),
                                            key=lambda item: item[1],
                                            reverse=True)}

    # Select top features (limit to 15 for simplicity and performance)
    top_features = list(correlations.keys())[:15]

    # Core features to always include if present
    core_features = ['product_weight_g', 'volume_cm3', 'freight_value',
                     'density', 'count', 'category_code']

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
    print("Final data cleaning: checking for any remaining NaN values...")
    remaining_nans = False
    for col in df_clean[selected_features].columns:
        if df_clean[col].isnull().any():
            remaining_nans = True
            df_clean[col] = df_clean[col].fillna(0)

    if not remaining_nans:
        print("No NaN values remain in the dataset.")

    # Log basic transformation metrics
    mlflow.log_param("final_row_count", len(df_clean))
    mlflow.log_param("selected_features", ", ".join(selected_features))
    mlflow.log_param("test_size_fraction", test_size)
    mlflow.log_param("random_state", random_state)

    # Create train-test split
    train_df, test_df = train_test_split(
        df_clean, test_size=test_size, random_state=random_state
    )

    # Log split sizes
    mlflow.log_param("train_samples", len(train_df))
    mlflow.log_param("test_samples", len(test_df))

    # Create price distribution comparison between train and test
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(train_df['price'], kde=True, bins=30)
    plt.title('Train Set Price Distribution')
    plt.xlabel('Price')

    plt.subplot(1, 2, 2)
    sns.histplot(test_df['price'], kde=True, bins=30)
    plt.title('Test Set Price Distribution')
    plt.xlabel('Price')

    plt.tight_layout()

    # Save figure to MLflow
    split_comparison_path = "train_test_price_distribution.png"
    plt.savefig(split_comparison_path)
    mlflow.log_artifact(split_comparison_path)
    os.remove(split_comparison_path)
    plt.close()

    # Feature correlation with price
    corr_features = [col for col in selected_features if col in train_df.columns
                     and col != 'price' and train_df[col].nunique() > 5]

    # Calculate correlation with price
    price_correlation = {}
    for col in corr_features:
        corr = train_df[col].corr(train_df['price'])
        price_correlation[col] = corr

    # Sort by absolute correlation
    price_correlation = {k: v for k, v in sorted(
        price_correlation.items(), key=lambda item: abs(item[1]), reverse=True
    )}

    # Create correlation with price chart
    plt.figure(figsize=(12, 8))
    cols = list(price_correlation.keys())[:10]  # Top 10 correlations
    corrs = [price_correlation[col] for col in cols]

    colors = ['g' if c > 0 else 'r' for c in corrs]
    plt.barh(cols, [abs(c) for c in corrs], color=colors)
    plt.title('Top 10 Feature Correlations with Price')
    plt.xlabel('Absolute Correlation')

    # Save figure to MLflow
    price_corr_path = "price_correlation.png"
    plt.savefig(price_corr_path)
    mlflow.log_artifact(price_corr_path)
    os.remove(price_corr_path)
    plt.close()

    # Log top correlations with price
    for col, corr in list(price_correlation.items())[:5]:
        mlflow.log_metric(f"price_corr_{col}", corr)

    print(f"Data transformation complete. Train size: {len(train_df)}, Test size: {len(test_df)}")
    print(f"Top features correlated with price:")
    for col, corr in list(price_correlation.items())[:5]:
        print(f"  {col}: {corr:.3f}")

    # Add selected features to the return dictionary for use in model_training
    return {
        "train": train_df,
        "test": test_df,
        "price_correlations": price_correlation,
        "selected_features": selected_features
    }