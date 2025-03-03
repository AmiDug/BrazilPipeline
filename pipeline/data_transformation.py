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

    # Remove outliers from price
    # Extremely high or low prices can distort the model
    q1 = df_clean['price'].quantile(0.01)  # 1st percentile
    q3 = df_clean['price'].quantile(0.99)  # 99th percentile

    df_clean = df_clean[(df_clean['price'] >= q1) & (df_clean['price'] <= q3)]

    # Remove outliers from physical dimensions
    for col in ['product_weight_g', 'volume_cm3']:
        q1 = df_clean[col].quantile(0.01)
        q3 = df_clean[col].quantile(0.99)
        df_clean = df_clean[(df_clean[col] >= q1) & (df_clean[col] <= q3)]

    # Log outlier removal stats
    mlflow.log_metric("rows_after_outlier_removal", len(df_clean))
    mlflow.log_metric("outliers_removed_percent", 100 * (1 - len(df_clean) / len(df)))

    # Handle any remaining missing values
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            # Fill missing with median
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Create category_code column (convert categories to numbers)
    df_clean['category_code'] = pd.Categorical(df_clean['category']).codes

    # Calculate category average price (for potential feature engineering)
    category_avg_price = df_clean.groupby('category')['price'].mean().to_dict()
    df_clean['category_avg_price'] = df_clean['category'].map(category_avg_price)

    # Create price ratio feature (how expensive is this item compared to category average)
    df_clean['price_ratio'] = df_clean['price'] / df_clean['category_avg_price']

    # Log basic transformation metrics
    mlflow.log_param("final_row_count", len(df_clean))
    mlflow.log_param("test_size_fraction", test_size)  # Changed parameter name
    mlflow.log_param("random_state", random_state)

    # Create train-test split
    train_df, test_df = train_test_split(
        df_clean, test_size=test_size, random_state=random_state
    )

    # Log split sizes
    mlflow.log_param("train_samples", len(train_df))  # Changed parameter name
    mlflow.log_param("test_samples", len(test_df))    # Changed parameter name

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
    numeric_cols = [col for col in train_df.select_dtypes(include=['number']).columns
                    if col != 'price' and train_df[col].nunique() > 5]

    # Calculate correlation with price
    price_correlation = {}
    for col in numeric_cols:
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

    return {
        "train": train_df,
        "test": test_df,
        "price_correlations": price_correlation
    }