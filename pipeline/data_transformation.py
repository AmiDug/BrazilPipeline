import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import mlflow


def data_transformation(df, test_size=0.2, random_state=42):

    print("Starting data transformation...")

    # Make a copy to avoid modifying the original

    df_transformed = df.copy()

    # ---- 1. DATA TYPE CONVERSION ----
    # Convert specific columns to appropriate types
    # Example: Convert 'category' column to categorical type
    if 'category' in df_transformed.columns:
        df_transformed['category'] = df_transformed['category'].astype('category')

    # ---- 2. MISSING VALUE HANDLING ----
    # Log missing value counts before handling
    missing_before = df_transformed.isnull().sum().sum()
    mlflow.log_metric("missing_values_before", missing_before)

    # For numeric columns, impute with median
    numeric_cols = df_transformed.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        numeric_imputer = SimpleImputer(strategy='median')
        df_transformed[numeric_cols] = numeric_imputer.fit_transform(df_transformed[numeric_cols])

    # For categorical columns, impute with most frequent
    cat_cols = df_transformed.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        for col in cat_cols:
            if df_transformed[col].isnull().any():
                most_frequent = df_transformed[col].mode()[0]
                df_transformed[col].fillna(most_frequent, inplace=True)

    # Log missing values after handling
    missing_after = df_transformed.isnull().sum().sum()
    mlflow.log_metric("missing_values_after", missing_after)

    # ---- 3. FEATURE ENGINEERING ----
    # Example: Create ratio features for price data
    if all(col in df_transformed.columns for col in ['price', 'rate']):
        # Value-for-rating ratio (price efficiency)
        df_transformed['price_per_rating'] = df_transformed['price'] / df_transformed['rate'].replace(0, 0.1)

    if all(col in df_transformed.columns for col in ['price', 'count']):
        # Price per count (bulk pricing indicator)
        df_transformed['price_per_count'] = df_transformed['price'] / df_transformed['count'].replace(0, 1)

    # ---- 4. HANDLING MULTICOLLINEARITY ----
    # For a simple version, we'll just track correlation
    if len(numeric_cols) > 1:
        corr_matrix = df_transformed[numeric_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [(i, j, corr_matrix.loc[i, j]) for i in upper_tri.index
                     for j in upper_tri.columns if upper_tri.loc[i, j] > 0.8]

        if high_corr:
            # Log high correlations for review
            for i, j, corr in high_corr:
                mlflow.log_param(f"high_corr_{i}_{j}", f"{corr:.2f}")

    # ---- 5. DATA SPLIT ----
    # Create train-test split
    df_train, df_test = train_test_split(
        df_transformed, test_size=test_size, random_state=random_state
    )

    # Log transformation metrics
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("train_samples", len(df_train))
    mlflow.log_param("test_samples", len(df_test))
    mlflow.log_param("features_count", df_transformed.shape[1])
    mlflow.log_param("engineered_features", df_transformed.shape[1] - df.shape[1])

    print(f"Data transformation complete. Train size: {len(df_train)}, Test size: {len(df_test)}")
    print(f"Features: {df_transformed.shape[1]} (added {df_transformed.shape[1] - df.shape[1]} new)")

    return {
        "train": df_train,
        "test": df_test,
        "feature_names": list(df_transformed.columns)
    }