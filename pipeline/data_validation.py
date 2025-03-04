import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mlflow


def data_validation(df):
    """
    Validate the dataset by checking for missing values, data types, and distributions
    """
    print("Starting data validation...")

    # Check if DataFrame is None or empty
    if df is None or len(df) == 0:
        print("Error: DataFrame is empty or None")
        return False, {"error": "Empty DataFrame"}, None

    # Create a validation results dictionary
    validation_results = {}

    # 1. Check required columns
    required_columns = ['price', 'product_id', 'order_id', 'product_category_name_english',
                        'product_weight_g', 'product_length_cm', 'product_height_cm',
                        'product_width_cm', 'customer_state', 'seller_state']

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"Warning: Missing required columns: {missing_columns}")
        validation_results['missing_columns'] = missing_columns

    # 2. Check for missing values in key columns
    if 'price' not in missing_columns:
        price_nulls = df['price'].isnull().sum()
        if price_nulls > 0:
            print(f"Warning: {price_nulls} missing price values")
            validation_results['price_nulls'] = int(price_nulls)

    # Check categorical columns for nulls
    categorical_cols = ['product_category_name_english', 'customer_state', 'seller_state',
                        'payment_type']
    categorical_nulls = {}

    for col in categorical_cols:
        if col in df.columns:
            nulls = df[col].isnull().sum()
            if nulls > 0:
                categorical_nulls[col] = int(nulls)

    if categorical_nulls:
        print(f"Warning: Missing values in categorical columns: {categorical_nulls}")
        validation_results['categorical_nulls'] = categorical_nulls

    # 3. Check for duplicated products and orders
    duplicate_products = df.duplicated(['product_id', 'order_id']).sum()
    if duplicate_products > 0:
        print(f"Warning: {duplicate_products} duplicated product-order combinations")
        validation_results['duplicate_products'] = int(duplicate_products)

    # 4. Check data types
    if 'price' in df.columns and not pd.api.types.is_numeric_dtype(df['price']):
        print("Warning: Price column is not numeric")
        validation_results['price_type_error'] = True

    # 5. Check for outliers in price
    if 'price' in df.columns:
        q1 = df['price'].quantile(0.25)
        q3 = df['price'].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        price_outliers = (df['price'] > upper_bound).sum()

        if price_outliers > 0:
            print(f"Warning: {price_outliers} price outliers detected")
            validation_results['price_outliers'] = int(price_outliers)
            validation_results['price_upper_bound'] = float(upper_bound)

    # 6. Create visualizations
    try:
        # Price distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df['price'].clip(upper=df['price'].quantile(0.95)), bins=50)
        plt.title('Price Distribution (95th percentile)')
        plt.xlabel('Price')
        plt.tight_layout()

        # Save figure for MLflow
        price_dist_path = "price_distribution.png"
        plt.savefig(price_dist_path)
        try:
            mlflow.log_artifact(price_dist_path)
        except:
            pass
        os.remove(price_dist_path)
        plt.close()

        # Category distribution
        if 'product_category_name_english' in df.columns:
            plt.figure(figsize=(12, 8))
            top_categories = df['product_category_name_english'].value_counts().head(15)
            sns.barplot(y=top_categories.index, x=top_categories.values)
            plt.title('Top 15 Product Categories')
            plt.xlabel('Count')
            plt.tight_layout()

            # Save figure for MLflow
            category_dist_path = "category_distribution.png"
            plt.savefig(category_dist_path)
            try:
                mlflow.log_artifact(category_dist_path)
            except:
                pass
            os.remove(category_dist_path)
            plt.close()

        # State distribution
        if 'customer_state' in df.columns:
            plt.figure(figsize=(12, 6))
            top_states = df['customer_state'].value_counts().head(10)
            sns.barplot(x=top_states.index, y=top_states.values)
            plt.title('Top 10 Customer States')
            plt.ylabel('Count')
            plt.tight_layout()

            # Save figure for MLflow
            state_dist_path = "state_distribution.png"
            plt.savefig(state_dist_path)
            try:
                mlflow.log_artifact(state_dist_path)
            except:
                pass
            os.remove(state_dist_path)
            plt.close()

        # Freight vs. Price scatter
        if all(col in df.columns for col in ['freight_value', 'price']):
            plt.figure(figsize=(10, 6))
            df_sample = df.sample(min(5000, len(df)))
            plt.scatter(df_sample['freight_value'], df_sample['price'], alpha=0.5)
            plt.title('Price vs. Freight Value')
            plt.xlabel('Freight Value')
            plt.ylabel('Price')
            plt.tight_layout()

            # Save figure for MLflow
            freight_price_path = "freight_vs_price.png"
            plt.savefig(freight_price_path)
            try:
                mlflow.log_artifact(freight_price_path)
            except:
                pass
            os.remove(freight_price_path)
            plt.close()

    except Exception as e:
        print(f"Warning: Error creating visualizations: {e}")
        validation_results['visualization_error'] = str(e)

    # 7. Calculate basic statistics for reporting
    try:
        stats = {
            'total_rows': len(df),
            'price_mean': float(df['price'].mean()),
            'price_median': float(df['price'].median()),
            'price_min': float(df['price'].min()),
            'price_max': float(df['price'].max()),
            'total_product_categories': int(df['product_category_name_english'].nunique()),
            'total_customer_states': int(df['customer_state'].nunique()),
            'total_seller_states': int(df['seller_state'].nunique())
        }

        validation_results['stats'] = stats

        print(f"Dataset has {stats['total_rows']} rows")
        print(
            f"Price range: ${stats['price_min']:.2f} - ${stats['price_max']:.2f} (median: ${stats['price_median']:.2f})")
        print(f"Product categories: {stats['total_product_categories']}")
        print(f"Customer states: {stats['total_customer_states']}")
        print(f"Seller states: {stats['total_seller_states']}")

        # Log statistics to MLflow
        try:
            mlflow.log_metric("mean_price", stats['price_mean'])
            mlflow.log_metric("median_price", stats['price_median'])
            mlflow.log_metric("num_categories", stats['total_product_categories'])
        except:
            pass

    except Exception as e:
        print(f"Warning: Error calculating statistics: {e}")
        validation_results['stats_error'] = str(e)

    # 8. Overall validation decision
    validation_passed = not bool(
        missing_columns or
        ('price_nulls' in validation_results and validation_results['price_nulls'] > 0) or
        ('price_type_error' in validation_results)
    )

    if validation_passed:
        print("Data validation passed")
    else:
        print("Data validation failed with warnings (continuing pipeline)")

    return validation_passed, validation_results, df