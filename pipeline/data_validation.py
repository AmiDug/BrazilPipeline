import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mlflow


def data_validation(df):
    """Validate dataset: check missing values, data types, and distributions"""
    if df is None or len(df) == 0:
        print("Error: DataFrame is empty or None")
        return False, {"error": "Empty DataFrame"}, None

    results = {}

    # Check required columns
    req_cols = ['price', 'product_id', 'order_id', 'product_category_name_english',
                'product_weight_g', 'product_length_cm', 'product_height_cm',
                'product_width_cm', 'customer_state', 'seller_state']

    missing_cols = [col for col in req_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        results['missing_columns'] = missing_cols

    # Check price-related issues
    if 'price' in df.columns:
        # Price nulls
        if (price_nulls := df['price'].isnull().sum()) > 0:
            print(f"Warning: {price_nulls} missing price values")
            results['price_nulls'] = int(price_nulls)

        # Price data type
        if not pd.api.types.is_numeric_dtype(df['price']):
            print("Warning: Price column is not numeric")
            results['price_type_error'] = True

        # Price outliers
        q1, q3 = df['price'].quantile([0.25, 0.75])
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        if (outliers := (df['price'] > upper).sum()) > 0:
            print(f"Warning: {outliers} price outliers detected")
            results['price_outliers'] = int(outliers)
            results['price_upper_bound'] = float(upper)

    # Check categorical nulls
    cat_cols = ['product_category_name_english', 'customer_state', 'seller_state', 'payment_type']
    cat_nulls = {col: int(df[col].isnull().sum()) for col in cat_cols
                 if col in df.columns and df[col].isnull().sum() > 0}
    if cat_nulls:
        print(f"Warning: Missing values in categorical columns: {cat_nulls}")
        results['categorical_nulls'] = cat_nulls

    # Check duplicates
    if (dupes := df.duplicated(['product_id', 'order_id']).sum()) > 0:
        print(f"Warning: {dupes} duplicated product-order combinations")
        results['duplicate_products'] = int(dupes)

    # Helper function for visualization
    def save_plot(path):
        try:
            plt.savefig(path)
            try:
                mlflow.log_artifact(path)
            except:
                pass
            os.remove(path)
        except Exception as e:
            results.setdefault('viz_errors', []).append(str(e))
        finally:
            plt.close()

    # Create visualizations
    try:
        # Price distribution
        if 'price' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df['price'].clip(upper=df['price'].quantile(0.95)), bins=50)
            plt.title('Price Distribution (95th percentile)')
            plt.xlabel('Price')
            plt.tight_layout()
            save_plot("price_distribution.png")

        # Category distribution
        if 'product_category_name_english' in df.columns:
            plt.figure(figsize=(12, 8))
            top_cats = df['product_category_name_english'].value_counts().head(15)
            sns.barplot(y=top_cats.index, x=top_cats.values)
            plt.title('Top 15 Product Categories')
            plt.xlabel('Count')
            plt.tight_layout()
            save_plot("category_distribution.png")

        # State distribution
        if 'customer_state' in df.columns:
            plt.figure(figsize=(12, 6))
            top_states = df['customer_state'].value_counts().head(10)
            sns.barplot(x=top_states.index, y=top_states.values)
            plt.title('Top 10 Customer States')
            plt.ylabel('Count')
            plt.tight_layout()
            save_plot("state_distribution.png")

        # Freight vs. Price scatter
        if all(col in df.columns for col in ['freight_value', 'price']):
            plt.figure(figsize=(10, 6))
            sample = df.sample(min(5000, len(df)))
            plt.scatter(sample['freight_value'], sample['price'], alpha=0.5)
            plt.title('Price vs. Freight Value')
            plt.xlabel('Freight Value');
            plt.ylabel('Price')
            plt.tight_layout()
            save_plot("freight_vs_price.png")
    except Exception as e:
        print(f"Warning: Visualization error: {e}")
        results['viz_error'] = str(e)

    # Calculate statistics
    try:
        stats = {
            'total_rows': len(df),
            'price_mean': float(df['price'].mean()),
            'price_median': float(df['price'].median()),
            'price_min': float(df['price'].min()),
            'price_max': float(df['price'].max()),
            'total_categories': int(df['product_category_name_english'].nunique()),
            'customer_states': int(df['customer_state'].nunique()),
            'seller_states': int(df['seller_state'].nunique())
        }

        results['stats'] = stats
        print(f"Dataset: {stats['total_rows']} rows | Price: ${stats['price_min']:.2f}-${stats['price_max']:.2f} "
              f"(median: ${stats['price_median']:.2f}) | {stats['total_categories']} categories")

        # Log to MLflow
        try:
            mlflow.log_metrics({
                'mean_price': stats['price_mean'],
                'median_price': stats['price_median'],
                'num_categories': stats['total_categories']
            })
        except:
            pass

    except Exception as e:
        print(f"Warning: Statistics error: {e}")
        results['stats_error'] = str(e)

    # Validation decision
    passed = not bool(
        missing_cols or
        results.get('price_nulls', 0) > 0 or
        'price_type_error' in results
    )

    print(f"Data validation {'passed' if passed else 'failed with warnings (continuing pipeline)'}")

    return passed, results, df