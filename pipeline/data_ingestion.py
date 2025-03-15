import pandas as pd
import os
import mlflow
from kaggle.api.kaggle_api_extended import KaggleApi


def data_ingestion():
    """Download and prepare Olist dataset focusing on key features"""
    data_dir = "./olist_data"

    # Download data if not exists
    if not os.path.exists(data_dir):
        try:
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files('olistbr/brazilian-ecommerce', path=data_dir, unzip=True)
            print("Dataset downloaded successfully.")
        except Exception as e:
            print(f"Error downloading from Kaggle API: {e}")
            print("Please ensure you have set up your Kaggle API credentials.")
            return None

    try:
        # Load datasets with a dictionary for cleaner access
        dfs = {
            'items': pd.read_csv(f"{data_dir}/olist_order_items_dataset.csv"),
            'products': pd.read_csv(f"{data_dir}/olist_products_dataset.csv"),
            'orders': pd.read_csv(f"{data_dir}/olist_orders_dataset.csv"),
            'categories': pd.read_csv(f"{data_dir}/product_category_name_translation.csv"),
            'customers': pd.read_csv(f"{data_dir}/olist_customers_dataset.csv"),
            'sellers': pd.read_csv(f"{data_dir}/olist_sellers_dataset.csv"),
            'payments': pd.read_csv(f"{data_dir}/olist_order_payments_dataset.csv"),
            'reviews': pd.read_csv(f"{data_dir}/olist_order_reviews_dataset.csv")
        }

        print(f"Loaded datasets - orders: {len(dfs['orders'])}, products: {len(dfs['products'])}")

        # Join orders with customer geographic data
        orders_geo = pd.merge(
            dfs['orders'],
            dfs['customers'][['customer_id', 'customer_state', 'customer_city']],
            on='customer_id', how='left'
        )

        # Build the final dataset through sequential merges
        result = pd.merge(dfs['items'], dfs['products'], on='product_id', how='left')
        result = pd.merge(result, dfs['categories'], on='product_category_name', how='left')
        result = pd.merge(
            result,
            dfs['sellers'][['seller_id', 'seller_state', 'seller_city']],
            on='seller_id', how='left'
        )
        result = pd.merge(
            result,
            orders_geo[['order_id', 'customer_state', 'customer_city']],
            on='order_id', how='left'
        )

        # Add payment information
        payment_agg = dfs['payments'].groupby('order_id').agg({
            'payment_installments': 'mean',
            'payment_type': lambda x: x.value_counts().index[0] if len(x) > 0 else 'unknown'
        }).reset_index()
        result = pd.merge(result, payment_agg, on='order_id', how='left')

        # Add review scores
        review_agg = dfs['reviews'].groupby('order_id').agg({'review_score': 'mean'}).reset_index()
        result = pd.merge(result, review_agg, on='order_id', how='left')

        print(f"Final dataset shape: {result.shape}")

        # Log to MLflow
        try:
            mlflow.log_param("dataset_source", "Olist E-commerce Dataset")
            mlflow.log_param("product_count", len(dfs['products']))
            mlflow.log_param("order_count", len(dfs['orders']))
            mlflow.log_param("joined_data_shape", str(result.shape))
        except:
            print("Warning: Could not log to MLflow (continuing)")

        return result

    except Exception as e:
        print(f"Error processing Olist data: {e}")
        return None