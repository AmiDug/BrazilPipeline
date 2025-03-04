import pandas as pd
import numpy as np
import os
import mlflow
from kaggle.api.kaggle_api_extended import KaggleApi
from datetime import datetime


def data_ingestion():
    """
    Download from Kaggle API and prepare Olist dataset focusing on key features
    """
    data_dir = "./olist_data"

    # Check if data directory exists
    if not os.path.exists(data_dir):
        print("Downloading Olist E-commerce dataset from Kaggle...")
        try:
            # Initialize Kaggle API and authenticate
            api = KaggleApi()
            api.authenticate()

            # Download the dataset
            api.dataset_download_files(
                'olistbr/brazilian-ecommerce',
                path=data_dir,
                unzip=True
            )
            print("Dataset downloaded successfully.")
        except Exception as e:
            print(f"Error downloading from Kaggle API: {e}")
            print("Please ensure you have set up your Kaggle API credentials.")
            return None

    try:
        # Load core datasets
        order_items_df = pd.read_csv(f"{data_dir}/olist_order_items_dataset.csv")
        products_df = pd.read_csv(f"{data_dir}/olist_products_dataset.csv")
        orders_df = pd.read_csv(f"{data_dir}/olist_orders_dataset.csv")
        category_df = pd.read_csv(f"{data_dir}/product_category_name_translation.csv")
        customers_df = pd.read_csv(f"{data_dir}/olist_customers_dataset.csv")
        sellers_df = pd.read_csv(f"{data_dir}/olist_sellers_dataset.csv")
        payments_df = pd.read_csv(f"{data_dir}/olist_order_payments_dataset.csv")
        reviews_df = pd.read_csv(f"{data_dir}/olist_order_reviews_dataset.csv")

        print(f"Loaded datasets - orders: {len(orders_df)}, products: {len(products_df)}")

        # Join customers with orders to get geographic data
        orders_with_geo = pd.merge(orders_df, customers_df[['customer_id', 'customer_state', 'customer_city']],
                                   on='customer_id', how='left')

        # Join order items with product data
        product_data = pd.merge(
            order_items_df,
            products_df,
            on='product_id',
            how='left'
        )

        # Merge with categories
        product_data = pd.merge(
            product_data,
            category_df,
            on='product_category_name',
            how='left'
        )

        # Add seller info
        product_data = pd.merge(
            product_data,
            sellers_df[['seller_id', 'seller_state', 'seller_city']],
            on='seller_id',
            how='left'
        )

        # Add geographic data
        product_data = pd.merge(
            product_data,
            orders_with_geo[['order_id', 'customer_state', 'customer_city']],
            on='order_id',
            how='left'
        )

        # Add payment info (carefully)
        payment_agg = payments_df.groupby('order_id').agg({
            'payment_installments': 'mean',
            'payment_type': lambda x: x.value_counts().index[0] if len(x) > 0 else 'unknown'
        }).reset_index()

        product_data = pd.merge(
            product_data,
            payment_agg,
            on='order_id',
            how='left'
        )

        # Add review data
        review_agg = reviews_df.groupby('order_id').agg({
            'review_score': 'mean'
        }).reset_index()

        product_data = pd.merge(
            product_data,
            review_agg,
            on='order_id',
            how='left'
        )

        print(f"Final dataset shape: {product_data.shape}")

        # Log info using MLflow
        try:
            mlflow.log_param("dataset_source", "Olist E-commerce Dataset")
            mlflow.log_param("product_count", len(products_df))
            mlflow.log_param("order_count", len(orders_df))
            mlflow.log_param("joined_data_shape", str(product_data.shape))
        except:
            print("Warning: Could not log to MLflow (continuing)")

        return product_data

    except Exception as e:
        print(f"Error processing Olist data: {e}")
        return None