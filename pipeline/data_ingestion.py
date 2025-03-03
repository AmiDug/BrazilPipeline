import pandas as pd
import os
import mlflow
from kaggle.api.kaggle_api_extended import KaggleApi
import numpy as np
from datetime import datetime


def data_ingestion():
    # Check if data is already downloaded
    data_dir = "./olist_data"
    if not os.path.exists(data_dir):
        print("Downloading Olist E-commerce dataset from Kaggle...")

        # Authenticate and download (requires kaggle.json in ~/.kaggle/)
        try:
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                'olistbr/brazilian-ecommerce',
                path=data_dir,
                unzip=True
            )
            print("Dataset downloaded successfully.")
        except Exception as e:
            print(f"Error downloading from Kaggle API: {e}")
            print("Using backup method to load data...")
            # If Kaggle authentication fails, you can also download manually
            os.makedirs(data_dir, exist_ok=True)

    # Load the datasets
    try:
        print("Loading Olist order items dataset...")
        order_items_df = pd.read_csv(f"{data_dir}/olist_order_items_dataset.csv")

        print("Available columns:", order_items_df.columns.tolist())
        print("Number of orders:", order_items_df['order_id'].nunique())
        print("Number of products:", order_items_df['product_id'].nunique())

        # Create product aggregation by grouping on product_id
        product_stats = order_items_df.groupby('product_id').agg({
            'price': ['mean', 'count', 'std'],
            'freight_value': ['mean']
        })

        # Flatten MultiIndex columns
        product_stats.columns = ['_'.join(col).strip() for col in product_stats.columns.values]
        product_stats = product_stats.reset_index()

        # Rename columns
        product_stats.rename(columns={
            'price_mean': 'price',
            'price_count': 'count',
            'price_std': 'price_std',
            'freight_value_mean': 'freight_value'
        }, inplace=True)

        # Replace NaN with 0 for price_std (happens when count=1)
        product_stats['price_std'] = product_stats['price_std'].fillna(0)

        # Create synthetic features needed for the model
        # Set product_id as id
        product_stats['id'] = product_stats['product_id']

        # Create synthetic weight and volume - correlated with price
        np.random.seed(42)  # For reproducibility
        product_stats['product_weight_g'] = product_stats['price'] * 2 + np.random.normal(0, 10, len(product_stats))

        # Ensure positive weights
        product_stats['product_weight_g'] = product_stats['product_weight_g'].clip(10, None)

        # Create synthetic volume
        product_stats['volume_cm3'] = product_stats['product_weight_g'] * 0.8 + np.random.normal(0, 20,
                                                                                                 len(product_stats))
        product_stats['volume_cm3'] = product_stats['volume_cm3'].clip(5, None)

        # Create synthetic categories based on price ranges
        price_bins = [0, 50, 100, 200, 500, float('inf')]
        category_names = ['budget', 'economy', 'standard', 'premium', 'luxury']
        product_stats['category'] = pd.cut(product_stats['price'], bins=price_bins, labels=category_names)

        # Convert category to object type (string) to match validation expectations
        product_stats['category'] = product_stats['category'].astype(str)

        # Create density feature
        product_stats['density'] = product_stats['product_weight_g'] / product_stats['volume_cm3']

        # Create price per gram feature
        product_stats['price_per_gram'] = product_stats['price'] / product_stats['product_weight_g']

        # Create synthetic title and description length as float64 (not int)
        product_stats['title_length'] = 20.0 + np.random.normal(0, 5, len(product_stats))
        product_stats['title_length'] = product_stats['title_length'].clip(10, 40).astype(float)

        product_stats['description_length'] = 100.0 + np.random.normal(0, 20, len(product_stats))
        product_stats['description_length'] = product_stats['description_length'].clip(50, 200).astype(float)

        # Create synthetic image count as float64 (not int32)
        product_stats['image_count'] = np.random.randint(1, 6, len(product_stats)).astype(float)

        # Add a synthetic title
        product_stats['title'] = 'Product ' + product_stats['id'].astype(str)

        # Add a synthetic description
        product_stats['description'] = 'Description for product ' + product_stats['id'].astype(str)

        # Add a placeholder image URL
        product_stats['image'] = 'https://example.com/img/' + product_stats['id'].astype(str)

        # Add a derived rating (not in original dataset)
        product_stats['rate'] = 3.5 + np.random.normal(0, 0.5, len(product_stats))
        product_stats['rate'] = product_stats['rate'].clip(1, 5).round(1)

        # Select final columns for our analysis
        keep_columns = [
            'id', 'title', 'price', 'description', 'category', 'image', 'rate', 'count',
            'product_weight_g', 'volume_cm3', 'freight_value', 'title_length',
            'description_length', 'image_count', 'density', 'price_per_gram'
        ]

        # Keep only needed columns
        result_df = product_stats[keep_columns].copy()

        # Drop any rows with missing values in critical columns
        result_df = result_df.dropna(subset=['price', 'product_weight_g', 'volume_cm3'])

        # Ensure all data types match what's expected in validation
        result_df['price'] = result_df['price'].astype(float)
        result_df['count'] = result_df['count'].astype(int)
        result_df['rate'] = result_df['rate'].astype(float)
        result_df['title_length'] = result_df['title_length'].astype(float)
        result_df['description_length'] = result_df['description_length'].astype(float)
        result_df['image_count'] = result_df['image_count'].astype(float)
        result_df['category'] = result_df['category'].astype(object)

        # Log data statistics
        print(f"Loaded {len(result_df)} products from Olist order items dataset")
        print(f"Price range: ${result_df['price'].min():.2f} - ${result_df['price'].max():.2f}")

        # Log MLflow params
        mlflow.log_param("dataset_source", "Olist Order Items")
        mlflow.log_param("row_count", len(result_df))
        mlflow.log_param("price_range", f"${result_df['price'].min():.2f} - ${result_df['price'].max():.2f}")

        # Save and log sample data
        sample_df = result_df.head(10)
        temp_path = "olist_sample.csv"
        sample_df.to_csv(temp_path, index=False)
        mlflow.log_artifact(temp_path)
        os.remove(temp_path)

        return result_df

    except Exception as e:
        print(f"Error processing Olist data: {e}")
        raise