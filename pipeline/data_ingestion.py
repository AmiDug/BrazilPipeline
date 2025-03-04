import pandas as pd
import numpy as np
import os
import mlflow
from kaggle.api.kaggle_api_extended import KaggleApi
from datetime import datetime


def data_ingestion():
    """
    Load and process Olist datasets to create product features.
    """
    # Check if data is already downloaded
    data_dir = "./olist_data"
    if not os.path.exists(data_dir):
        print("Downloading Olist E-commerce dataset from Kaggle...")
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
            os.makedirs(data_dir, exist_ok=True)

    try:
        print("Loading Olist datasets...")

        # Load core datasets
        order_items_df = pd.read_csv(f"{data_dir}/olist_order_items_dataset.csv")
        products_df = pd.read_csv(f"{data_dir}/olist_products_dataset.csv")
        orders_df = pd.read_csv(f"{data_dir}/olist_orders_dataset.csv")
        reviews_df = pd.read_csv(f"{data_dir}/olist_order_reviews_dataset.csv")
        category_df = pd.read_csv(f"{data_dir}/product_category_name_translation.csv")

        print(f"Total orders: {order_items_df['order_id'].nunique()}")
        print(f"Total products: {order_items_df['product_id'].nunique()}")

        # Process dates
        order_items_df['shipping_limit_date'] = pd.to_datetime(order_items_df['shipping_limit_date'])
        order_items_df['shipping_month'] = order_items_df['shipping_limit_date'].dt.month
        order_items_df['shipping_weekday'] = order_items_df['shipping_limit_date'].dt.weekday

        # Join product data with categories
        products_with_categories = pd.merge(
            products_df,
            category_df,
            on='product_category_name',
            how='left'
        )

        # Join with reviews to get product ratings
        reviews_agg = reviews_df.groupby('order_id').agg({
            'review_score': 'mean'
        }).reset_index()

        orders_with_reviews = pd.merge(orders_df, reviews_agg, on='order_id', how='left')

        # Join order items with products data
        product_data = pd.merge(
            order_items_df,
            products_with_categories,
            on='product_id',
            how='left'
        )

        # Aggregate features at the product level
        product_features = product_data.groupby('product_id').agg({
            # Price statistics
            'price': ['mean', 'std', 'min', 'max', 'count'],
            # Freight value statistics
            'freight_value': ['mean', 'std', 'min', 'max'],
            # Product characteristics
            'product_weight_g': ['first'],
            'product_length_cm': ['first'],
            'product_height_cm': ['first'],
            'product_width_cm': ['first'],
            # Order position statistics
            'order_item_id': ['mean'],
            # Unique seller count
            'seller_id': pd.Series.nunique,
            # Category name
            'product_category_name_english': ['first'],
            # Shipping date statistics
            'shipping_month': ['mean'],
            'shipping_weekday': ['mean']
        })

        # Flatten the MultiIndex columns
        product_features.columns = ['_'.join(col).strip() for col in product_features.columns.values]
        product_features = product_features.reset_index()

        # Rename columns for clarity
        product_features.rename(columns={
            'product_id': 'id',
            'price_mean': 'price',
            'price_count': 'order_count',
            'price_std': 'price_variance',
            'seller_id_nunique': 'unique_seller_count',
            'order_item_id_mean': 'avg_position_in_order',
            'product_category_name_english_first': 'category'
        }, inplace=True)

        # Rename physical attribute columns
        for attr in ['weight_g', 'length_cm', 'height_cm', 'width_cm']:
            old_col = f'product_{attr}_first'
            new_col = f'product_{attr}'
            if old_col in product_features.columns:
                product_features.rename(columns={old_col: new_col}, inplace=True)

        # Calculate volume
        if all(col in product_features.columns for col in
               ['product_length_cm', 'product_height_cm', 'product_width_cm']):
            product_features['volume_cm3'] = (
                    product_features['product_length_cm'] *
                    product_features['product_height_cm'] *
                    product_features['product_width_cm']
            )

        # Calculate additional features
        product_features['price_to_freight_ratio'] = product_features['price'] / product_features[
            'freight_value_mean'].replace(0, 1)

        # Add density if both weight and volume exist
        if 'product_weight_g' in product_features.columns and 'volume_cm3' in product_features.columns:
            product_features['density'] = product_features['product_weight_g'] / product_features['volume_cm3'].replace(
                0, np.nan)
            product_features['price_per_gram'] = product_features['price'] / product_features[
                'product_weight_g'].replace(0, np.nan)

        # Add text-based features
        if 'product_name_lenght' in products_df.columns:
            name_lengths = products_df[['product_id', 'product_name_lenght']].rename(
                columns={'product_name_lenght': 'title_length'})
            product_features = pd.merge(product_features, name_lengths, left_on='id', right_on='product_id', how='left')
            product_features.drop('product_id', axis=1, inplace=True)

        if 'product_description_lenght' in products_df.columns:
            desc_lengths = products_df[['product_id', 'product_description_lenght']].rename(
                columns={'product_description_lenght': 'description_length'})
            product_features = pd.merge(product_features, desc_lengths, left_on='id', right_on='product_id', how='left')
            product_features.drop('product_id', axis=1, inplace=True)

            # Add description-to-price ratio (more expensive items often have shorter descriptions)
            product_features['description_price_ratio'] = product_features['description_length'] / product_features[
                'price'].replace(0, np.nan)

        if 'product_photos_qty' in products_df.columns:
            photo_counts = products_df[['product_id', 'product_photos_qty']].rename(
                columns={'product_photos_qty': 'image_count'})
            product_features = pd.merge(product_features, photo_counts, left_on='id', right_on='product_id', how='left')
            product_features.drop('product_id', axis=1, inplace=True)

        # Add seller reputation features
        if 'seller_id' in order_items_df.columns:
            # Get seller order count
            seller_stats = order_items_df.groupby('seller_id').agg({
                'order_id': 'nunique',
            }).reset_index()
            seller_stats.columns = ['seller_id', 'seller_order_count']

            # Merge with product data
            temp_df = pd.merge(
                product_data[['product_id', 'seller_id']].drop_duplicates(),
                seller_stats,
                on='seller_id'
            )
            product_seller_stats = temp_df.groupby('product_id').agg({
                'seller_order_count': 'mean'
            }).reset_index()

            product_features = pd.merge(product_features, product_seller_stats,
                                        left_on='id', right_on='product_id', how='left')
            if 'product_id' in product_features.columns:
                product_features.drop('product_id', axis=1, inplace=True)

        # Add placeholder columns for validation compatibility
        product_features['title'] = product_features['category'].fillna('Unknown')
        product_features['description'] = "Product description"
        product_features['image'] = "image_url"
        product_features['rate'] = 0  # Default rating
        product_features['count'] = product_features['order_count']
        product_features['freight_value'] = product_features['freight_value_mean']

        # Log-transform price for better distribution
        product_features['log_price'] = np.log1p(product_features['price'])

        # Fill missing values
        for col in product_features.columns:
            if product_features[col].dtype in [np.float64, np.float32]:
                product_features[col] = product_features[col].fillna(0)

        # Ensure order_count is integer type
        product_features['order_count'] = product_features['order_count'].astype(int)

        # Log data statistics
        print(f"Processed {len(product_features)} products from Olist datasets")
        print(f"Price range: ${product_features['price'].min():.2f} - ${product_features['price'].max():.2f}")
        print(f"Average orders per product: {product_features['order_count'].mean():.2f}")

        # Log MLflow parameters
        mlflow.log_param("dataset_source", "Merged Olist Datasets")
        mlflow.log_param("product_count", len(product_features))
        mlflow.log_param("price_range",
                         f"${product_features['price'].min():.2f} - ${product_features['price'].max():.2f}")

        # Save and log sample data
        sample_df = product_features.head(10)
        temp_path = "olist_sample.csv"
        sample_df.to_csv(temp_path, index=False)
        mlflow.log_artifact(temp_path)
        os.remove(temp_path)

        return product_features

    except Exception as e:
        print(f"Error processing Olist data: {e}")
        raise