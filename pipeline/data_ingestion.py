import pandas as pd
from sqlalchemy import create_engine
import mlflow
import os
import random


def data_ingestion():

    # Load data from database
    engine = create_engine('sqlite:///../instance/Store.sqlite3?mode=ro')
    df = pd.read_sql_query("SELECT * FROM store", engine)

    # Log sample of original data
    sample_df = df.head()
    temp_path = "df_head.csv"
    sample_df.to_csv(temp_path, index=False)
    mlflow.log_artifact(temp_path)
    os.remove(temp_path)

    # Log original data size
    mlflow.log_param("original_row_count", len(df))

    # Generate synthetic data (always create 400 synthetic items)
    df = generate_synthetic_data(df)

    # Log final data size
    mlflow.log_param("final_row_count", len(df))

    return df


def generate_synthetic_data(df, total_synthetic=400):

    print(f"Generating {total_synthetic} synthetic store items...")

    # List to store all rows (original + synthetic)
    all_rows = []

    # First, add all original rows
    all_rows.extend(df.to_dict('records'))

    # Get the highest ID
    max_id = df['id'].max() if not df.empty else 0

    # Title modifiers for variations
    title_modifiers = ["Premium", "Deluxe", "Standard", "Basic", "Pro",
                       "Elite", "Value", "Essential", "Advanced", "Classic"]
    colors = ["Red", "Blue", "Green", "Black", "White", "Gray", "Purple",
              "Yellow", "Orange", "Pink"]
    size_variants = ["Small", "Medium", "Large", "XL", "XXL", "Mini", "Maxi", "Compact"]

    # Create synthetic variations
    for i in range(total_synthetic):
        # Select a random original row to base this synthetic item on
        original_row = df.iloc[random.randint(0, len(df) - 1)].copy()

        # Create a new row based on the original
        new_row = dict(original_row)

        # Assign a new unique ID
        max_id += 1
        new_row['id'] = max_id

        # Modify the title with random variations
        original_title = new_row['title']
        mod_strategy = random.choice([1, 2, 3])

        if mod_strategy == 1:
            # Add a modifier
            modifier = random.choice(title_modifiers)
            new_row['title'] = f"{modifier} {original_title}"
        elif mod_strategy == 2:
            # Add a color
            color = random.choice(colors)
            new_row['title'] = f"{color} {original_title}"
        else:
            # Add a size/variant indicator
            size = random.choice(size_variants)
            new_row['title'] = f"{original_title} - {size} Version"

        # Modify the price (±30%)
        price_variation = random.uniform(0.7, 1.3)
        new_row['price'] = round(float(new_row['price']) * price_variation, 2)

        # Modify the rating (±1.0 but keep within 1-5 range)
        rating_variation = random.uniform(-1.0, 1.0)
        new_row['rate'] = max(1.0, min(5.0, float(new_row['rate']) + rating_variation))
        new_row['rate'] = round(new_row['rate'], 1)

        # Modify the count (±50%)
        count_variation = random.uniform(0.5, 1.5)
        new_row['count'] = round(float(new_row['count']) * count_variation)

        # Add to the collection
        all_rows.append(new_row)

    # Create a new DataFrame with all rows
    expanded_df = pd.DataFrame(all_rows)

    # Ensure all columns have the original types
    for col in df.columns:
        expanded_df[col] = expanded_df[col].astype(df[col].dtype)

    print(f"Dataset expanded from {len(df)} to {len(expanded_df)} rows")
    return expanded_df