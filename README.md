Description:

GPU-accelerated ML pipeline for price prediction based on this Kaggle dataset: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?select=olist_order_payments_dataset.csv

Uses SciKit-Learn, Tensorflow, Keras and XGBoost models.

Open a local mlflow tracking server from (.venv): MLflow server --backend-store-uri sqlite:///instance/mlflow.db --default-artifact-root ./mlflow-artifacts --host 127.0.0.1 --port 8080

Serve a consumable FastAPI from (.venv): uvicorn api.main:app --host 0.0.0.0 --port 8000

You can run the pipeline from the API using curl commands or by visiting http://localhost:8000/

Report:

The pipeline will use both supervised and deep learning models to train different models so that we can compare their performance on the dataset. 

The datasets consists of around 100k transactions made between the years 2016 and 2018 och the Brazillian E-commerce website Olist[1].

There are many 9 different datasets including for customers, geolocation, order items, order payments, order reviews, orders, products, sellers and product categories.

All 9 datasets will be joined for the purposes of training the model and extracting valuable information regarding which features are useful for predicting the price of products during transactions.

Price prediction is very important for e-commerce websites to make sure they can maximize profits for the products they are selling in a data-informed way, if a bad price is given for a product
it can either lead to a net loss for the company or the price may be set so high that no transaction will be made at all, if the price is too high it may even lead to
claims of price gouging for the retailer which affects reputation and may even lead to legal action against the corporate entity.

According to our pipeline the merged datasets consist of 112650 rows with 52 columns, the disk size is 126.19 MB. 19 of the features were numerical while the rest were categorical.

The data was cleaned through various means. 8427 price outliers were identified and and 211 in the 99.9th percentile were removed, 
extreme outliers can overly bias the data despite making up a small amount of the total amount of entries.

The price distribution following this step can be seen in this image:

![price_distribution]([https://github.com/user-attachments/assets/de55678a-5b93-4ea5-9eae-7160d5d5f57a](https://github.com/AmiDug/BrazilPipeline/blob/master/documents/price_distribution.png))

There were also 10225 product-order combinations that were duplicates and were therefore removed, duplicate data allows a single data entry have several times the training impact that it should have.

Categorical features such as product_category_name_english, customer_state, seller_state and payment_type had missing values that were imputed with "unknown" values rather than dropping the entire feature,
this preserves the data volume while not letting missing value affect the results.

Categorical features were also converted to numerical represented ones that are more appropriate for mathematical machine models through label encoding.

Most features were determined to be non-predictive and were either merged with other features to create something with predictive power or in most cases simply dropped, 
out of 52 columns the training only happened with 16 features, engineered or otherwise.



[1]Olist. (n.d.). Brazilian E-Commerce Public Dataset by Olist [Data set]. Kaggle. https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

![category_distribution](https://github.com/user-attachments/assets/254cbd16-6eed-471c-b409-8bb15f698772)
![error_by_price](https://github.com/user-attachments/assets/981d6508-a0b8-4dbf-98ec-3764e2e76878)
![error_distribution](https://github.com/user-attachments/assets/f434442f-b4bb-41a5-8f0a-35781ed7c977)
![feature_importance](https://github.com/user-attachments/assets/ba90f1b3-ac48-4845-b519-297370fe09d3)
![freight_vs_price](https://github.com/user-attachments/assets/84f0e34e-7d08-4696-8e9e-c55b786ef0a6)
![pred_vs_actual](https://github.com/user-attachments/assets/21c0223c-e511-40f0-b250-7258aba472a5)
![price_distribution](https://github.com/user-attachments/assets/de55678a-5b93-4ea5-9eae-7160d5d5f57a)
![state_distribution](https://github.com/user-attachments/assets/6e2908ae-fb0d-4fd8-922d-3679b678e66f)
