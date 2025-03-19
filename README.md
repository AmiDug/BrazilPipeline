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

![price_distribution](https://raw.githubusercontent.com/AmiDug/BrazilPipeline/refs/heads/master/documents/price_distribution.png)

There were also 10225 product-order combinations that were duplicates and were therefore removed, duplicate data allows a single data entry have several times the training impact that it should have.

Categorical features such as product_category_name_english, customer_state, seller_state and payment_type had missing values that were imputed with "unknown" values rather than dropping the entire feature,
this preserves the data volume while not letting missing value affect the results.

Categorical features were also converted to numerical represented ones that are more appropriate for mathematical machine models through label encoding.

Most features were determined to be non-predictive and were either merged with other features to create something with predictive power or in most cases simply dropped, 
out of 52 columns the training only happened with 16 features, engineered or otherwise.

An Exploratory Data Analysis or EDA was performed and there were some interesting findings.

Firstly there was significant variation in pricing that was based on products categories, in particular electronics and furniture commanded higher prices than other categories.

![category_distribution](https://raw.githubusercontent.com/AmiDug/BrazilPipeline/refs/heads/master/documents/category_distribution.png)

Geographic analysis revealed that certain regions such as São Paulo and Rio de Janeiro were more represented than others.

![state_distribution](https://raw.githubusercontent.com/AmiDug/BrazilPipeline/refs/heads/master/documents/state_distribution.png)

Feature importance analysis showed that there are significant relationships between price and the volume and weight of the cargo.

![feature_importance](https://raw.githubusercontent.com/AmiDug/BrazilPipeline/refs/heads/master/documents/xgb_feature_importance.png)

Feature engineering was employed to merge features together to create better correlations. An example is the creation of a volume feature that combined the length, height and width of a product. The volume feature turned out to be highly correlated to price.
There were also some features that were clear example of data leaks such as payment amount which will obviously be very highly correlated with the product price, the difference only consisting of shipment costs, these were removed.

Four models were employed, a decision tree, random forest, XGBoost and a neutral network.

![model summary](https://raw.githubusercontent.com/AmiDug/BrazilPipeline/refs/heads/master/documents/model_summary.png)

Decision Tree:

![image](https://github.com/user-attachments/assets/de7a24c1-d32c-4a7e-a8e6-18c9dbc15768)


XGBoost had the best accuracy at 76.8%. This model acquired a result of 58% before tuning because it would take too long to train it properly on a CPU, so computation was switched to a GPU and amount of rounds were increased which gave a better accuracy.
XGBoost's inbuilt cross-validation was also used in order to get several samplings of the data to make sure that the accuracy would be fairly consistent across different parts of the dataset.

[1]Olist. (n.d.). Brazilian E-Commerce Public Dataset by Olist [Data set]. Kaggle. https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
