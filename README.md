ML pipeline for price prediction based on this Kaggle dataset: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?select=olist_order_payments_dataset.csv

Uses SciKit-Learn, Tensorflow, Keras and XGBoost models.

Open a local mlflow tracking server from (.venv): MLflow server --backend-store-uri sqlite:///instance/mlflow.db --default-artifact-root ./mlflow-artifacts --host 127.0.0.1 --port 8080

Serve a consumable FastAPI from (.venv): uvicorn api.main:app --host 0.0.0.0 --port 8000

You can run the pipeline from the API using curl commands or by visiting http://localhost:8000/

![category_distribution](https://github.com/user-attachments/assets/254cbd16-6eed-471c-b409-8bb15f698772)
![error_by_price](https://github.com/user-attachments/assets/981d6508-a0b8-4dbf-98ec-3764e2e76878)
![error_distribution](https://github.com/user-attachments/assets/f434442f-b4bb-41a5-8f0a-35781ed7c977)
![feature_importance](https://github.com/user-attachments/assets/ba90f1b3-ac48-4845-b519-297370fe09d3)
![freight_vs_price](https://github.com/user-attachments/assets/84f0e34e-7d08-4696-8e9e-c55b786ef0a6)
![pred_vs_actual](https://github.com/user-attachments/assets/21c0223c-e511-40f0-b250-7258aba472a5)
![price_distribution](https://github.com/user-attachments/assets/de55678a-5b93-4ea5-9eae-7160d5d5f57a)
![state_distribution](https://github.com/user-attachments/assets/6e2908ae-fb0d-4fd8-922d-3679b678e66f)
