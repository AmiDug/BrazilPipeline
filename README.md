ML pipeline for price prediction based on this dataset: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?select=olist_order_payments_dataset.csv

open a local mlflow tracking server from (.venv): mlflow server --backend-store-uri sqlite:///instance/mlflow.db --default-artifact-root ./mlflow-artifacts --host 127.0.0.1 --port 8080

Analysis by Claude 3.7:

Analysis: Product Price Prediction Results
Performance Assessment
Yes, the results look excellent and represent a significant improvement over previous attempts. Here's why this is a good model:
1. Solid Performance Metrics

Gradient Boosting: 56.85% R² on test data
This perfectly matches the ~57% R² you saw in the reference notebook
The RMSE of 107.32 is reasonable given the price range (0.85-6735.00)

2. No Signs of Overfitting

Validation R² (54.26%) and test R² (56.85%) are very close
The model actually performs slightly better on unseen data, which is ideal
Error distribution looks normally distributed around zero (Image 3)

3. No Evidence of Data Leakage

Key features are all legitimate product attributes:

freight_value (27.49%)
product_description_lenght (15.02%)
product_weight_g (11.17%)
payment_installments (10.20%)


No price-derived features appear in the model
No payment_value in the features (which would have caused leakage)

4. Geographic Features Impact

Your hypothesis about geographic features was correct:

seller_state_freq appears in top features
seller_state_RJ is in the top 15 features
These contribute meaningfully but aren't dominant



5. Realistic Error Patterns

The error by price range chart (Image 2) shows a pattern that makes business sense:

Lower-priced items have higher percentage errors (117% for 0-50 range)
Mid-range prices (100-500) have the best accuracy (~35% MAPE)
This is a normal and expected pattern in price prediction



Additional Insights

Top Categories (Image 1): Bed/bath items and health/beauty products are dominant categories, which could be used for business targeting.
Price vs. Freight (Image 5): Shows a positive correlation but with significant variation, confirming this is a legitimate predictor without being a data leak.
Predicted vs. Actual (Image 6): Shows good alignment along the diagonal, with more variance at higher prices (which is expected with fewer samples).
Price Distribution (Image 7): Very right-skewed distribution, which explains why the model has different error rates across price ranges.

This implementation successfully addresses your concerns about data leakage and overfitting while confirming your hypothesis about geographic data importance. The 56.85% R² is excellent for this type of e-commerce price prediction.

![category_distribution](https://github.com/user-attachments/assets/254cbd16-6eed-471c-b409-8bb15f698772)
![error_by_price](https://github.com/user-attachments/assets/981d6508-a0b8-4dbf-98ec-3764e2e76878)
![error_distribution](https://github.com/user-attachments/assets/f434442f-b4bb-41a5-8f0a-35781ed7c977)
![feature_importance](https://github.com/user-attachments/assets/ba90f1b3-ac48-4845-b519-297370fe09d3)
![freight_vs_price](https://github.com/user-attachments/assets/84f0e34e-7d08-4696-8e9e-c55b786ef0a6)
![pred_vs_actual](https://github.com/user-attachments/assets/21c0223c-e511-40f0-b250-7258aba472a5)
![price_distribution](https://github.com/user-attachments/assets/de55678a-5b93-4ea5-9eae-7160d5d5f57a)
![state_distribution](https://github.com/user-attachments/assets/6e2908ae-fb0d-4fd8-922d-3679b678e66f)




