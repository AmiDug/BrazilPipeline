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

Metrics:

mean_price
120.65373901464716
median_price
74.99
num_categories
71
best_val_rmse
107.75023559937428
best_val_r2
0.5425742106574566
decision_tree_test_rmse
128.77551088220966
decision_tree_test_r2
0.37870607319366734
decision_tree_test_mape
83.86863241607062
random_forest_test_rmse
112.68717335815334
random_forest_test_r2
0.5242492125848797
random_forest_test_mape
74.19356022904037
gradient_boosting_test_rmse
107.31913800016176
gradient_boosting_test_r2
0.5684959296044287
gradient_boosting_test_mape
66.88991787041637
neural_network_test_rmse
116.25515556123001
neural_network_test_r2
0.4936451331022329
neural_network_test_mape
70.32033684470358
best_test_rmse
107.31913800016176
best_test_r2
0.5684959296044287
best_test_mape
66.88991787041637

test run:

C:\Users\dugie\PycharmProjects\BrazilPipeline\.venv\Scripts\python.exe C:\Users\dugie\PycharmProjects\BrazilPipeline\pipeline\run_pipeline.py 
2025-03-04 23:24:34.664058: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-04 23:24:35.663700: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

==================================================
STEP 1: DATA INGESTION
==================================================
Loaded datasets - orders: 99441, products: 32951
Final dataset shape: (112650, 23)

==================================================
STEP 2: DATA VALIDATION
==================================================
Starting data validation...
Warning: Missing values in categorical columns: {'product_category_name_english': 1627, 'payment_type': 3}
Warning: 10225 duplicated product-order combinations
Warning: 8427 price outliers detected
Dataset has 112650 rows
Price range: $0.85 - $6735.00 (median: $74.99)
Product categories: 71
Customer states: 27
Seller states: 23
Data validation passed

==================================================
STEP 3: DATA TRANSFORMATION
==================================================
Starting data transformation...
Handling missing values...
Filling 1627 missing product_category_name_english values with 'unknown'
Filling 3 missing payment_type values with 'unknown'
Creating features...
Encoding categorical features...
Handling outliers...
Removed 211 price outliers
Finalizing features...
Final dataset shape: (112439, 48)
Selected 46 features for modeling
Split data into 89951 training samples and 22488 test samples

==================================================
STEP 4: MODEL TRAINING
==================================================
Starting model training...
Training data: 71961 samples, 46 features
Validation data: 17990 samples
Training Decision Tree model...
Decision Tree - Validation RMSE: 126.86, R²: 0.37

Top 10 Decision Tree feature importances:
  freight_value: 0.3475
  product_description_lenght: 0.1572
  payment_installments: 0.1045
  product_weight_g: 0.0945
  product_category_name_english_watches_gifts: 0.0521
  product_photos_qty: 0.0492
  product_width_cm: 0.0276
  seller_state_freq: 0.0275
  product_height_cm: 0.0239
  product_length_cm: 0.0229
🏃 View run Decision_Tree at: http://127.0.0.1:8080/#/experiments/8/runs/735d31f35b6f4fd8bf7d2326651e5b03
🧪 View experiment at: http://127.0.0.1:8080/#/experiments/8
Training Random Forest model...
Random Forest - Validation RMSE: 110.42, R²: 0.52

Top 10 Random Forest feature importances:
  freight_value: 0.2811
  product_description_lenght: 0.1470
  payment_installments: 0.0974
  product_weight_g: 0.0867
  volume_cm3: 0.0418
  product_category_name_english_watches_gifts: 0.0413
  product_width_cm: 0.0361
  product_length_cm: 0.0343
  product_height_cm: 0.0331
  seller_state_freq: 0.0270
🏃 View run Random_Forest at: http://127.0.0.1:8080/#/experiments/8/runs/bd126d94d7b4460d89028c4b3b285322
🧪 View experiment at: http://127.0.0.1:8080/#/experiments/8
Training Gradient Boosting model...
Gradient Boosting - Validation RMSE: 107.75, R²: 0.54

Top 10 Gradient Boosting feature importances:
  freight_value: 0.2749
  product_description_lenght: 0.1502
  product_weight_g: 0.1117
  payment_installments: 0.1020
  product_category_name_english_watches_gifts: 0.0440
  product_length_cm: 0.0371
  product_height_cm: 0.0362
  volume_cm3: 0.0319
  product_width_cm: 0.0316
  product_photos_qty: 0.0308
🏃 View run Gradient_Boosting at: http://127.0.0.1:8080/#/experiments/8/runs/d93c183a627446ba85fc7e0179f82ca3
🧪 View experiment at: http://127.0.0.1:8080/#/experiments/8
Training Neural Network model...
C:\Users\dugie\PycharmProjects\BrazilPipeline\.venv\Lib\site-packages\keras\src\layers\core\dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
2025-03-04 23:25:19.758953: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
563/563 ━━━━━━━━━━━━━━━━━━━━ 0s 590us/step
Neural Network - Validation RMSE: 117.83, R²: 0.45
🏃 View run Neural_Network at: http://127.0.0.1:8080/#/experiments/8/runs/0472e67edd9a4c529373b5eabba2b1ba
🧪 View experiment at: http://127.0.0.1:8080/#/experiments/8

Best model: gradient_boosting with validation R²: 0.5426

==================================================
STEP 5: MODEL EVALUATION
==================================================
Starting model evaluation...
Decision Tree - Test RMSE: 128.78, R²: 0.3787, MAPE: 83.87%
Random Forest - Test RMSE: 112.69, R²: 0.5242, MAPE: 74.19%
Gradient Boosting - Test RMSE: 107.32, R²: 0.5685, MAPE: 66.89%
703/703 ━━━━━━━━━━━━━━━━━━━━ 0s 491us/step
Neural Network - Test RMSE: 116.26, R²: 0.4936, MAPE: 70.32%

Best model on test data: gradient_boosting with R²: 0.5685

Top 10 feature importances:
  freight_value: 0.2749
  product_description_lenght: 0.1502
  product_weight_g: 0.1117
  payment_installments: 0.1020
  product_category_name_english_watches_gifts: 0.0440
  product_length_cm: 0.0371
  product_height_cm: 0.0362
  volume_cm3: 0.0319
  product_width_cm: 0.0316
  product_photos_qty: 0.0308
C:\Users\dugie\PycharmProjects\BrazilPipeline\pipeline\model_evaluation.py:199: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  bucket_metrics = test_with_pred.groupby('price_bucket').agg({

Error by price range:
  0-50: MAPE: 117.02%, Count: 7754
  50-100: MAPE: 46.56%, Count: 6685
  100-200: MAPE: 34.31%, Count: 5387
  200-500: MAPE: 35.40%, Count: 1989
  500-1000: MAPE: 43.58%, Count: 518
  1000+: MAPE: 50.26%, Count: 155

==================================================
PIPELINE COMPLETED SUCCESSFULLY
==================================================
Dataset: Olist E-commerce Dataset
Best model: gradient_boosting
RMSE: R$107.32
MAE: R$53.75
R²: 0.5685
MAPE: 66.89%
==================================================
🏃 View run capable-sloth-972 at: http://127.0.0.1:8080/#/experiments/8/runs/2f5b493a92334d6fb549111e8c31df70
🧪 View experiment at: http://127.0.0.1:8080/#/experiments/8

Process finished with exit code 0




