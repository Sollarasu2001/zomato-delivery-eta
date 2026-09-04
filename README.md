# Production-Grade Delivery ETA Prediction

Predict food-delivery duration in minutes using courier, order, traffic, weather, vehicle, location, and timing signals.

## Dataset

Primary source: Zomato Delivery Operations Analytics Dataset.

- Original source: https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset
- Mirror/metadata: https://huggingface.co/datasets/allenborochin/zomato_delivery_EDA
- Target: `Time_taken (min)`
- Original dataset: 45,584 rows x 20 columns.

We intentionally start from the raw dataset. Do not use the cleaned derivative as the training source because the project is designed to implement cleaning and missing-value handling ourselves.

## Environment

Python 3.9.19

## First milestone

1. Acquire raw CSV into `data/raw/`.
2. Run schema/data-quality validation.
3. Produce a reproducible data-quality report.
4. Identify leakage candidates and define the prediction-time feature contract.















ZOMATO_DELIVERY_ETA/
│
├── data/
│   ├── raw/
│   │   └── delivery_data.csv
│   │
│   └── interim/
│       ├── cleaned_data.parquet
│       ├── features.parquet
│       ├── x_train.parquet
│       ├── x_test.parquet
│       ├── y_train.parquet
│       └── y_test.parquet
│
├── models/
│   └── polynomial_ridge.joblib
│
├── reports/
│   └── figures/
│       ├── actual_vs_predicted.png
│       └── residual_distribution.png
│
├── scripts/
│   ├── build_features.py
│   ├── split_data.py
│   ├── ...
│   ├── train_final_model.py
│   ├── verify_model.py
│   ├── evaluate_final_model.py
│   ├── error_analysis.py
│   ├── interpret_final_model.py
│   └── train_with_mlflow.py
│
├── src/
│   └── delivery_eta/
│       │
│       ├── __init__.py
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── app.py
│       │   │
│       │   ├── routes/
│       │   │   ├── __init__.py
│       │   │   ├── health.py
│       │   │   └── prediction.py
│       │   │
│       │   └── services/
│       │       ├── __init__.py
│       │       └── predictor.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   ├── validation.py
│       │   └── cleaning.py
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   ├── engineering.py
│       │   ├── preprocessing.py
│       │   ├── polynomial_preprocessing.py
│       │   ├── cyclic_preprocessing.py
│       │   ├── distance_preprocessing.py
│       │   ├── missing_indicator_preprocessing.py
│       │   └── final_preprocessing.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── linear.py
│       │   ├── ridge.py
│       │   ├── lasso.py
│       │   ├── elastic_net.py
│       │   └── polynomial.py
│       │
│       └── training/
│           ├── __init__.py
│           ├── split.py
│           ├── cross_validation.py
│           └── tuning.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_health.py
│   ├── test_prediction.py
│   └── test_predictor.py
│
├── Dockerfile
├── .dockerignore
├── pytest.ini
├── requirements.txt
├── requirements-dev.txt
└── README.md