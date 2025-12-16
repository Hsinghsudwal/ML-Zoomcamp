
<h1 align="center">
    <strong>Machine Learning Zoomcamp: 2025</strong>
</h1>

<p align="center">
Learn machine learning engineering from regression and classification to deployment and deep learning.
</p>

## Table of Contents
- [About ML Zoomcamp](#about-ml-zoomcamp)
- [Modules](#modules)
- [Capstone Project](#projects)
- [Certificates](#certificates)

## About ML Zoomcamp

Build and deploy machine learning systems. Focus on training the models to get them to work in production.

- Core ML algorithms and when to use them
- Scikit-Learn and XGBoost
- Preparing data, feature engineering
- Model evaluation and selection
- Deploying models with FastAPI, uv, Docker, and cloud platforms
- Using Kubernetes for ML model serving
- Deep learning with PyTorch and TensorFlow

## Modules

[**1. Introduction to Machine Learning**](01-intro/)

Learn the fundamentals: what ML is, when to use it, and how to approach ML problems using the CRISP-DM framework.

**Topics:**
- ML vs rule-based systems
- Supervised learning basics
- CRISP-DM methodology
- Model selection concepts
- Environment setup

* [Homework.md](01-intro/homework.md)
* [Homework.ipynb](01-intro/homework_01.ipynb)


[**Module 2: Machine Learning for Regression**](02-regression/)

Build a car price prediction model while learning linear regression, feature engineering, and regularization.

**Topics:**
- Linear regression (from scratch and with scikit-learn)
- Exploratory data analysis
- Feature engineering
- Regularization techniques
- Model validation

* [Homework.md](02-regression/homework.md)
* [Homework.ipynb](02-regression/homework_02.ipynb)

[**Module 3: Machine Learning for Classification**](03-classification/)

Create a customer churn prediction system using logistic regression and learn about feature selection.

**Topics:**
- Logistic regression
- Feature importance and selection
- Categorical variable encoding
- Model interpretation

* [Homework.md](03-classification/homework.md)
* [Homework.ipynb](03-classification/homework_03.ipynb)

[**Module 4: Evaluation Metrics for Classification**](04-evaluation/)

Learn how to properly evaluate classification models and handle imbalanced datasets.

**Topics:**
- Accuracy, precision, recall, F1-score
- ROC curves and AUC
- Cross-validation
- Confusion matrices
- Class imbalance handling

* [Homework.md](04-evaluation/homework.md)
* [Homework.ipynb](04-evaluation/homework_04.ipynb)

[**Module 5: Deploying Machine Learning Models**](05-deployment/)

Turn your models into web services and deploy them with Docker and cloud platforms.

**Topics:**

- Model serialization with Pickle
- FastAPI web services
- Docker containerization
- Cloud deployment

* [Homework.md](05-deployment/homework.md)
* [Homework.ipynb](05-deployment/homework_05.ipynb)

[**Module 6: Decision Trees & Ensemble Learning**](06-trees/)

Learn tree-based models and ensemble methods for better predictions.

**Topics:**
- Data cleaning and preparation
- Decision trees algorithm and parameter tuning
- Ensemble learning - Random Forest
- Gradient boosting (XGBoost) - parameter tuning
- Hyperparameter tuning
- Feature importance

* [Homework.md](06-trees/homework.md)
* [Homework.ipynb](06-trees/homework_06.ipynb)

[**Midterm Project**](https://github.com/Hsinghsudwal/ml_hospital_readmission)

Predict hospital readmissions within 30 days using machine learning to identify high-risk patients and improve care outcomes.

**Topics:**
- Project Overview
- Problem Statement and impact
- Dataset: Loading, Validating, EDA
    - Data Preprocessing: normalization, imputation, encoding, imbalance
    - Feature Engineering: date-time, flags, age.

- Pipeline:
    - Train/validation/test split
    - Modeling: Logistic Regression, Random Forest, XGBoost / LightGBM
    - Cross-validation
    - Hyperparameter tuning
    - Evaluation Metrics: with healthcare+risk prediction, include:
        - AUROC
        - Precision/Recall
        - F1-Score
        - Sensitivity (Recall) & Specificity
        - Confusion matrix
        - Calibration curve

- Deployment: model is served:
    - FastAPI / Flask
    - Docker container

- API Endpoints
- How to Run the Project
    - Setup environment
    - Install dependencies
    - Training instructions
    - Inference usage


[**Module 8: Neural Networks & Deep Learning**](08-deep-learning/)	

Introduction to neural networks using TensorFlow and Keras, including CNNs and transfer learning.	

**Topics:**
- Neural network fundamentals
- PyTorch
- TensorFlow & Keras
- Convolutional Neural Networks
- Transfer learning
- Model optimization

* [Homework.md](08-deep-learning/homework.md)
* [Homework.ipynb](08-deep-learning/homework_08.ipynb)


[**Module 9: Serverless**](09-serverless/)

Deep Learning: Deploy deep learning models using serverless technologies like AWS Lambda.

**Topics:**

- Serverless concepts
- Deploying Scikit-Learn models with AWS Lambda
- Deploying TensorFlow and PyTorch models with AWS Lambda
- API Gateway

* [Homework.md](09-serverless/homework.md)
* [Homework.ipynb](09-serverless/homework_09.ipynb)

[**Module 10: Kubernetes**](10-kubernetes/)

Learn to serve ML models at scale using Kubernetes and TensorFlow Serving.	

**Topics:**
- Kubernetes basics
- TensorFlow Serving
- Model deployment and scaling
- Load balancing

* [Homework.md](10-kubernetes/homework.md)

## Projects

[Capstone Project](projects/)

1. **Complete 2 out of 3 projects**:

    Choose a problem, find a dataset, and develop model. Deploy model into a web service (local deployment or cloud deployment for bonus points).

    - **Midterm Project**: Choose a problem, find a dataset, and develop model
    - **Capstone Project**: includes deploying a model as a web service

2. **Review 3 peers' projects** by the deadline





