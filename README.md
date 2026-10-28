# 💎 Diamond Dynamics: Price Prediction and Market Segmentation

> An end-to-end Machine Learning project that predicts diamond prices and segments diamonds into meaningful market categories using Regression, Artificial Neural Networks (ANN), K-Means Clustering, PCA, and Streamlit.

---

## 📌 Project Overview

The diamond market is driven by multiple quality attributes such as **carat, cut, clarity, color, and dimensions**. Accurate price prediction helps retailers optimize pricing strategies, while market segmentation enables efficient inventory management and personalized customer recommendations.

This project builds an intelligent system that:

- 💰 Predicts diamond prices using Machine Learning and ANN models
- 💎 Segments diamonds into market categories using clustering
- 📊 Performs detailed Exploratory Data Analysis (EDA)
- 🚀 Deploys an interactive Streamlit application for real-time predictions

---

# 🚀 Skills Demonstrated

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Data Visualization
- Feature Engineering
- Outlier Detection & Treatment
- Skewness Handling
- Feature Selection
- Regression Modeling
- Artificial Neural Networks (ANN)
- K-Means Clustering
- Principal Component Analysis (PCA)
- Model Evaluation
- Streamlit Deployment

---

# 🏢 Domain

- E-Commerce
- Luxury Goods Analytics
- Retail Pricing Optimization

---

# 📌 Problem Statement

Diamond pricing depends on several physical and quality-related characteristics including:

- Carat
- Cut
- Color
- Clarity
- Dimensions

The objective is to build an intelligent system capable of:

- Predicting the market price of diamonds.
- Grouping diamonds into meaningful market segments.
- Providing an interactive web application for real-time predictions.

---

# 🎯 Objectives

## Price Prediction

- Build multiple Machine Learning Regression models
- Build an Artificial Neural Network (ANN)
- Compare model performances
- Save the best-performing model

## Market Segmentation

- Cluster diamonds into meaningful market groups
- Visualize clusters using PCA
- Assign business-friendly cluster names

## Deployment

Develop a Streamlit application that:

- Predicts Diamond Price
- Predicts Market Segment
- Displays user-friendly results

---

# 🌍 Real-World Applications

- Dynamic Pricing Strategy
- Inventory Categorization
- Luxury Product Recommendation Systems
- Customer Segmentation
- Personalized Marketing
- Retail Pricing Optimization

---

# 📂 Dataset

**Source:** Diamond Dataset

**Rows:** 53,940

**Features:** 10

---

# 📊 Dataset Features

| Feature | Description |
|----------|-------------|
| carat | Weight of diamond |
| cut | Cut quality (Fair → Ideal) |
| color | Diamond color grade (D → J) |
| clarity | Inclusion quality |
| depth | Total depth percentage |
| table | Width of top facet (%) |
| price | Diamond price (USD) |
| x | Length (mm) |
| y | Width (mm) |
| z | Depth (mm) |

> **Note:** Price is converted from **USD to INR** during feature engineering.

---

# 🧹 Data Preprocessing

- Handle missing values
- Replace invalid zero values in x, y, z
- Remove duplicates
- Handle inconsistent records
- Drop unnecessary features (if required)

---

# 📦 Outlier Treatment

Outliers are handled using:

- IQR Method
- Z-Score Method

Visualization:

- Boxplots
- Distribution plots

---

# 📈 Skewness Treatment

Highly skewed features are transformed using:

- Log Transformation
- Square Root Transformation
- Box-Cox Transformation (where applicable)

---

# 📊 Exploratory Data Analysis (EDA)

Performed extensive EDA including:

- Distribution plots
- Count plots
- Box plots
- Scatter plots
- Pair plots
- Correlation Heatmap
- Regression plots
- Average price comparison
- Category-wise visualizations

---

# ⚙️ Feature Engineering

Derived additional features including:

- Price (USD → INR)
- Volume = x × y × z
- Price per Carat
- Dimension Ratio
- Carat Category
    - Light
    - Medium
    - Heavy

Additional domain-based engineered features were explored to improve model performance.

---

# 🎯 Feature Selection

Applied feature selection techniques such as:

- Correlation Matrix
- Feature Importance
- Recursive Feature Elimination (RFE)
- Variance Inflation Factor (VIF)

---

# 🔤 Encoding

Categorical variables were encoded using:

- Label Encoding
- Ordinal Encoding
- One-Hot Encoding

(depending on model requirements)

---

# 🤖 Machine Learning Models

## Regression Models

Implemented multiple regression algorithms:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- K-Nearest Neighbors Regressor
- XGBoost Regressor
- Artificial Neural Network (TensorFlow/Keras)

---

## Evaluation Metrics

Models were evaluated using:

- MAE
- MSE
- RMSE
- R² Score

The best-performing model was saved as:

```
best_price_prediction_model.pkl
```

---

# 📈 Market Segmentation

Implemented clustering using:

- K-Means Clustering

(Optional)

- Hierarchical Clustering
- DBSCAN

Optimal clusters determined using:

- Elbow Method
- Silhouette Score

---

# 📉 PCA Visualization

Principal Component Analysis (PCA) was used to:

- Reduce dimensionality
- Visualize clusters in 2D
- Improve interpretability

---

# 🏷 Cluster Labels

Clusters were analyzed and assigned meaningful business names such as:

| Cluster | Meaning |
|----------|----------|
| Premium Heavy Diamonds | High carat & premium pricing |
| Mid-range Balanced Diamonds | Medium size and price |
| Affordable Small Diamonds | Budget-friendly diamonds |

---

# 📱 Streamlit Application

The project includes an interactive Streamlit web application.

## 💰 Price Prediction

User Inputs:

- Carat
- Cut
- Color
- Clarity
- x
- y
- z

Output:

- Predicted Diamond Price (INR)

---

## 💎 Market Segmentation

Using the same inputs, the application predicts:

- Cluster Number
- Market Segment Name

Example:

```
Cluster 2

Premium Heavy Diamonds
```

---

# 🛠 Tech Stack

## Programming

- Python

## Data Processing

- NumPy
- Pandas

## Visualization

- Matplotlib
- Seaborn

## Machine Learning

- Scikit-learn
- XGBoost

## Deep Learning

- TensorFlow
- Keras

## Deployment

- Streamlit

---


# 📊 Expected Outputs

✔ Accurate Diamond Price Prediction

✔ Market Segment Classification

✔ Interactive Dashboard

✔ Business-friendly Cluster Labels

✔ PCA Cluster Visualization

---

# 📌 Project Deliverables

### ✔ Python Notebook

- Data Cleaning
- EDA
- Feature Engineering
- Feature Selection
- Regression Models
- ANN Model
- Clustering
- PCA
- Model Evaluation

---

### ✔ Saved Models

- Regression Model (.pkl)
- Clustering Model (.pkl)
- Encoders
- Scalers

---

### ✔ Streamlit Web App

- Predict Diamond Price
- Predict Market Segment
- Interactive UI
- Real-time Predictions

---

# 🏷 Technical Tags

`Machine Learning` `Regression` `Artificial Neural Network` `Random Forest` `Decision Tree` `XGBoost` `KNN` `Feature Engineering` `Feature Selection` `EDA` `Clustering` `K-Means` `PCA` `Streamlit` `TensorFlow` `Keras` `Scikit-Learn` `Diamond Price Prediction` `Market Segmentation` `Retail Analytics` `Luxury Goods Analytics`

