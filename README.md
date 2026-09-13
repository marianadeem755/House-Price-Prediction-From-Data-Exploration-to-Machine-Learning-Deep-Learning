# Advanced House Price Prediction

### From Data Exploration to Machine Learning & Deep Learning

* This is a complete end-to-end house price prediction notebook that covers **data exploration, missing-value handling, exploratory data analysis, outlier
handling, numerical scaling, categorical encoding, machine learning, model comparison, hyperparameter tuning, and a Deep Neural Network approach**.

* The project uses the **House Prices: Advanced Regression Techniques** dataset and predicts the `SalePrice` of residential properties.

## Project Overview

* House price prediction is a regression problem where the goal is to estimate the selling price of a property from its structural, quality, location, and other property-related characteristics.

* This notebook follows a practical machine learning workflow:

* **Dataset → Data Inspection → Missing Values → EDA → Outlier Analysis →
Scaling → Encoding → ML Model Training → Hyperparameter Tuning → Model
Comparison → Deep Learning → Submission**

The notebook explores both machine learning algorithms and a neural-network-based regression model to compare different approaches for the prediction task to get the best results.

## Objectives

The main objectives of this project are:

*  Explore the structure and characteristics of the house price dataset.
*  Understand numerical and categorical features.
*  Identify and analyze missing values.
*  Impute missing values using model-based approaches.
*  Check training and testing data for duplicates.
*  Perform exploratory data analysis on important features.
*  Analyze the distribution of `SalePrice`.
*  Study correlations between selected numerical features.
*  Examine potential outliers.
*   Scale numerical features.
*   Encode categorical features.
*   Train multiple regression algorithms.
*   Tune model hyperparameters using `GridSearchCV`.
*   Compare models using regression metrics.
*   Select the best machine learning model based on RMSE.
*   Build and train a Deep Neural Network for house price prediction.

## Dataset

The notebook uses the **House Prices: Advanced Regression Techniques** competition dataset.

### Dataset files

The notebook reads:

*   `train.csv`: Training data containing property features and the target `SalePrice`.
*   `test.csv`: Test data containing property features without the target.
*   `sample_submission.csv`: sample submission format.

The dataset is loaded from the Kaggle competition input directory:

And it is observed that:

*   **1,460 rows**
*   **81 columns**
*   **38 numerical columns**
*   **48 categorical columns**

The target variable is:

`SalePrice`

## Notebook Workflow

### 1. Dataset Loading

The training, testing, and sample submission files are loaded 
Then explore the First five rows of:

-   Training data
-   Testing data
-   Submission template

### 2. Data Exploration

Then explores: Dataset, Number of rows and columns, Data types, Categorical columns, Numerical columns

## Missing Value Analysis

Missing values are analyzed separately for the training and testing datasets.

It includes:

1.  Calculating missing-value percentages.
2.  Identifing columns containing missing values.
3.  Visualizes missingness using heatmaps.
4.  Separates categorical and numerical missing-value cases.
5.  Uses model-based imputation for missing values.

### Categorical imputation

For categorical features with missing values uses:

-   `LabelEncoder`
-   `IterativeImputer`
-   `RandomForestClassifier`

The available non-null records are used to train a classifier, which is then used to predict missing categorical values.

### Numerical imputation

For numerical missing values uses:

-   `IterativeImputer`
-   `RandomForestRegressor`

## Duplicate Checking

Then in the notebook checks both datasets for duplicate rows.

And it is observed that:

> No duplicates were found in the training and testing data.

# Exploratory Data Analysis

The EDA section investigates both categorical and numerical variables.

## Categorical Feature Exploration

The notebook examines features such as:

-   `Street`
-   `Alley`
-   `LotShape`
-   `Utilities`

Count-based visualizations are used to understand the distribution of categorical values.

# SalePrice Analysis

`SalePrice` is the prediction target.

The notebook uses descriptive statistics and a histogram to study its distribution.

# Correlation Analysis

A correlation heatmap is created for selected numerical variables,
including:

-   `MSSubClass`
-   `LotArea`
-   `OverallQual`
-   `OverallCond`
-   `YearBuilt`
-   `MiscVal`
-   `MoSold`
-   `YrSold`
-   `SalePrice`

The correlation matrix is used to investigate linear relationships between numerical variables and the target.

# Distribution Analysis

The notebook also visualizes distributions using KDE plots and histograms.

Variables explored include:

-   `SalePrice`
-   `MoSold`
-   `YrSold`

These visualizations help inspect the overall distribution and concentration of observations.

# Outlier Analysis

Potential outliers are investigated using box plots.

The numerical variables analyzed include:

-   `LotArea`
-   `OverallQual`
-   `OverallCond`
-   `YearBuilt`
-   `MiscVal`
-   `MoSold`
-   `YrSold`
-   `SalePrice`

# Feature Scaling

Numerical columns are scaled using `MinMaxScaler`.

# Categorical Encoding

Categorical columns are transformed into numerical representations using `LabelEncoder`

# Machine Learning

Multiple regression algorithms comparion occurs such as.

## Models Evaluated

### 1. Ridge Regression

A regularized linear regression approach using an `alpha` hyperparameter.

### 2. Linear Regression

A baseline linear regression model.

### 3. Decision Tree Regressor

A tree-based regression model capable of learning nonlinear relationships.

### 4. Random Forest Regressor

An ensemble of decision trees designed to improve predictive performance and robustness.

### 5. K-Nearest Neighbors Regressor

Predicts house prices based on nearby observations in feature space.

### 6. Support Vector Regressor (SVR)

A support-vector-based regression approach with multiple kernel configurations.

### 7. XGBRegressor

A gradient-boosting implementation from XGBoost.

### 8. Gradient Boosting Regressor

An ensemble boosting algorithm that sequentially improves predictions using weak learners.

# Hyperparameter Tuning

`GridSearchCV` is used to search through predefined hyperparameter combinations.

# Evaluation Metrics

Then evaluates the regression models using metrics such as:

1. Mean Absolute Error (MAE)

2. Mean Squared Error (MSE)

3. Root Mean Squared Error (RMSE)

## Deep Learning Approach

After the machine learning workflow, then explores a **Deep Neural Network (DNN)** for house price prediction.

# Technologies & Libraries

The notebook uses a broad Python data-science stack, including:

-   **Python**
-   **Pandas**: data loading and manipulation
-   **NumPy**: numerical computation
-   **Matplotlib**: visualization
-   **Seaborn**: statistical visualization
-   **Plotly**: interactive visualizations
-   **Scikit-learn**: preprocessing, imputation, model training, tuning, and evaluation
-   **XGBoost**: gradient boosting regression
-   **TensorFlow / Keras**: deep neural network


# End-to-End Pipeline

``` 
House Prices Dataset
        ↓
Load Train / Test / Submission Data
        ↓
Initial Data Inspection
        ↓
Data Types & Feature Analysis
        ↓
Missing Value Analysis
        ↓
Model-Based Missing Value Imputation
        ↓
Duplicate Checking
        ↓
Exploratory Data Analysis
        ↓
Target Distribution Analysis
        ↓
Correlation Analysis
        ↓
Outlier Investigation
        ↓
Numerical Feature Scaling
        ↓
Categorical Encoding
        ↓
Train / Validation Split
        ↓
Multiple ML Regression Models
        ↓
GridSearchCV Hyperparameter Tuning
        ↓
MAE / MSE / RMSE Evaluation
        ↓
Model Comparison
        ↓
KNN Selected as Best ML Model
        ↓
Generate ML Submission
        ↓
Deep Neural Network
        ↓
RMSE Loss + Adam Optimizer
        ↓
Early Stopping
        ↓
Save DNN Model
        ↓
Generate DNN Submission
```

# Generated Outputs

The notebook generates or saves the following model/submission file:

## Summary

* This project demonstrates a complete progression from **raw housing data to predictive modeling**, including machine learning and deep
learning.

* The strongest ML result comes from **K-Nearest Neighbors Regression**, while the Deep Neural Network section provides an additional neural-network-based approach for the same house price prediction problem.
