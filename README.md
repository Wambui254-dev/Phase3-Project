# Telecom Customer Churn Prediction Project

## Overview

This project aims to predict customer churn in the telecom industry using machine learning techniques. By identifying customers likely to leave, telecom companies can implement proactive retention strategies, reduce acquisition costs, and maximize customer lifetime value.

## Business Problem

Customer churn is a critical issue for telecom companies due to the high cost of acquiring new customers. Predicting churn enables targeted interventions to retain high-risk customers, improving overall business performance.

## Objectives

1. **Reduce customer churn by identifying high-risk customers:**  
   - I built and evaluated models (Logistic Regression and Decision Tree) to predict which customers are likely to churn.

2. **Improve retention strategies by understanding churn factors:**  
   - I performed exploratory data analysis (EDA) and visualized important features.
   - The decision tree visualization highlights key factors influencing churn.

3. **Minimize customer acquisition costs by maximizing customer lifetime value:**  
   - By accurately predicting churn, your models can help target retention efforts, supporting this objective.

## Dataset

The dataset contains 3,333 customer records with 20 features, including demographic information, service usage, and account details. The target variable is `churn`, indicating whether a customer left the service.

## Workflow

1. **Data Loading & Exploration**
   - Loaded the dataset and performed exploratory data analysis (EDA) to understand feature distributions and relationships.
   - Identified class imbalance in the target variable.

2. **Data Preprocessing**
   - Dropped irrelevant identifier columns.
   - Converted categorical features using one-hot encoding.
   - Scaled numeric features for better model performance.
   - Split the data into training and testing sets with stratification.

3. **Modeling**
   - Built a baseline Logistic Regression model.
   - Tuned a Decision Tree model using GridSearchCV for hyperparameter optimization.

4. **Evaluation**
   - Compared models using accuracy, recall, F1-score, and confusion matrices.
   - The tuned Decision Tree model outperformed Logistic Regression, especially in identifying churned customers.

5. **Visualization**
   - Visualized the top levels of the Decision Tree to interpret key decision points.

## Results

- **Logistic Regression:** Good baseline, but limited in detecting churned customers.
- **Decision Tree (Tuned):** Improved recall and F1-score for churn prediction, providing more balanced and actionable results.

## Conclusion

- Logistic Regression serves as a solid baseline.
- Hyperparameter-tuned Decision Tree significantly improves churn detection.
- Decision tree visualization aids in understanding important churn factors.
- Future enhancements could include ensemble models (e.g., Random Forest, XGBoost) and advanced cross-validation techniques.

## Tools & Technologies
Here are the main tools and technologies used in this project:

- **Python**: Main programming language for data analysis and modeling.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib & Seaborn**: For data visualization.
- **scikit-learn**: For machine learning, including preprocessing, model building (Logistic Regression, Decision Tree), hyperparameter tuning (GridSearchCV), and evaluation.
- **Jupyter Notebook**: For interactive coding, visualization, and documentation.

## **Data Sources and File Format Used**

- **Data Source:**  
  The dataset used in this project is bigml_59c28831336c6604c800002a.csv, which contains telecom customer information for churn prediction.

- **File Format:**  
  **CSV (Comma-Separated Values)** format: Used for storing and processing datasets. I loaded into the project using pandas’ `read_csv` function.

## **Statistical Techniques Used**
- **Exploratory Data Analysis (EDA):**  
  Used summary statistics, value counts, and visualizations (boxplots, countplots, heatmaps) to understand data distribution, relationships, and detect imbalances.

- **Feature Scaling:**  
  Applied StandardScaler to normalize numeric features, ensuring all features contribute equally to the model.

- **One-Hot Encoding:**  
  Converted categorical variables into numeric format using one-hot encoding for compatibility with machine learning algorithms.

- **Train-Test Split:**  
  Split the dataset into training and testing sets to evaluate model performance on unseen data.

- **Logistic Regression:**  
  Used as a baseline statistical classification technique for binary prediction (churn vs. not churn).

- **Decision Tree Classifier:**  
  Built a tree-based model to capture non-linear relationships and feature interactions.

- **Hyperparameter Tuning with GridSearchCV:**  
  Performed cross-validated grid search to find the best model parameters for the Decision Tree.

- **Model Evaluation Metrics:**  
  Used confusion matrix, accuracy, precision, recall, and F1-score to assess and compare model performance.

**Visualization Techniques Used**

- **Countplot:**  
  Used to show the distribution of the target variable (`churn`), helping to visualize class imbalance.

![81bd6300-a434-406b-8c4a-18307c2b3b79](https://github.com/user-attachments/assets/1736fc00-e686-4a1e-8e77-2ce04be7970b)

- **Boxplot:**  
  Used to compare the distribution of numerical features (like total day minutes) across churned and non-churned customers, highlighting differences and outliers.

![afd06962-8622-46d1-8adb-f7c1282ce90a](https://github.com/user-attachments/assets/822bc57e-4d9f-482c-bcf4-36a414a12ad0)

- **Correlation Heatmap:**  
  Visualizes the correlation between numerical features, making it easy to spot relationships and redundant features.

![735dbb36-4f0b-4f71-894b-346578ee9ca6](https://github.com/user-attachments/assets/c92f96a3-b680-4cfe-85c6-3a3b3658c28e)

- **Decision Tree Plot:**  
  Visualizes the structure of the trained Decision Tree model, showing how decisions are made and which features are most important for predicting churn.

![89446710-7dea-4ddc-bd1f-d219e2cc1c7e](https://github.com/user-attachments/assets/4fa855ad-ffae-4659-8ce7-b24d71f89283)

These visualizations help in understanding the data, identifying important features, and interpreting model decisions.

## **Integrated Development Environment (IDE)**
**Jupyter Notebook**: For interactive data analysis and documentation.

## 📂 Repository Structure

Data/ → Folder containing datasets for analysis

README.md → This document outlining project details

index.ipynb → Jupyter Notebook containing the full analysis

presentation.pdf → Presentation summarizing key insights and business recommendations for investors

## Author

>**Yvonne Wambui Karinge**
