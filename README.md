# Startup-Profit-Estimation

# 🚀 50 Startups Profit Estimation using Machine Learning

This project aims to predict the profit of 50 startups based on their spending in R&D, Administration, Marketing, and their geographical location using **Linear Regression**. The implementation is done in Python using **Scikit-learn**, and includes preprocessing, model training, evaluation, and visualization.

---

## 📁 Dataset

- **Source**: Kaggle / Local Archive
- **Features**:
  - `R&D Spend`: Amount spent on research and development
  - `Administration`: Admin-related expenses
  - `Marketing Spend`: Marketing budget
  - `State`: Location of the startup (categorical)
  - `Profit`: Target variable (to be predicted)

---

## 🧠 Objective

To build a regression model that predicts the **profit** of startups using their spending patterns and location.

---

## 🛠️ Tools & Libraries

- **Python**
- **Pandas**, **NumPy** – Data handling
- **Matplotlib**, **Seaborn** – Visualization
- **Scikit-learn** – Preprocessing, Model training, Evaluation

---

## 🔍 Project Workflow

1. **Data Loading**
2. **Exploratory Data Analysis (EDA)**  
   - Box plots for feature distribution  
   - Correlation checks
3. **Preprocessing**
   - OneHotEncoding for categorical column (`State`)
   - Train-Test Split (80-20)
4. **Model Building**
   - Linear Regression using a Scikit-learn Pipeline
5. **Model Evaluation**
   - Mean Squared Error (MSE)
   - R² Score
6. **Visualization**
   - Scatter Plot: Actual vs Predicted Profit
   - Box Plot: Feature distribution

---

## 📊 Results

- **Mean Squared Error**: _(e.g., 8855485.73)_
- **R² Score**: _(e.g., 0.92)_

> The model demonstrates strong predictive performance, with R&D Spend having the highest influence on the outcome.

---
