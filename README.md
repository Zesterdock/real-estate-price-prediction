
# 🏠 Rental Price Prediction in Pune, Maharashtra (India)

This project aims to predict rental prices of properties in Pune using machine learning techniques. Using structured housing data, we apply regression models like XGBoost and Random Forest to estimate prices based on features such as BHK, area, location, furnishing status, and more.

---

## 📁 Dataset

The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/anantsakhare/rental-price-of-indias-it-capital-pune-mh-ind) and contains information about rental listings in Pune. It includes:
- **Location**
- **Number of BHK**
- **Square Footage**
- **Furnishing Status**
- **Bathroom Count**
- **Balcony Count**
- **Rental Price**

---

## 🛠️ Project Pipeline

The pipeline consists of the following key steps:

1. **Data Preprocessing**  
   - Removing null/duplicate/incorrect values  
   - Normalizing formats and handling outliers  

2. **Feature Engineering**  
   - Feature selection  
   - Dimensionality reduction  
   - Encoding categorical features  

3. **Model Training**  
   - Using Random Forest Regressor and XGBoost  
   - Hyperparameter tuning and cross-validation  

4. **Evaluation**  
   - Metrics: MAE, RMSE, R² Score  
   - Comparison of models based on accuracy and robustness  

---

## 🔍 Exploratory Data Analysis

EDA was performed to:
- Visualize rental distributions across areas
- Study feature correlation
- Identify outliers and data inconsistencies

---

## 🤖 Machine Learning Models

### 🔸 XGBoost
- Handles missing values and non-linearity
- Fast and scalable tree boosting system  
- Key parameters:
  - `n_estimators=300`
  - `learning_rate=0.05`
  - `max_depth=10`
  - `subsample=0.8`

### 🔹 Random Forest Regressor
- Combines multiple decision trees
- Reduces overfitting
- Stable performance on mixed-type structured data

---

## 📊 Model Evaluation

Metrics used:
- **R² Score**: Measures variance explained  
- **MAE** (Mean Absolute Error)  
- **RMSE** (Root Mean Squared Error)

---

## 📈 Results

| Model          | R² Score | MAE   | RMSE  |
|----------------|----------|-------|-------|
| XGBoost        | ~0.87    | ~700  | ~1100 |
| Random Forest  | ~0.85    | ~750  | ~1200 |

---

## 📄 Flowchart

![Project Flowchart](A_flowchart_illustration_depicts_a_rental_price_pr.png)

---

## 💬 Future Scope

- Integration with real-time property listing platforms  
- Predicting sale prices alongside rental values  
- Extending the model to other Indian cities

---

## 📚 Citations

- Chen, T., & Guestrin, C. (2016). [XGBoost: A Scalable Tree Boosting System](https://dl.acm.org/doi/10.1145/2939672.2939785).
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Kaggle Dataset: [Rental Price of Pune](https://www.kaggle.com/datasets/anantsakhare/rental-price-of-indias-it-capital-pune-mh-ind)
