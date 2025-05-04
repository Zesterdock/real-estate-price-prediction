#Train and main
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("train.csv")
print("Dataset Loaded Successfully!")
print(df.shape) 

# Drop unnecessary columns
columns_to_drop = [
    'bathrooms', 'floor_number', 'floor_type', 'corner_pro', 'wheelchairadption', 
    'petfacility', 'no_room', 'pooja_room', 'study_room', 'others', 'servant_room', 
    'store_room', 'address'
]
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Handle categorical replacements
furnishing_map = {'Unfurnished': 0, 'Semifurnished': 1, 'Furnished': 2}
df['furnishing'] = df['furnishing'].map(furnishing_map)
df['avalable_for'].fillna(df['avalable_for'].mode()[0], inplace=True)

df['maintenance_amt'] = pd.to_numeric(df['maintenance_amt'], errors='coerce').fillna(0)

# Remove outliers using IQR method
Q1 = df['rent'].quantile(0.25)
Q3 = df['rent'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['rent'] >= (Q1 - 1.5 * IQR)) & (df['rent'] <= (Q3 + 1.5 * IQR))]

# One-Hot Encoding for categorical variables
categorical_columns = ['avalable_for', 'facing', 'powerbackup', 'gate_community']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Define features and target variable
X = df.drop(columns=['rent']).apply(pd.to_numeric, errors='coerce')
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

y = np.log1p(df['rent'])

# Scale numeric columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Feature selection & data preparation complete!")

# Train XGBoost Model on train-test split
xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=10, random_state=42)
xgb_model.fit(X_train, y_train)

# Make Predictions on test set
y_pred_log = xgb_model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)

# Evaluate Model Performance on test set
mae_xgb_test = mean_absolute_error(y_test_actual, y_pred)
mse_xgb_test = mean_squared_error(y_test_actual, y_pred)
r2_xgb_test = r2_score(y_test_actual, y_pred)

print("\nXGBoost Model Performance on Test Set:")
print(f"Mean Absolute Error (MAE): {mae_xgb_test}")
print(f"Mean Squared Error (MSE): {mse_xgb_test}")
print(f"R² Score: {r2_xgb_test}")

# Train XGBoost Model on full training data
xgb_model_full = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=10, random_state=42)
xgb_model_full.fit(X_scaled, y)

# Make predictions on the full training data
y_pred_log_full = xgb_model_full.predict(X_scaled)
y_pred_full = np.expm1(y_pred_log_full)
y_actual_full = np.expm1(y)

# Evaluate Model Performance on full training data
mae_xgb_full = mean_absolute_error(y_actual_full, y_pred_full)
mse_xgb_full = mean_squared_error(y_actual_full, y_pred_full)
r2_xgb_full = r2_score(y_actual_full, y_pred_full)

print("\nXGBoost Model Performance on Full Training Data:")
print(f"Mean Absolute Error (MAE): {mae_xgb_full}")
print(f"Mean Squared Error (MSE): {mse_xgb_full}")
print(f"R² Score: {r2_xgb_full}")

# Feature Importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': xgb_model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Make predictions on train set for train-test comparison
y_train_pred_log = xgb_model.predict(X_train)
y_train_pred = np.expm1(y_train_pred_log)
y_train_actual = np.expm1(y_train)

# Visualizations
sns.set_style("whitegrid")

# Rent Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['rent'], bins=30, kde=True)
plt.title('Price Distribution of Rent')
plt.xlabel('Rent')
plt.ylabel('Frequency')
plt.savefig('rent_distribution.png')
plt.close()

# Area Sizes
plt.figure(figsize=(8, 6))
sns.histplot(df['area'], bins=30, kde=True)
plt.title('Distribution of Area Sizes')
plt.xlabel('Area (sq ft)')
plt.ylabel('Frequency')
plt.savefig('area_distribution.png')
plt.close()

# Percentage of Furnished and Unfurnished Flats
plt.figure(figsize=(6, 6))
furnishing_labels = ['Unfurnished', 'Semifurnished', 'Furnished']
furnishing_counts = df['furnishing'].value_counts().sort_index()
plt.pie(furnishing_counts, labels=furnishing_labels, autopct='%1.1f%%', colors=['blue', 'green', 'orange'])
plt.title('Percentage of Furnished and Unfurnished Flats')
plt.savefig('furnishing_distribution.png')
plt.close()

# Rent Amount by Number of Bedrooms
plt.figure(figsize=(8, 6))
sns.boxplot(x='bedroom', y='rent', data=df)
plt.title('Rent Amount Depending on Number of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Rent')
plt.savefig('rent_by_bedrooms.png')
plt.close()

# COMPARISON VISUALIZATIONS

# Model Metrics Comparison
metrics = ['MAE', 'MSE', 'R²']
test_metrics = [mae_xgb_test, mse_xgb_test, r2_xgb_test]
train_metrics = [mae_xgb_full, mse_xgb_full, r2_xgb_full]

plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(metrics))

plt.bar(index, test_metrics, bar_width, label='Test Set (Train-Test Split)')
plt.bar(index + bar_width, train_metrics, bar_width, label='Full Training Data')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Model Performance Comparison')
plt.xticks(index + bar_width / 2, metrics)
plt.legend()
plt.savefig('metrics_comparison.png')
plt.close()

# Predicted vs Actual Comparison - Test Set
plt.figure(figsize=(8, 6))
plt.scatter(y_test_actual, y_pred, alpha=0.5, color="blue", label="Test Set")
plt.plot([min(y_test_actual), max(y_test_actual)], [min(y_test_actual), max(y_test_actual)], 
         linestyle='--', color='red', label="Perfect Prediction")
plt.xlabel("Actual Rent")
plt.ylabel("Predicted Rent")
plt.title("Test Set: Actual vs Predicted Rent")
plt.legend()
plt.savefig('test_actual_vs_predicted.png')
plt.close()

# Predicted vs Actual Comparison - Train Set
plt.figure(figsize=(8, 6))
plt.scatter(y_train_actual, y_train_pred, alpha=0.5, color="green", label="Train Set")
plt.plot([min(y_train_actual), max(y_train_actual)], [min(y_train_actual), max(y_train_actual)], 
         linestyle='--', color='red', label="Perfect Prediction")
plt.xlabel("Actual Rent")
plt.ylabel("Predicted Rent")
plt.title("Train Set: Actual vs Predicted Rent")
plt.legend()
plt.savefig('train_actual_vs_predicted.png')
plt.close()

# Predicted vs Actual Comparison - Full Training Data
plt.figure(figsize=(8, 6))
plt.scatter(y_actual_full, y_pred_full, alpha=0.5, color="purple", label="Full Training Data")
plt.plot([min(y_actual_full), max(y_actual_full)], [min(y_actual_full), max(y_actual_full)], 
         linestyle='--', color='red', label="Perfect Prediction")
plt.xlabel("Actual Rent")
plt.ylabel("Predicted Rent")
plt.title("Full Training Data: Actual vs Predicted Rent")
plt.legend()
plt.savefig('full_actual_vs_predicted.png')
plt.close()

# Combined Actual vs Predicted plot
plt.figure(figsize=(10, 8))
plt.scatter(y_test_actual, y_pred, alpha=0.5, color="blue", label="Test Set")
plt.scatter(y_train_actual, y_train_pred, alpha=0.5, color="green", label="Train Set")
plt.scatter(y_actual_full, y_pred_full, alpha=0.3, color="purple", label="Full Training Data")
plt.plot([min(min(y_test_actual), min(y_actual_full)), max(max(y_test_actual), max(y_actual_full))], 
         [min(min(y_test_actual), min(y_actual_full)), max(max(y_test_actual), max(y_actual_full))], 
         linestyle='--', color='red', label="Perfect Prediction")
plt.xlabel("Actual Rent")
plt.ylabel("Predicted Rent")
plt.title("Comparison of Actual vs Predicted Rent Across Different Datasets")
plt.legend()
plt.savefig('combined_actual_vs_predicted.png')
plt.close()

# Error Distribution Comparison
test_errors = y_test_actual - y_pred
train_errors = y_train_actual - y_train_pred
full_errors = y_actual_full - y_pred_full

plt.figure(figsize=(10, 6))
sns.kdeplot(test_errors, color="blue", label="Test Set Errors")
sns.kdeplot(train_errors, color="green", label="Train Set Errors")
sns.kdeplot(full_errors, color="purple", label="Full Training Data Errors")
plt.xlabel("Prediction Error")
plt.ylabel("Density")
plt.title("Error Distribution Comparison")
plt.legend()
plt.savefig('error_distribution.png')
plt.close()

# Feature Importance Top 10
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(10)
sns.barplot(x="Importance", y="Feature", data=top_features)
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("Visualization complete! All plots have been saved.")



