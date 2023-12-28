import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("salary.csv")
pd.set_option('display.max_columns', None)

# Check data for null values and display null count per column
print("Null Values in Dataset")
print(data.isnull().sum())

# Sort the data by years for easier analysis
data.sort_values(["years"], axis=0, ascending=[True], inplace=True)

# EDA: Visualize data distribution and relationships
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
axes[0].hist(data['years'], bins=20, color='skyblue', edgecolor='black')
axes[0].set_title('Distribution of Years')
axes[0].set_xlabel('Years')
axes[0].set_ylabel('Frequency')

axes[1].scatter(data['years'], data['salary'], color='coral', alpha=0.7)
axes[1].set_title('Relationship between Years and Salary')
axes[1].set_xlabel('Years')
axes[1].set_ylabel('Salary')
plt.tight_layout()
plt.show()

# Split the data into features (X) and target variable (y)
X = data[['years']]
y = data['salary']

# Initialize the linear regression model
model = LinearRegression()

# Use k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Adjust the number of splits as needed

# Lists to store results
mse_list = []
r2_list = []

# Perform cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse_fold = mean_squared_error(y_test, y_pred)
    r2_fold = r2_score(y_test, y_pred)

    mse_list.append(mse_fold)
    r2_list.append(r2_fold)

# Calculate mean scores over folds
mse_cv = np.mean(mse_list)
r2_cv = np.mean(r2_list)

print(f'Mean Squared Error (Cross-Validated): {mse_cv:.2f}')
print(f'R-squared (R2) Score (Cross-Validated): {r2_cv:.2f}')

# Plot the original data and the regression line
plt.scatter(X, y, label='Original Data')
plt.xlabel('Years')
plt.ylabel('Salary')
plt.legend()
plt.title('Linear Regression: Salary vs. Years of Experience')
plt.text(2, 80000, f'Mean Squared Error (Cross-Validated): {mse_cv:.2f}\nR-squared (R2) Score (Cross-Validated): {r2_cv:.2f}',
         fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
plt.show()
