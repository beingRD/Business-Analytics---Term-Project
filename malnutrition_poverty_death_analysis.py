import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Read the datasets
malnutrition_data = pd.read_csv('dataset/malnutrition_deaths_by_age.csv')
causes_data = pd.read_csv('dataset/cause_of_deaths.csv')
poverty_data = pd.read_csv('dataset/poverty_data.csv')

# Merge the datasets based on the country/entity column
merged_data = malnutrition_data.merge(causes_data, on='Code')
merged_data = pd.merge(merged_data, poverty_data, left_on='Code', right_on='ISO')

# Create correlation heatmaps
datasets = {'malnutrition_data': malnutrition_data, 'causes_data': causes_data, 'poverty_data': poverty_data, 'merged_data': merged_data}
for name, data in datasets.items():
    corr = data.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(10,10))
    sn.heatmap(corr, annot=True, cmap='coolwarm')
    #plt.title(f'Correlation Matrix - {name}')
    plt.show()

# Print descriptive statistics
print(merged_data.describe())

# Set the value for X and Y
X = merged_data.drop(columns=['Nutritional Deficiencies'])
Y = merged_data['Nutritional Deficiencies']

# Ensure all columns in X are numeric and handle if not
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

# Fit the linear regression model
reg_model = LinearRegression().fit(X_train, y_train)

# Print the model coefficients
print('Intercept:', reg_model.intercept_)
print('Coefficients:', list(zip(X.columns, reg_model.coef_)))

# Predict on the test set
y_pred = reg_model.predict(X_test)

# Create a DataFrame to compare the actual and predicted values
reg_model_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})
print(reg_model_diff)

# Evaluation metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print('Mean Absolute Error:', mae)
print('Mean Square Error:', mse)
print('Root Mean Square Error:', rmse)

# Residual plot
plt.scatter(y_pred, y_test-y_pred)
plt.title("Residual plot")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.show()

# Actual vs Predicted values
plt.scatter(y_test, y_pred, color='blue', label='Predicted Values')
plt.scatter(y_test, y_test, color='red', label='Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()
