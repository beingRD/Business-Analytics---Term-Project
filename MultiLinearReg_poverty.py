import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

# We will use some methods from the sklearn module
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

# Read the malnutrition-related deaths dataset
malnutrition_data = pd.read_csv('dataset/malnutrition_deaths_by_age.csv')
# Read the causes of death dataset
causes_data = pd.read_csv('dataset/cause_of_deaths.csv')
# Read the poverty dataset
poverty_data = pd.read_csv('dataset/poverty_data.csv')

# Merge the datasets based on the country/entity column
merged_data = malnutrition_data.merge(causes_data, on='Code')
merged_data = pd.merge(merged_data, poverty_data, left_on='Code', right_on='ISO')

# Remove non-numeric columns
merged_data_numeric = merged_data.select_dtypes(include=[np.number])

# Print descriptive statistics
print(merged_data_numeric.describe())

# Create the correlation matrix and represent it as a heatmap.
correlation_matrix = merged_data_numeric.corr()
sn.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Setting the value for X and Y
X = merged_data_numeric.iloc[:, 0:-1]
Y = merged_data_numeric.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

# Fit the linear regression model
reg_model = LinearRegression().fit(X_train, y_train)

# Printing the model coefficients
print('Intercept:', reg_model.intercept_)
print('Coefficients:', list(zip(X.columns, reg_model.coef_)))

# Predict on the test set
y_pred = reg_model.predict(X_test)

# Actual value and the predicted value
reg_model_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})
print(reg_model_diff)

# Evaluation metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print('Mean Absolute Error:', mae)
print('Mean Square Error:', mse)
print('Root Mean Square Error:', rmse)
