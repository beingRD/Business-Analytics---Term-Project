import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load the data
data = pd.read_csv('dataset/cause_of_deaths.csv')

# EDA - Investigating specific connections
cause_1 = 'Meningitis'
cause_2 = "Alzheimer's Disease and Other Dementias"

# Scatter plot
plt.scatter(data[cause_1], data[cause_2])
plt.xlabel(cause_1)
plt.ylabel(cause_2)
plt.title('Connection between ' + cause_1 + ' and ' + cause_2)

# Linear regression
slope, intercept, r_value, p_value, std_err = linregress(data[cause_1], data[cause_2])
regression_line = slope * data[cause_1] + intercept
plt.plot(data[cause_1], regression_line, color='red', label='Regression Line')

# Add equation and R-squared value as text
equation = f'Line Equation: y = {slope:.2f}x + {intercept:.2f}'
r_squared = f'R-squared: {r_value**2:.2f}'
plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes, fontsize=10, ha='left', va='top')
plt.text(0.05, 0.88, r_squared, transform=plt.gca().transAxes, fontsize=10, ha='left', va='top')

# Beautify the plot
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
