import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import matplotlib.ticker as mticker

# Load the data
data = pd.read_csv('dataset/cause_of_deaths.csv')

# Filter data for the desired years
start_year = 1990
end_year = 2019
data_filtered = data[(data['Year'] >= start_year) & (data['Year'] <= end_year)]

# EDA - Investigating specific connections
cause_1 = 'Protein-Energy Malnutrition'
cause_2 = 'Nutritional Deficiencies'

# Scatter plot
plt.scatter(data_filtered[cause_1], data_filtered[cause_2])
plt.xlabel(cause_1)
plt.ylabel(cause_2)
plt.title('Connection between ' + cause_1 + ' and ' + cause_2)

# Linear regression
slope, intercept, r_value, p_value, std_err = linregress(data_filtered[cause_1], data_filtered[cause_2])
regression_line = slope * data_filtered[cause_1] + intercept
plt.plot(data_filtered[cause_1], regression_line, color='red', label='Regression Line')

# Add equation and R-squared value as text
equation = f'Line Equation: y = {slope:.2f}x + {intercept:.2f}'
r_squared = f'R-squared: {r_value**2:.2f}'
plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes, fontsize=10, ha='left', va='top')
plt.text(0.05, 0.88, r_squared, transform=plt.gca().transAxes, fontsize=10, ha='left', va='top')

# Beautify the plot
plt.legend()
plt.tight_layout()

# Show the scatter plot
plt.show()

# Bar plot - Top affected countries
top_countries = data_filtered.groupby('Country/Territory')[[cause_1, cause_2]].sum().nlargest(10, [cause_1, cause_2]) # Convert to thousands
ax = top_countries.plot(kind='bar')
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))  # Format y-axis ticks with thousand separator

plt.xlabel('Country')
plt.ylabel('Number of Deaths')
plt.title(f'Top 10 Countries Most Affected by {cause_1} and {cause_2} ({start_year}-{end_year})')

# Beautify the plot
plt.tight_layout()

# Show the bar plot
plt.show()
