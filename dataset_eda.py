import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact, Dropdown

# Load the data
data = pd.read_csv('dataset/cause_of_deaths.csv')

# EDA - Summary statistics
print("Summary Statistics:")
print(data.describe().to_string())

# EDA - Correlation matrix
correlation_matrix = data.drop(['Country/Territory', 'Code', 'Year'], axis=1).corr()
print("\nCorrelation Matrix:")
print(correlation_matrix.to_string())

# EDA - Top causes of death
total_deaths = data.drop(['Country/Territory', 'Code'], axis=1).sum(axis=0)
top_causes = total_deaths.sort_values(ascending=False)[:5]
print("\nTop Causes of Death:")
print(top_causes.to_string())

# EDA - Plotting causes of death over time
years = data['Year'].unique()

"""
# Plotting causes of death individually
causes = data.columns[3:]
totals = data.groupby('Year')[causes].sum()

plt.stackplot(years, totals.values.T, labels=causes)

plt.xlabel('Year')
plt.ylabel('No. of Deaths')
plt.title('Causes of Death Over Time')
plt.legend(loc='upper right')
plt.show()
"""
# Plotting causes of death individually
causes = data.columns[3:]
totals = data.groupby('Year')[causes].sum()

fig, ax = plt.subplots()

def update_legend(cause):
    ax.cla()
    ax.stackplot(years, totals.values.T, labels=causes)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Deaths')
    ax.set_title('Causes of Death Over Time')
    ax.legend(loc='upper right')

dropdown = Dropdown(options=causes)
interact(update_legend, cause=dropdown)

plt.show()

# EDA - Investigating specific connections
cause_1 = 'Meningitis'
cause_2 = "Alzheimer's Disease and Other Dementias"

plt.scatter(data[cause_1], data[cause_2])
plt.xlabel(cause_1)
plt.ylabel(cause_2)
plt.title('Connection between ' + cause_1 + ' and ' + cause_2)
plt.show()
