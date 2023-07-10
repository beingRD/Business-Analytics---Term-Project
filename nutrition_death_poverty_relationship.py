import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
cause_of_deaths_data = pd.read_csv('dataset/cause_of_deaths.csv')
poverty_data = pd.read_csv('dataset/MPI_national.csv')

# Merge the datasets based on the common identifier (ISO in MPI_national and Code in cause_of_deaths)
merged_data = pd.merge(cause_of_deaths_data, poverty_data, left_on='Code', right_on='ISO')

# Select the variables of interest
death_cause_1 = 'Protein-Energy Malnutrition'
death_cause_2 = 'Nutritional Deficiencies'
poverty_measure_1 = 'MPI Urban'
poverty_measure_2 = 'MPI Rural'

# Scatter plot: Death causes vs Poverty measures
plt.scatter(merged_data[death_cause_1], merged_data[poverty_measure_1], label=death_cause_1)
plt.scatter(merged_data[death_cause_2], merged_data[poverty_measure_2], label=death_cause_2)
plt.xlabel('Death Causes')
plt.ylabel('Poverty Measures')
plt.title('Relationship between Death Causes and Poverty Measures')
plt.legend()
plt.show()

# Line plot: Death causes and Poverty measures over time
years = merged_data['Year'].unique()
fig, ax = plt.subplots()
ax.plot(years, merged_data.groupby('Year')[death_cause_1].sum(), label=death_cause_1)
ax.plot(years, merged_data.groupby('Year')[death_cause_2].sum(), label=death_cause_2)
ax.set_xlabel('Year')
ax.set_ylabel('Number of Deaths')
ax.set_title('Death Causes over Time')
ax.legend()
plt.show()

# Heatmap: Correlation between Death causes and Poverty measures
correlation_matrix = merged_data[[death_cause_1, death_cause_2, poverty_measure_1, poverty_measure_2]].corr()
plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Correlation between Death Causes and Poverty Measures')
plt.show()
