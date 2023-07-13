import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the malnutrition-related deaths dataset
malnutrition_data = pd.read_csv('dataset/malnutrition_deaths_by_age.csv')

# Read the causes of death dataset
causes_data = pd.read_csv('dataset/cause_of_deaths.csv')

# Read the poverty dataset
poverty_data = pd.read_csv('dataset/poverty_data.csv')

# Merge the datasets based on the country/entity column
merged_data = malnutrition_data.merge(causes_data, on='Code')
merged_data = pd.merge(merged_data, poverty_data, left_on='Code', right_on='ISO')

# Age-wise Death Rate Plot
age_groups = ['Deaths - Protein-energy malnutrition - Sex: Both - Age: 70+ years (Number)',
              'Deaths - Protein-energy malnutrition - Sex: Both - Age: 50-69 years (Number)',
              'Deaths - Protein-energy malnutrition - Sex: Both - Age: 15-49 years (Number)',
              'Deaths - Protein-energy malnutrition - Sex: Both - Age: 5-14 years (Number)',
              'Deaths - Protein-energy malnutrition - Sex: Both - Age: Under 5 (Number)']
age_deaths = merged_data.groupby('Code')[age_groups].sum()

# Plotting age-wise death rate
age_deaths.T.plot(kind='line', marker='o')

plt.xlabel('Age Group')
plt.ylabel('Number of Deaths')
plt.title('Age-wise Death Rate')
plt.legend(title='Country')
plt.show()

# Type of Malnutrition Causing Deaths Filtered by Gender

# Correlation Analysis with Poverty
poverty_correlation = merged_data[['Deaths - Protein-energy malnutrition - Sex: Both - Age: 70+ years (Number)',
                                  'Deaths - Protein-energy malnutrition - Sex: Both - Age: 50-69 years (Number)',
                                  'Deaths - Protein-energy malnutrition - Sex: Both - Age: 15-49 years (Number)',
                                  'Deaths - Protein-energy malnutrition - Sex: Both - Age: 5-14 years (Number)',
                                  'Deaths - Protein-energy malnutrition - Sex: Both - Age: Under 5 (Number)',
                                  'Headcount Ratio Rural']].corr()

sns.heatmap(poverty_correlation, annot=True, cmap='coolwarm')

plt.title('Correlation: Malnutrition Deaths and Poverty Rate')
plt.show()
