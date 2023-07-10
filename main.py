import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Read the Global Superstore dataset in .xlsx format
df = pd.read_csv("dataset/cause_of_deaths.csv")

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head(5).to_string(index=False))

# Display information about the dataset
print("\nInformation about the dataset:")
print(df.info())

# Display statistics about the dataset
print("\nStatistics about the dataset:")

# Transpose the describe() output for better readability
print(df.describe().transpose().to_string())

