# STEP 0: Load Libraries and Data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
file_path = r"C:\KS\Bullying & Economic indicators.csv"
df = pd.read_csv(file_path)

# Display first few rows
print(df.head())



# STEP 1: Max and Min Bullying Analysis
print("=== HIGHEST BULLYING RATES ===")
print("Highest Total Bullying:", df.loc[df['Total'].idxmax()][['Country', 'Total']])
print("Highest Male Bullying:", df.loc[df['Male'].idxmax()][['Country', 'Male']])
print("Highest Female Bullying:", df.loc[df['Female'].idxmax()][['Country', 'Female']])

print("\n=== LOWEST BULLYING RATES ===")
print("Lowest Total Bullying:", df.loc[df['Total'].idxmin()][['Country', 'Total']])
print("Lowest Male Bullying:", df.loc[df['Male'].idxmin()][['Country', 'Male']])
print("Lowest Female Bullying:", df.loc[df['Female'].idxmin()][['Country', 'Female']])
