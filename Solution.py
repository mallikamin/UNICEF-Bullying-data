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
