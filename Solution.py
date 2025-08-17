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








# STEP 2: Visualization - Top and Bottom 10 Countries
top_10_total = df.nlargest(10, 'Total')
bottom_10_total = df.nsmallest(10, 'Total')

fig, ax = plt.subplots(3, 1, figsize=(14, 18))

# Total Bullying
ax[0].barh(top_10_total['Country'], top_10_total['Total'], color='red', alpha=0.7)
ax[0].set_title('Top 10 Countries - Total Bullying (%)')
ax[0].invert_yaxis()

ax[1].barh(bottom_10_total['Country'], bottom_10_total['Total'], color='green', alpha=0.7)
ax[1].set_title('Bottom 10 Countries - Total Bullying (%)')
ax[1].invert_yaxis()

# Gender Comparison for Top 10 Total
top_10_gender = top_10_total[['Country', 'Male', 'Female']].set_index('Country')
top_10_gender.plot(kind='bar', ax=ax[2], color=['skyblue', 'lightcoral'])
ax[2].set_title('Male vs Female Bullying in Top 10 Total Bullying Countries')
ax[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
