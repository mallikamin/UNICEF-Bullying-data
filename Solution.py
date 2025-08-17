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








# STEP 2: Separate Visualizations for Top & Bottom 10 Countries

import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = r"C:\KS\Bullying & Economic indicators.csv"
df = pd.read_csv(file_path)

# Sort data for top and bottom 10
top_10_total = df.nlargest(10, 'Total')
bottom_10_total = df.nsmallest(10, 'Total')

# --- FIGURE 1: Top 10 Countries - Total Bullying ---
plt.figure(figsize=(10, 6))
plt.barh(top_10_total['Country'], top_10_total['Total'], color='red', alpha=0.7)
plt.xlabel('Bullying Rate (%)')
plt.title('Top 10 Countries - Total Bullying Rate')
plt.gca().invert_yaxis()  # Highest on top
plt.tight_layout()
plt.show()

# --- FIGURE 2: Bottom 10 Countries - Total Bullying ---
plt.figure(figsize=(10, 6))
plt.barh(bottom_10_total['Country'], bottom_10_total['Total'], color='green', alpha=0.7)
plt.xlabel('Bullying Rate (%)')
plt.title('Bottom 10 Countries - Total Bullying Rate')
plt.gca().invert_yaxis()  # Lowest on top
plt.tight_layout()
plt.show()

# --- FIGURE 3: Male vs Female Bullying in Top 10 Total Bullying Countries ---
top_10_gender = top_10_total[['Country', 'Male', 'Female']].set_index('Country')

plt.figure(figsize=(12, 6))
top_10_gender.plot(kind='bar', color=['skyblue', 'lightcoral'], width=0.8)
plt.title('Male vs Female Bullying in Top 10 Total Bullying Countries')
plt.xlabel('Country')
plt.ylabel('Bullying Rate (%)')
plt.xticks(rotation=45, ha='right')
plt.legend(['Male', 'Female'])
plt.tight_layout()
plt.show()

# --- FIGURE 4: Male vs Female Bullying in Bottom 10 Total Bullying Countries ---
bottom_10_gender = bottom_10_total[['Country', 'Male', 'Female']].set_index('Country')

plt.figure(figsize=(12, 6))
bottom_10_gender.plot(kind='bar', color=['skyblue', 'lightcoral'], width=0.8)
plt.title('Male vs Female Bullying in Bottom 10 Total Bullying Countries')
plt.xlabel('Country')
plt.ylabel('Bullying Rate (%)')
plt.xticks(rotation=45, ha='right')
plt.legend(['Male', 'Female'])
plt.tight_layout()
plt.show()
