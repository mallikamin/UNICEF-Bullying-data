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








# STEP 3: GDP vs Bullying Analysis
plt.figure(figsize=(14, 6))

# Scatter plot: GDP vs Total Bullying
plt.subplot(1, 3, 1)
plt.scatter(df['GDP Per Capita (USD)'], df['Total'], alpha=0.6, color='purple')
plt.xlabel('GDP Per Capita (USD)')
plt.ylabel('Total Bullying (%)')
plt.title('GDP vs Total Bullying')

# Add trend line
z = np.polyfit(df['GDP Per Capita (USD)'], df['Total'], 1)
p = np.poly1d(z)
plt.plot(df['GDP Per Capita (USD)'], p(df['GDP Per Capita (USD)']), "r--", alpha=0.8)

# Male
plt.subplot(1, 3, 2)
plt.scatter(df['GDP Per Capita (USD)'], df['Male'], alpha=0.6, color='blue')
plt.xlabel('GDP Per Capita (USD)')
plt.ylabel('Male Bullying (%)')
plt.title('GDP vs Male Bullying')

z_m = np.polyfit(df['GDP Per Capita (USD)'], df['Male'], 1)
p_m = np.poly1d(z_m)
plt.plot(df['GDP Per Capita (USD)'], p_m(df['GDP Per Capita (USD)']), "r--", alpha=0.8)

# Female
plt.subplot(1, 3, 3)
plt.scatter(df['GDP Per Capita (USD)'], df['Female'], alpha=0.6, color='pink')
plt.xlabel('GDP Per Capita (USD)')
plt.ylabel('Female Bullying (%)')
plt.title('GDP vs Female Bullying')

z_f = np.polyfit(df['GDP Per Capita (USD)'], df['Female'], 1)
p_f = np.poly1d(z_f)
plt.plot(df['GDP Per Capita (USD)'], p_f(df['GDP Per Capita (USD)']), "r--", alpha=0.8)

plt.tight_layout()
plt.show()

# Correlation
print("Correlation between GDP and Bullying:")
print("Total:", df['GDP Per Capita (USD)'].corr(df['Total']).round(3))
print("Male:", df['GDP Per Capita (USD)'].corr(df['Male']).round(3))
print("Female:", df['GDP Per Capita (USD)'].corr(df['Female']).round(3))
















# STEP 4: Poverty Rate vs Bullying
plt.figure(figsize=(14, 6))

# Scatter plots
plt.subplot(1, 3, 1)
plt.scatter(df['Poverty Rate (%)'], df['Total'], alpha=0.6, color='orange')
plt.xlabel('Poverty Rate (%)')
plt.ylabel('Total Bullying (%)')
plt.title('Poverty vs Total Bullying')

z_t = np.polyfit(df['Poverty Rate (%)'], df['Total'], 1)
p_t = np.poly1d(z_t)
plt.plot(df['Poverty Rate (%)'], p_t(df['Poverty Rate (%)']), "r--", alpha=0.8)

plt.subplot(1, 3, 2)
plt.scatter(df['Poverty Rate (%)'], df['Male'], alpha=0.6, color='blue')
plt.xlabel('Poverty Rate (%)')
plt.ylabel('Male Bullying (%)')
plt.title('Poverty vs Male Bullying')

z_m = np.polyfit(df['Poverty Rate (%)'], df['Male'], 1)
p_m = np.poly1d(z_m)
plt.plot(df['Poverty Rate (%)'], p_m(df['Poverty Rate (%)']), "r--", alpha=0.8)

plt.subplot(1, 3, 3)
plt.scatter(df['Poverty Rate (%)'], df['Female'], alpha=0.6, color='pink')
plt.xlabel('Poverty Rate (%)')
plt.ylabel('Female Bullying (%)')
plt.title('Poverty vs Female Bullying')

z_f = np.polyfit(df['Poverty Rate (%)'], df['Female'], 1)
p_f = np.poly1d(z_f)
plt.plot(df['Poverty Rate (%)'], p_f(df['Poverty Rate (%)']), "r--", alpha=0.8)

plt.tight_layout()
plt.show()

# Correlation
print("Correlation between Poverty Rate and Bullying:")
print("Total:", df['Poverty Rate (%)'].corr(df['Total']).round(3))
print("Male:", df['Poverty Rate (%)'].corr(df['Male']).round(3))
print("Female:", df['Poverty Rate (%)'].corr(df['Female']).round(3))


















# STEP 5: Education Spending vs Bullying
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.scatter(df['Education Spending (% of GDP)'], df['Total'], alpha=0.6, color='green')
plt.xlabel('Education Spending (% of GDP)')
plt.ylabel('Total Bullying (%)')
plt.title('Education Spending vs Total Bullying')

z_t = np.polyfit(df['Education Spending (% of GDP)'], df['Total'], 1)
p_t = np.poly1d(z_t)
plt.plot(df['Education Spending (% of GDP)'], p_t(df['Education Spending (% of GDP)']), "r--", alpha=0.8)

plt.subplot(1, 3, 2)
plt.scatter(df['Education Spending (% of GDP)'], df['Male'], alpha=0.6, color='blue')
plt.xlabel('Education Spending (% of GDP)')
plt.ylabel('Male Bullying (%)')
plt.title('Education Spending vs Male Bullying')

z_m = np.polyfit(df['Education Spending (% of GDP)'], df['Male'], 1)
p_m = np.poly1d(z_m)
plt.plot(df['Education Spending (% of GDP)'], p_m(df['Education Spending (% of GDP)']), "r--", alpha=0.8)

plt.subplot(1, 3, 3)
plt.scatter(df['Education Spending (% of GDP)'], df['Female'], alpha=0.6, color='pink')
plt.xlabel('Education Spending (% of GDP)')
plt.ylabel('Female Bullying (%)')
plt.title('Education Spending vs Female Bullying')

z_f = np.polyfit(df['Education Spending (% of GDP)'], df['Female'], 1)
p_f = np.poly1d(z_f)
plt.plot(df['Education Spending (% of GDP)'], p_f(df['Education Spending (% of GDP)']), "r--", alpha=0.8)

plt.tight_layout()
plt.show()

# Correlation
print("Correlation between Education Spending and Bullying:")
print("Total:", df['Education Spending (% of GDP)'].corr(df['Total']).round(3))
print("Male:", df['Education Spending (% of GDP)'].corr(df['Male']).round(3))
print("Female:", df['Education Spending (% of GDP)'].corr(df['Female']).round(3))












import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = r"C:\KS\Bullying & Economic indicators.csv"
df = pd.read_csv(file_path)

# Clean data: drop rows with missing Male/Female
df_clean = df.dropna(subset=['Male', 'Female']).copy()
df_clean['Male'] = pd.to_numeric(df_clean['Male'], errors='coerce')
df_clean['Female'] = pd.to_numeric(df_clean['Female'], errors='coerce')
df_clean = df_clean.dropna(subset=['Male', 'Female'])

# Ensure categorical consistency
df_clean['Income Level'] = df_clean['Income Level'].astype('category')
df_clean['Region'] = df_clean['Region'].astype('category')

# Set style for better visuals
sns.set_style("whitegrid")


plt.figure(figsize=(6, 6))
sns.boxplot(data=df_clean, x='Income Level', y='Total', palette='Set2')
plt.title('Distribution of Total Bullying by Income Level')
plt.xlabel('Income Level')
plt.ylabel('Bullying Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()







# STEP 7: Summary Insights
print("\n" + "="*60)
print("            FINAL ANALYSIS SUMMARY")
print("="*60)

print("1. Highest Total Bullying: ", df.loc[df['Total'].idxmax()]['Country'], f"({df['Total'].max()}%)")
print("2. Lowest Total Bullying: ", df.loc[df['Total'].idxmin()]['Country'], f"({df['Total'].min()}%)")

print(f"3. Correlation Summary:")
print(f"   GDP vs Total Bullying: {df['GDP Per Capita (USD)'].corr(df['Total']):.3f}")
print(f"   Poverty vs Total Bullying: {df['Poverty Rate (%)'].corr(df['Total']):.3f}")
print(f"   Education Spending vs Total Bullying: {df['Education Spending (% of GDP)'].corr(df['Total']):.3f}")

print(f"\n4. Income Level Trends:")
income_avg = df.groupby('Income Level')['Total'].mean().sort_values()
for level, avg in income_avg.items():
    print(f"   {level}: {avg:.2f}%")

print(f"\n5. Region with Highest Average Bullying: {df.groupby('Region')['Total'].mean().idxmax()}")
print(f"   Region with Lowest Average Bullying: {df.groupby('Region')['Total'].mean().idxmin()}")














