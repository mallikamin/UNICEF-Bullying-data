
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the data
file_path = r'C:\KS\Bullying & Economic indicators.csv'
df = pd.read_csv(file_path)

print("=== UNICEF BULLYING DATA ANALYSIS ===")
print(f"Dataset shape: {df.shape}")
print(f"Countries analyzed: {df['Country'].nunique()}")
print("\nDataset Overview:")
print(df.head())

# Clean the data - handle missing values and data types
print("\n=== DATA CLEANING ===")
# Convert numeric columns
numeric_cols = ['Total', 'Male', 'Female', 'GDP Per Capita (USD)', 'Poverty Rate (%)', 'Education Spending (% of GDP)']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("Missing values per column:")
print(df.isnull().sum())

# Basic Statistics
print("\n=== BASIC STATISTICS ===")
print("Bullying Statistics:")
print(df[['Total', 'Male', 'Female']].describe())

print("\nEconomic Indicators Statistics:")
print(df[['GDP Per Capita (USD)', 'Poverty Rate (%)', 'Education Spending (% of GDP)']].describe())

# ===============================
# HIGHEST AND LOWEST BULLYING RATES
# ===============================

def analyze_extremes(df, column, label):
    """Analyze highest and lowest values for a given column"""
    print(f"\n=== {label.upper()} ANALYSIS ===")
    
    # Remove rows with missing data for this analysis
    clean_df = df.dropna(subset=[column])
    
    if len(clean_df) == 0:
        print(f"No data available for {label}")
        return
    
    # Top 10 highest
    highest = clean_df.nlargest(10, column)[['Country', column, 'Region', 'Income Level']]
    print(f"\nTop 10 Highest {label}:")
    print(highest.to_string(index=False))
    
    # Top 10 lowest
    lowest = clean_df.nsmallest(10, column)[['Country', column, 'Region', 'Income Level']]
    print(f"\nTop 10 Lowest {label}:")
    print(lowest.to_string(index=False))
    
    return highest, lowest

# Analyze extremes for each category
total_high, total_low = analyze_extremes(df, 'Total', 'Total Bullying Rates')
male_high, male_low = analyze_extremes(df, 'Male', 'Male Bullying Rates')  
female_high, female_low = analyze_extremes(df, 'Female', 'Female Bullying Rates')

# ===============================
# REGIONAL ANALYSIS
# ===============================

print("\n=== REGIONAL ANALYSIS ===")
regional_stats = df.groupby('Region').agg({
    'Total': ['count', 'mean', 'median', 'std'],
    'Male': ['mean', 'median'],
    'Female': ['mean', 'median'],
    'GDP Per Capita (USD)': ['mean', 'median'],
    'Poverty Rate (%)': ['mean', 'median'],
    'Education Spending (% of GDP)': ['mean', 'median']
}).round(2)

print("Regional Statistics Summary:")
print(regional_stats)

# ===============================
# INCOME LEVEL ANALYSIS
# ===============================

print("\n=== INCOME LEVEL ANALYSIS ===")
income_stats = df.groupby('Income Level').agg({
    'Total': ['count', 'mean', 'median', 'std'],
    'Male': ['mean', 'median'],
    'Female': ['mean', 'median'], 
    'GDP Per Capita (USD)': ['mean', 'median'],
    'Poverty Rate (%)': ['mean', 'median'],
    'Education Spending (% of GDP)': ['mean', 'median']
}).round(2)

print("Income Level Statistics Summary:")
print(income_stats)

# ===============================
# CORRELATION ANALYSIS
# ===============================

print("\n=== CORRELATION ANALYSIS ===")

# Calculate correlations
correlation_cols = ['Total', 'Male', 'Female', 'GDP Per Capita (USD)', 'Poverty Rate (%)', 'Education Spending (% of GDP)']
correlation_matrix = df[correlation_cols].corr()

print("Correlation Matrix:")
print(correlation_matrix.round(3))

# Key correlations to highlight
print("\n=== KEY FINDINGS ===")

# GDP correlations
gdp_total_corr = df['GDP Per Capita (USD)'].corr(df['Total'])
gdp_male_corr = df['GDP Per Capita (USD)'].corr(df['Male']) 
gdp_female_corr = df['GDP Per Capita (USD)'].corr(df['Female'])

print(f"GDP per Capita vs Bullying Correlations:")
print(f"  - Total Bullying: {gdp_total_corr:.3f}")
print(f"  - Male Bullying: {gdp_male_corr:.3f}")
print(f"  - Female Bullying: {gdp_female_corr:.3f}")

# Poverty correlations
poverty_total_corr = df['Poverty Rate (%)'].corr(df['Total'])
poverty_male_corr = df['Poverty Rate (%)'].corr(df['Male'])
poverty_female_corr = df['Poverty Rate (%)'].corr(df['Female'])

print(f"\nPoverty Rate vs Bullying Correlations:")
print(f"  - Total Bullying: {poverty_total_corr:.3f}")
print(f"  - Male Bullying: {poverty_male_corr:.3f}")
print(f"  - Female Bullying: {poverty_female_corr:.3f}")

# Education spending correlations
edu_total_corr = df['Education Spending (% of GDP)'].corr(df['Total'])
edu_male_corr = df['Education Spending (% of GDP)'].corr(df['Male'])
edu_female_corr = df['Education Spending (% of GDP)'].corr(df['Female'])

print(f"\nEducation Spending vs Bullying Correlations:")
print(f"  - Total Bullying: {edu_total_corr:.3f}")
print(f"  - Male Bullying: {edu_male_corr:.3f}")
print(f"  - Female Bullying: {edu_female_corr:.3f}")

# Gender gap analysis
df['Gender_Gap'] = df['Male'] - df['Female']
gender_gap_mean = df['Gender_Gap'].mean()
print(f"\nGender Gap Analysis:")
print(f"  - Average Gender Gap (Male - Female): {gender_gap_mean:.2f}")
print(f"  - Countries where females have higher bullying rates: {(df['Gender_Gap'] < 0).sum()}")
print(f"  - Countries where males have higher bullying rates: {(df['Gender_Gap'] > 0).sum()}")

# ===============================
# STATISTICAL SIGNIFICANCE TESTS
# ===============================

print("\n=== STATISTICAL SIGNIFICANCE TESTS ===")

# Test if there are significant differences between income levels
income_groups = df.groupby('Income Level')['Total'].apply(list)
if len(income_groups) > 1:
    try:
        f_stat, p_value = stats.f_oneway(*[group for group in income_groups if len(group) > 0])
        print(f"ANOVA test for bullying rates across income levels:")
        print(f"  F-statistic: {f_stat:.3f}, p-value: {p_value:.3f}")
        if p_value < 0.05:
            print("  Result: Significant differences exist between income groups")
        else:
            print("  Result: No significant differences between income groups")
    except:
        print("  Could not perform ANOVA test")

# Test correlation significance
def test_correlation_significance(x, y, label):
    """Test if correlation is statistically significant"""
    clean_data = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(clean_data) > 10:
        corr_coef, p_value = stats.pearsonr(clean_data['x'], clean_data['y'])
        significance = "significant" if p_value < 0.05 else "not significant"
        print(f"{label}: r={corr_coef:.3f}, p={p_value:.3f} ({significance})")

print("\nCorrelation Significance Tests:")
test_correlation_significance(df['GDP Per Capita (USD)'], df['Total'], "GDP vs Total Bullying")
test_correlation_significance(df['Poverty Rate (%)'], df['Total'], "Poverty vs Total Bullying")
test_correlation_significance(df['Education Spending (% of GDP)'], df['Total'], "Education Spending vs Total Bullying")

print("\n=== ANALYSIS COMPLETE ===")
print("Run the visualization scripts for detailed charts and graphs.")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set up the plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Load the data
file_path = r'C:\KS\Bullying & Economic indicators.csv'
df = pd.read_csv(file_path)

# Convert numeric columns
numeric_cols = ['Total', 'Male', 'Female', 'GDP Per Capita (USD)', 'Poverty Rate (%)', 'Education Spending (% of GDP)']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("Creating comprehensive visualizations for bullying data analysis...")

# ===============================
# 1. BASIC BULLYING DISTRIBUTIONS
# ===============================

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Bullying Rates Distribution Analysis', fontsize=16, fontweight='bold')

# Total bullying distribution
axes[0,0].hist(df['Total'].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].set_title('Distribution of Total Bullying Rates')
axes[0,0].set_xlabel('Bullying Rate (%)')
axes[0,0].set_ylabel('Number of Countries')
axes[0,0].axvline(df['Total'].mean(), color='red', linestyle='--', label=f'Mean: {df["Total"].mean():.1f}%')
axes[0,0].legend()

# Male vs Female bullying comparison
gender_data = df[['Male', 'Female']].dropna()
axes[0,1].scatter(gender_data['Male'], gender_data['Female'], alpha=0.6, s=50)
axes[0,1].plot([0, 80], [0, 80], 'r--', label='Equal rates line')
axes[0,1].set_xlabel('Male Bullying Rate (%)')
axes[0,1].set_ylabel('Female Bullying Rate (%)')
axes[0,1].set_title('Male vs Female Bullying Rates')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Box plots by region
region_data = df.dropna(subset=['Total'])
axes[1,0].boxplot([region_data[region_data['Region'] == region]['Total'].values 
                   for region in region_data['Region'].unique()], 
                   labels=[region.replace(' & ', '\n& ') for region in region_data['Region'].unique()])
axes[1,0].set_title('Bullying Rates by Region')
axes[1,0].set_ylabel('Bullying Rate (%)')
axes[1,0].tick_params(axis='x', rotation=45)

# Box plots by income level
income_order = ['Low income', 'Lower middle income', 'Upper middle income', 'High income']
income_data = []
income_labels = []
for income in income_order:
    data = df[df['Income Level'] == income]['Total'].dropna().values
    if len(data) > 0:
        income_data.append(data)
        income_labels.append(income.replace(' ', '\n'))

axes[1,1].boxplot(income_data, labels=income_labels)
axes[1,1].set_title('Bullying Rates by Income Level')
axes[1,1].set_ylabel('Bullying Rate (%)')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# ===============================
# 2. ECONOMIC CORRELATIONS
# ===============================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Economic Indicators vs Bullying Rates', fontsize=16, fontweight='bold')

# GDP correlations
for i, (gender, color) in enumerate([('Total', 'blue'), ('Male', 'green'), ('Female', 'red')]):
    clean_data = df[['GDP Per Capita (USD)', gender]].dropna()
    
    axes[0,i].scatter(clean_data['GDP Per Capita (USD)'], clean_data[gender], 
                      alpha=0.6, color=color, s=50)
    axes[0,i].set_xlabel('GDP Per Capita (USD)')
    axes[0,i].set_ylabel(f'{gender} Bullying Rate (%)')
    axes[0,i].set_title(f'GDP vs {gender} Bullying')
    axes[0,i].grid(True, alpha=0.3)
    
    # Add trend line
    if len(clean_data) > 1:
        z = np.polyfit(clean_data['GDP Per Capita (USD)'], clean_data[gender], 1)
        p = np.poly1d(z)
        axes[0,i].plot(clean_data['GDP Per Capita (USD)'], p(clean_data['GDP Per Capita (USD)']), 
                       "r--", alpha=0.8)
        
        # Add correlation coefficient
        corr = clean_data['GDP Per Capita (USD)'].corr(clean_data[gender])
        axes[0,i].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[0,i].transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Poverty correlations
for i, (gender, color) in enumerate([('Total', 'blue'), ('Male', 'green'), ('Female', 'red')]):
    clean_data = df[['Poverty Rate (%)', gender]].dropna()
    
    axes[1,i].scatter(clean_data['Poverty Rate (%)'], clean_data[gender], 
                      alpha=0.6, color=color, s=50)
    axes[1,i].set_xlabel('Poverty Rate (%)')
    axes[1,i].set_ylabel(f'{gender} Bullying Rate (%)')
    axes[1,i].set_title(f'Poverty vs {gender} Bullying')
    axes[1,i].grid(True, alpha=0.3)
    
    # Add trend line
    if len(clean_data) > 1:
        z = np.polyfit(clean_data['Poverty Rate (%)'], clean_data[gender], 1)
        p = np.poly1d(z)
        axes[1,i].plot(clean_data['Poverty Rate (%)'], p(clean_data['Poverty Rate (%)']), 
                       "r--", alpha=0.8)
        
        # Add correlation coefficient
        corr = clean_data['Poverty Rate (%)'].corr(clean_data[gender])
        axes[1,i].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[1,i].transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()

# ===============================
# 3. EDUCATION SPENDING ANALYSIS
# ===============================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Education Spending vs Bullying Rates', fontsize=16, fontweight='bold')

for i, (gender, color) in enumerate([('Total', 'blue'), ('Male', 'green'), ('Female', 'red')]):
    clean_data = df[['Education Spending (% of GDP)', gender]].dropna()
    
    axes[i].scatter(clean_data['Education Spending (% of GDP)'], clean_data[gender], 
                    alpha=0.6, color=color, s=50)
    axes[i].set_xlabel('Education Spending (% of GDP)')
    axes[i].set_ylabel(f'{gender} Bullying Rate (%)')
    axes[i].set_title(f'Education Spending vs {gender} Bullying')
    axes[i].grid(True, alpha=0.3)
    
    # Add trend line
    if len(clean_data) > 1:
        z = np.polyfit(clean_data['Education Spending (% of GDP)'], clean_data[gender], 1)
        p = np.poly1d(z)
        axes[i].plot(clean_data['Education Spending (% of GDP)'], p(clean_data['Education Spending (% of GDP)']), 
                     "r--", alpha=0.8)
        
        # Add correlation coefficient
        corr = clean_data['Education Spending (% of GDP)'].corr(clean_data[gender])
        axes[i].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[i].transAxes,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()

# ===============================
# 4. CORRELATION HEATMAP
# ===============================

plt.figure(figsize=(10, 8))
correlation_cols = ['Total', 'Male', 'Female', 'GDP Per Capita (USD)', 'Poverty Rate (%)', 'Education Spending (% of GDP)']
correlation_matrix = df[correlation_cols].corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": .8}, mask=mask)
plt.title('Correlation Matrix: Bullying Rates and Economic Indicators', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ===============================
# 5. REGIONAL AND INCOME COMPARISONS
# ===============================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Regional and Income Level Analysis', fontsize=16, fontweight='bold')

# Regional averages
regional_means = df.groupby('Region')[['Total', 'Male', 'Female']].mean()
regional_means.plot(kind='bar', ax=axes[0,0], color=['blue', 'green', 'red'], alpha=0.7)
axes[0,0].set_title('Average Bullying Rates by Region')
axes[0,0].set_ylabel('Bullying Rate (%)')
axes[0,0].tick_params(axis='x', rotation=45)
axes[0,0].legend()

# Income level averages
income_order = ['Low income', 'Lower middle income', 'Upper middle income', 'High income']
income_means = df.groupby('Income Level')[['Total', 'Male', 'Female']].mean().reindex(income_order)
income_means.plot(kind='bar', ax=axes[0,1], color=['blue', 'green', 'red'], alpha=0.7)
axes[0,1].set_title('Average Bullying Rates by Income Level')
axes[0,1].set_ylabel('Bullying Rate (%)')
axes[0,1].tick_params(axis='x', rotation=45)
axes[0,1].legend()

# GDP by region
regional_gdp = df.groupby('Region')['GDP Per Capita (USD)'].mean().sort_values()
axes[1,0].bar(range(len(regional_gdp)), regional_gdp.values, color='orange', alpha=0.7)
axes[1,0].set_xticks(range(len(regional_gdp)))
axes[1,0].set_xticklabels(regional_gdp.index, rotation=45)
axes[1,0].set_title('Average GDP per Capita by Region')
axes[1,0].set_ylabel('GDP per Capita (USD)')

# Poverty by income level
income_poverty = df.groupby('Income Level')['Poverty Rate (%)'].mean().reindex(income_order)
axes[1,1].bar(range(len(income_poverty)), income_poverty.values, color='red', alpha=0.7)
axes[1,1].set_xticks(range(len(income_poverty)))
axes[1,1].set_xticklabels(income_poverty.index, rotation=45)
axes[1,1].set_title('Average Poverty Rate by Income Level')
axes[1,1].set_ylabel('Poverty Rate (%)')

plt.tight_layout()
plt.show()

# ===============================
# 6. EXTREME VALUES VISUALIZATION
# ===============================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Countries with Extreme Bullying Rates', fontsize=16, fontweight='bold')

# Top 10 and Bottom 10 for each category
categories = [('Total', 'Total Bullying'), ('Male', 'Male Bullying'), ('Female', 'Female Bullying')]

for i, (col, title) in enumerate(categories):
    clean_df = df.dropna(subset=[col])
    
    # Top 10
    top10 = clean_df.nlargest(10, col)
    axes[0,i].barh(range(len(top10)), top10[col].values, color='red', alpha=0.7)
    axes[0,i].set_yticks(range(len(top10)))
    axes[0,i].set_yticklabels([name[:15] + '...' if len(name) > 15 else name for name in top10['Country']])
    axes[0,i].set_title(f'Top 10 Highest {title}')
    axes[0,i].set_xlabel('Bullying Rate (%)')
    
    # Bottom 10
    bottom10 = clean_df.nsmallest(10, col)
    axes[1,i].barh(range(len(bottom10)), bottom10[col].values, color='green', alpha=0.7)
    axes[1,i].set_yticks(range(len(bottom10)))
    axes[1,i].set_yticklabels([name[:15] + '...' if len(name) > 15 else name for name in bottom10['Country']])
    axes[1,i].set_title(f'Top 10 Lowest {title}')
    axes[1,i].set_xlabel('Bullying Rate (%)')

plt.tight_layout()
plt.show()

# ===============================
# 7. GENDER GAP ANALYSIS
# ===============================

plt.figure(figsize=(14, 8))

# Calculate gender gap (Male - Female)
gender_data = df[['Country', 'Male', 'Female', 'Region']].dropna()
gender_data['Gender_Gap'] = gender_data['Male'] - gender_data['Female']

# Sort by gender gap
gender_data_sorted = gender_data.sort_values('Gender_Gap')

# Create color map based on whether gap is positive or negative
colors = ['red' if gap > 0 else 'blue' for gap in gender_data_sorted['Gender_Gap']]

plt.barh(range(len(gender_data_sorted)), gender_data_sorted['Gender_Gap'], color=colors, alpha=0.7)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.8)
plt.xlabel('Gender Gap (Male - Female Bullying Rate)')
plt.ylabel('Countries')
plt.title('Gender Gap in Bullying Rates by Country\n(Red: Males higher, Blue: Females higher)', fontweight='bold')

# Add legend
red_patch = plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.7, label='Males higher')
blue_patch = plt.Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.7, label='Females higher')
plt.legend(handles=[red_patch, blue_patch])

plt.tight_layout()
plt.show()

print("All visualizations have been created successfully!")
print("\nKey Charts Generated:")
print("1. Bullying Rates Distribution Analysis")
print("2. Economic Indicators vs Bullying Rates")
print("3. Education Spending vs Bullying Rates") 
print("4. Correlation Matrix Heatmap")
print("5. Regional and Income Level Analysis")
print("6. Countries with Extreme Bullying Rates")
print("7. Gender Gap Analysis")
