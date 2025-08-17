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






















import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Load the data
file_path = r'C:\KS\Bullying & Economic indicators.csv'
df = pd.read_csv(file_path)

# Convert numeric columns
numeric_cols = ['Total', 'Male', 'Female', 'GDP Per Capita (USD)', 'Poverty Rate (%)', 'Education Spending (% of GDP)']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("=== GDP PER CAPITA vs BULLYING RATES ANALYSIS ===")
print("Investigating: Does higher GDP per capita translate to lower bullying rates?")
print("="*70)

# ===============================
# 1. BASIC GDP-BULLYING CORRELATIONS
# ===============================

print("\n1. CORRELATION ANALYSIS")
print("-" * 40)

# Calculate correlations
gdp_total_corr = df['GDP Per Capita (USD)'].corr(df['Total'])
gdp_male_corr = df['GDP Per Capita (USD)'].corr(df['Male'])
gdp_female_corr = df['GDP Per Capita (USD)'].corr(df['Female'])

print(f"GDP Per Capita vs Bullying Correlations:")
print(f"├── Total Bullying:  r = {gdp_total_corr:.4f}")
print(f"├── Male Bullying:   r = {gdp_male_corr:.4f}")
print(f"└── Female Bullying: r = {gdp_female_corr:.4f}")

# Test statistical significance
def correlation_significance(x, y, label):
    clean_data = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(clean_data) > 2:
        corr, p_value = stats.pearsonr(clean_data['x'], clean_data['y'])
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        return corr, p_value, significance, len(clean_data)
    return None, None, None, 0

print(f"\nStatistical Significance Tests:")
corr_total, p_total, sig_total, n_total = correlation_significance(df['GDP Per Capita (USD)'], df['Total'], "Total")
corr_male, p_male, sig_male, n_male = correlation_significance(df['GDP Per Capita (USD)'], df['Male'], "Male")
corr_female, p_female, sig_female, n_female = correlation_significance(df['GDP Per Capita (USD)'], df['Female'], "Female")

print(f"├── Total (n={n_total}):  r = {corr_total:.4f}, p = {p_total:.4f} {sig_total}")
print(f"├── Male (n={n_male}):   r = {corr_male:.4f}, p = {p_male:.4f} {sig_male}")
print(f"└── Female (n={n_female}): r = {corr_female:.4f}, p = {p_female:.4f} {sig_female}")

# ===============================
# 2. GDP QUARTILE ANALYSIS
# ===============================

print(f"\n2. GDP QUARTILE ANALYSIS")
print("-" * 40)

# Create GDP quartiles
gdp_clean = df.dropna(subset=['GDP Per Capita (USD)', 'Total'])
gdp_clean['GDP_Quartile'] = pd.qcut(gdp_clean['GDP Per Capita (USD)'], 4, 
                                   labels=['Q1 (Lowest)', 'Q2 (Low-Mid)', 'Q3 (Mid-High)', 'Q4 (Highest)'])

quartile_stats = gdp_clean.groupby('GDP_Quartile').agg({
    'GDP Per Capita (USD)': ['count', 'mean', 'min', 'max'],
    'Total': ['mean', 'std'],
    'Male': ['mean', 'std'], 
    'Female': ['mean', 'std']
}).round(2)

print("GDP Quartile Statistics:")
print(quartile_stats)

# ANOVA test for significant differences between quartiles
quartile_groups_total = [group['Total'].values for name, group in gdp_clean.groupby('GDP_Quartile')]
quartile_groups_male = [group['Male'].dropna().values for name, group in gdp_clean.groupby('GDP_Quartile')]
quartile_groups_female = [group['Female'].dropna().values for name, group in gdp_clean.groupby('GDP_Quartile')]

f_total, p_total = stats.f_oneway(*quartile_groups_total)
f_male, p_male = stats.f_oneway(*[g for g in quartile_groups_male if len(g) > 0])
f_female, p_female = stats.f_oneway(*[g for g in quartile_groups_female if len(g) > 0])

print(f"\nANOVA Tests (differences between GDP quartiles):")
print(f"├── Total Bullying:  F = {f_total:.3f}, p = {p_total:.4f}")
print(f"├── Male Bullying:   F = {f_male:.3f}, p = {p_male:.4f}")
print(f"└── Female Bullying: F = {f_female:.3f}, p = {p_female:.4f}")

# ===============================
# 3. INCOME LEVEL ANALYSIS
# ===============================

print(f"\n3. INCOME LEVEL ANALYSIS")
print("-" * 40)

income_order = ['Low income', 'Lower middle income', 'Upper middle income', 'High income']
income_analysis = df.groupby('Income Level').agg({
    'GDP Per Capita (USD)': ['count', 'mean', 'median'],
    'Total': ['mean', 'median', 'std'],
    'Male': ['mean', 'median', 'std'],
    'Female': ['mean', 'median', 'std']
}).round(2)

print("Bullying Rates by Income Level:")
for income in income_order:
    if income in income_analysis.index:
        row = income_analysis.loc[income]
        gdp_mean = row[('GDP Per Capita (USD)', 'mean')]
        total_mean = row[('Total', 'mean')]
        male_mean = row[('Male', 'mean')]
        female_mean = row[('Female', 'mean')]
        count = int(row[('GDP Per Capita (USD)', 'count')])
        
        print(f"\n{income} (n={count}):")
        print(f"├── Avg GDP: ${gdp_mean:,.0f}")
        print(f"├── Total Bullying: {total_mean:.1f}% (±{row[('Total', 'std')]:.1f})")
        print(f"├── Male Bullying: {male_mean:.1f}% (±{row[('Male', 'std')]:.1f})")
        print(f"└── Female Bullying: {female_mean:.1f}% (±{row[('Female', 'std')]:.1f})")

# ===============================
# 4. REGIONAL GDP-BULLYING PATTERNS
# ===============================

print(f"\n4. REGIONAL PATTERNS")
print("-" * 40)

regional_analysis = df.groupby('Region').agg({
    'GDP Per Capita (USD)': ['count', 'mean', 'median'],
    'Total': ['mean', 'std'],
    'Male': ['mean', 'std'],
    'Female': ['mean', 'std']
}).round(2)

print("Regional GDP and Bullying Analysis:")
for region in regional_analysis.index:
    row = regional_analysis.loc[region]
    gdp_mean = row[('GDP Per Capita (USD)', 'mean')]
    total_mean = row[('Total', 'mean')]
    count = int(row[('GDP Per Capita (USD)', 'count')])
    
    print(f"\n{region} (n={count}):")
    print(f"├── Avg GDP: ${gdp_mean:,.0f}")
    print(f"└── Avg Total Bullying: {total_mean:.1f}%")

# Calculate regional correlations
print(f"\nRegional GDP-Bullying Correlations:")
for region in df['Region'].unique():
    region_data = df[df['Region'] == region]
    if len(region_data.dropna(subset=['GDP Per Capita (USD)', 'Total'])) > 3:
        corr = region_data['GDP Per Capita (USD)'].corr(region_data['Total'])
        n = len(region_data.dropna(subset=['GDP Per Capita (USD)', 'Total']))
        print(f"├── {region[:25]:<25}: r = {corr:>6.3f} (n={n})")

# ===============================
# 5. OUTLIER ANALYSIS
# ===============================

print(f"\n5. OUTLIER ANALYSIS")
print("-" * 40)

# High GDP, High Bullying
high_gdp = df[df['GDP Per Capita (USD)'] > df['GDP Per Capita (USD)'].quantile(0.75)]
high_gdp_high_bullying = high_gdp[high_gdp['Total'] > high_gdp['Total'].median()]

print("High GDP + High Bullying Countries (Unexpected):")
if not high_gdp_high_bullying.empty:
    for _, country in high_gdp_high_bullying.iterrows():
        print(f"├── {country['Country']}: GDP ${country['GDP Per Capita (USD)']:,.0f}, Bullying {country['Total']:.1f}%")
else:
    print("├── No clear outliers found")

# Low GDP, Low Bullying  
low_gdp = df[df['GDP Per Capita (USD)'] < df['GDP Per Capita (USD)'].quantile(0.25)]
low_gdp_low_bullying = low_gdp[low_gdp['Total'] < low_gdp['Total'].median()]

print(f"\nLow GDP + Low Bullying Countries (Positive outliers):")
if not low_gdp_low_bullying.empty:
    for _, country in low_gdp_low_bullying.iterrows():
        print(f"├── {country['Country']}: GDP ${country['GDP Per Capita (USD)']:,.0f}, Bullying {country['Total']:.1f}%")
else:
    print("├── No clear outliers found")

# ===============================
# 6. REGRESSION ANALYSIS
# ===============================

print(f"\n6. REGRESSION ANALYSIS")
print("-" * 40)

def perform_regression(x, y, label):
    """Perform linear regression and return results"""
    clean_data = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(clean_data) < 3:
        return None
    
    X = clean_data[['x']]
    y_clean = clean_data['y']
    
    model = LinearRegression()
    model.fit(X, y_clean)
    
    y_pred = model.predict(X)
    r2 = r2_score(y_clean, y_pred)
    
    return {
        'slope': model.coef_[0],
        'intercept': model.intercept_,
        'r2': r2,
        'n': len(clean_data)
    }

# Perform regressions
regression_results = {}
for bullying_type in ['Total', 'Male', 'Female']:
    result = perform_regression(df['GDP Per Capita (USD)'], df[bullying_type], bullying_type)
    if result:
        regression_results[bullying_type] = result

print("Linear Regression Results (Bullying = slope × GDP + intercept):")
for bullying_type, results in regression_results.items():
    slope = results['slope']
    intercept = results['intercept']
    r2 = results['r2']
    n = results['n']
    
    # Calculate practical impact
    gdp_change = 10000  # $10,000 increase in GDP
    bullying_change = slope * gdp_change
    
    print(f"\n{bullying_type} Bullying (n={n}):")
    print(f"├── Equation: Bullying = {slope:.6f} × GDP + {intercept:.2f}")
    print(f"├── R² = {r2:.4f} ({r2*100:.1f}% variance explained)")
    print(f"└── Impact: $10k GDP increase → {bullying_change:.2f}% change in bullying")

# ===============================
# 7. GENDER-SPECIFIC FINDINGS
# ===============================

print(f"\n7. GENDER-SPECIFIC ANALYSIS")
print("-" * 40)

# Compare male vs female GDP correlations
gender_comparison = df[['GDP Per Capita (USD)', 'Male', 'Female']].dropna()
if not gender_comparison.empty:
    male_gdp_corr = gender_comparison['GDP Per Capita (USD)'].corr(gender_comparison['Male'])
    female_gdp_corr = gender_comparison['GDP Per Capita (USD)'].corr(gender_comparison['Female'])
    
    print(f"GDP Impact Comparison:")
    print(f"├── Males:   r = {male_gdp_corr:.4f}")
    print(f"└── Females: r = {female_gdp_corr:.4f}")
    
    if abs(male_gdp_corr) > abs(female_gdp_corr):
        stronger_gender = "males"
        difference = abs(male_gdp_corr) - abs(female_gdp_corr)
    else:
        stronger_gender = "females"
        difference = abs(female_gdp_corr) - abs(male_gdp_corr)
    
    print(f"\nGDP has a stronger correlation with bullying among {stronger_gender}")
    print(f"Difference in correlation strength: {difference:.4f}")

# Gender gap analysis by GDP level
gender_comparison['Gender_Gap'] = gender_comparison['Male'] - gender_comparison['Female']
gdp_gender_corr = gender_comparison['GDP Per Capita (USD)'].corr(gender_comparison['Gender_Gap'])

print(f"\nGender Gap Analysis:")
print(f"├── GDP vs Gender Gap correlation: r = {gdp_gender_corr:.4f}")
if gdp_gender_corr > 0:
    print(f"└── Higher GDP → Males have relatively higher bullying rates")
elif gdp_gender_corr < 0:
    print(f"└── Higher GDP → Females have relatively higher bullying rates")
else:
    print(f"└── No clear relationship between GDP and gender gap")

# ===============================
# 8. KEY INSIGHTS SUMMARY
# ===============================

print(f"\n8. KEY INSIGHTS & CONCLUSIONS")
print("=" * 50)

print(f"\n📊 CORRELATION STRENGTH:")
if abs(gdp_total_corr) > 0.5:
    strength = "Strong"
elif abs(gdp_total_corr) > 0.3:
    strength = "Moderate"
elif abs(gdp_total_corr) > 0.1:
    strength = "Weak"
else:
    strength = "Very weak"

direction = "negative" if gdp_total_corr < 0 else "positive"
print(f"├── {strength} {direction} correlation (r = {gdp_total_corr:.3f})")

print(f"\n💡 MAIN FINDINGS:")
if gdp_total_corr < -0.2:
    print(f"├── ✅ Higher GDP generally associated with LOWER bullying rates")
elif gdp_total_corr > 0.2:
    print(f"├── ⚠️  Higher GDP associated with HIGHER bullying rates (unexpected)")
else:
    print(f"├── ❌ No clear relationship between GDP and bullying rates")

# Statistical significance summary
if p_total < 0.001:
    print(f"├── ✅ Relationship is highly statistically significant (p < 0.001)")
elif p_total < 0.05:
    print(f"├── ✅ Relationship is statistically significant (p < 0.05)")
else:
    print(f"├── ❌ Relationship is not statistically significant (p = {p_total:.3f})")

print(f"\n🚻 GENDER DIFFERENCES:")
if abs(male_gdp_corr - female_gdp_corr) > 0.1:
    if abs(male_gdp_corr) > abs(female_gdp_corr):
        print(f"├── GDP has stronger impact on MALE bullying rates")
    else:
        print(f"├── GDP has stronger impact on FEMALE bullying rates")
else:
    print(f"├── GDP impact is similar for both genders")

print(f"\n📈 PRACTICAL IMPLICATIONS:")
if 'Total' in regression_results:
    slope = regression_results['Total']['slope']
    impact_10k = slope * 10000
    if abs(impact_10k) > 1:
        print(f"├── Every $10,000 GDP increase → {impact_10k:.1f}% change in bullying")
    else:
        print(f"├── GDP changes have minimal practical impact on bullying rates")

print(f"\n🎯 RECOMMENDATIONS FOR KS CONSULTING:")
print(f"├── Focus analysis on income-level and regional patterns")
print(f"├── Investigate cultural and policy factors beyond economic indicators")
print(f"├── Consider non-linear relationships and interaction effects")
print(f"└── Explore education spending and poverty rates as stronger predictors")

print(f"\n" + "="*70)
print("GDP ANALYSIS COMPLETE - Run poverty and education scripts for full picture")
print("="*70)
