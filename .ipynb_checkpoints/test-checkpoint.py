# ===============================
# SMARTSHINE CAR WASH DATA PREPROCESSING
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# --------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------
df = pd.read_csv("smartshine_carwash_messy_data.csv")

print("Initial Dataset Info:\n")
print(df.info(), "\n")
print("Sample Records:\n", df.head(), "\n")

# --------------------------------------------------------------------
# BEFORE CLEANING VISUALIZATION (Bar chart + sample heatmap)
# --------------------------------------------------------------------
missing_before = df.isna().sum()

# Bar chart for missing values before cleaning
missing_before[missing_before > 0].plot(kind='bar', figsize=(8,4), color='red')
plt.title("Before Cleaning - Missing Values per Column")
plt.ylabel("Count of Missing Entries")
plt.tight_layout()
plt.show()

# Optional: Heatmap sample (first 50 rows)
plt.figure(figsize=(8,4))
sns.heatmap(df.head(50).isna(), cbar=False, cmap='Reds')
plt.title("Before Cleaning - Missing Value Pattern (First 50 Rows)")
plt.show()

# --------------------------------------------------------------------
# 2. DATA CLEANING
# --------------------------------------------------------------------
# Fill missing values safely
df['customer_rating'] = df['customer_rating'].fillna(df['customer_rating'].median())
df['service_duration'] = df['service_duration'].fillna(df['service_duration'].mean())
df['payment_method'] = df['payment_method'].fillna('Unknown')

# Convert service_date (auto-detect mixed formats)
df['service_date'] = pd.to_datetime(df['service_date'], format='mixed', errors='coerce')

# Remove duplicates
df.drop_duplicates(inplace=True)

# Standardize text columns
text_cols = ['vehicle_type', 'service_type', 'payment_method']
df[text_cols] = df[text_cols].apply(lambda col: col.astype(str).str.title())

# --------------------------------------------------------------------
# AFTER CLEANING VISUALIZATION
# --------------------------------------------------------------------
missing_after = df.isna().sum()
missing_df = pd.DataFrame({
    'Before Cleaning': missing_before,
    'After Cleaning': missing_after
})
missing_df.plot(kind='bar', figsize=(10,5), rot=45)
plt.title('Missing Values per Column: Before vs After Cleaning')
plt.ylabel('Count of Missing Entries')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
sns.heatmap(df.head(50).isna(), cbar=False, cmap='Greens')
plt.title("After Cleaning - Missing Value Pattern (First 50 Rows)")
plt.show()

# --------------------------------------------------------------------
# 3. DATA INTEGRATION
# --------------------------------------------------------------------
df['service_year'] = df['service_date'].dt.year
df['service_month'] = df['service_date'].dt.month
df['day_of_week'] = df['service_date'].dt.day_name()

# Customer visit frequency
visit_freq = df.groupby('customer_id')['service_date'].count().rename('visit_frequency')
df = df.merge(visit_freq, on='customer_id', how='left')
df.head()

# Visualization: Monthly service volume
monthly = df.groupby('service_month')['service_cost'].count()
monthly.plot(kind='bar', figsize=(8,4), color='skyblue')
plt.title('Number of Services per Month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------
# 4. DATA REDUCTION
# --------------------------------------------------------------------
#Use a correlation matrix + heatmap to detect features that may
# be redundant or highly correlated, which we might reduce or drop.
corr = df.select_dtypes(include=np.number).corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix (Data Reduction Check)')
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------
# 5. DATA TRANSFORMATION
# --------------------------------------------------------------------
# Encode categorical text columns
for col in text_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Define numeric columns
num_cols = ['service_cost', 'service_duration', 'customer_rating', 'visit_frequency']

# Convert stray text to numeric (safe)
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

# Scale numeric features
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Visualization: distribution after scaling
df[num_cols].hist(figsize=(10,6))
plt.suptitle('Numerical Feature Distributions After Scaling')
plt.tight_layout()
plt.show()
'''### 🔹 Step 5: Data Transformation

Here, we:
- Encoded categorical columns using `LabelEncoder()`
- Scaled numeric columns to 0–1 using `MinMaxScaler`
- Visualized the distributions to check for skew, imbalance, or issues

👉 This ensures all data is model-ready and numerically balanced — essential for gradient-based or distance-based ML algorithms.'''
# --------------------------------------------------------------------
# 6. DATA DISCRETIZATION
# --------------------------------------------------------------------
df['cost_tier'] = pd.qcut(df['service_cost'], q=3, labels=['Low','Medium','High'])
df['loyalty_tier'] = pd.qcut(df['visit_frequency'], q=3, labels=['Occasional','Regular','Frequent'])

sns.countplot(data=df, x='cost_tier', hue='loyalty_tier')
plt.title('Customer Segmentation by Cost and Loyalty')
plt.xlabel('Service Cost Tier')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------
# 7. DATA AUGMENTATION
# --------------------------------------------------------------------
aug_df = df.copy()
noise = np.random.normal(0, 0.02, aug_df[num_cols].shape)
aug_df[num_cols] = np.clip(aug_df[num_cols] + noise, 0, 1)
augmented_df = pd.concat([df, aug_df], ignore_index=True)

plt.figure(figsize=(6,4))
sns.kdeplot(df['service_cost'], label='Original')
sns.kdeplot(augmented_df['service_cost'], label='Augmented')
plt.legend()
plt.title('Service Cost Distribution: Original vs Augmented')
plt.tight_layout()
plt.show()

print(f"Original dataset size: {len(df)}")
print(f"Augmented dataset size: {len(augmented_df)}")
