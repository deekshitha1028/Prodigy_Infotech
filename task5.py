# -----------------------------------------------------------
# Task-05: Traffic Accident Data Analysis (FARS 2016 Format)
# Internship: Prodigy InfoTech
# Dataset: acc_16.csv.zip
# -----------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os

# ---------------- STEP 1: Extract & Load Dataset ----------------
zip_file = "acc_16.csv.zip"

if not os.path.exists("acc_16.csv"):
    with zipfile.ZipFile(zip_file, 'r') as z:
        z.extractall()
        print("✅ Zip file extracted successfully!")

# Load dataset
df = pd.read_csv("acc_16.csv", low_memory=False)
print("✅ Dataset Loaded Successfully!")
print(f"Shape: {df.shape}")
print("First 10 columns:", list(df.columns[:10]))

# ---------------- STEP 2: Select and Clean Key Columns ----------------
columns = ['URBANICITY', 'WEATHR_IM', 'LGTCON_IM', 'MAXSEV_IM', 'VE_TOTAL', 'REGION']
df = df[columns].dropna()

print("\n✅ Selected columns for analysis:", columns)
print(df.head())

# ---------------- STEP 3: Map Encoded Columns ----------------
# URBANICITY (1=Rural, 2=Urban)
urban_map = {1: 'Rural', 2: 'Urban'}
df['URBANICITY'] = df['URBANICITY'].map(urban_map)

# Weather condition mapping (simplified example from NHTSA FARS)
weather_map = {
    1: 'Clear', 2: 'Rain', 3: 'Sleet/Hail', 4: 'Snow',
    5: 'Fog/Smog', 6: 'Severe Wind', 7: 'Blowing Sand/Dust',
    8: 'Other', 9: 'Unknown'
}
df['WEATHR_IM'] = df['WEATHR_IM'].map(weather_map).fillna('Unknown')

# Light condition mapping
light_map = {
    1: 'Daylight', 2: 'Dark - Lighted', 3: 'Dark - Unlighted',
    4: 'Dawn', 5: 'Dusk', 6: 'Other', 7: 'Unknown'
}
df['LGTCON_IM'] = df['LGTCON_IM'].map(light_map).fillna('Unknown')

# Region mapping (simplified based on FARS regions)
region_map = {
    1: 'Northeast', 2: 'Midwest', 3: 'South', 4: 'West'
}
df['REGION'] = df['REGION'].map(region_map).fillna('Unknown')

# ---------------- STEP 4: Visualization ----------------
sns.set(style="whitegrid")

# 1️⃣ Urban vs Rural Accidents
plt.figure(figsize=(6,4))
sns.countplot(x='URBANICITY', data=df, palette='Set2')
plt.title('🚗 Accidents by Area Type')
plt.xlabel('Area')
plt.ylabel('Number of Accidents')
plt.tight_layout()
plt.show()

# 2️⃣ Weather Conditions
plt.figure(figsize=(8,5))
sns.countplot(y='WEATHR_IM', data=df, order=df['WEATHR_IM'].value_counts().index, palette='coolwarm')
plt.title('🌦️ Accidents by Weather Condition')
plt.xlabel('Number of Accidents')
plt.ylabel('Weather Condition')
plt.tight_layout()
plt.show()

# 3️⃣ Light Conditions
plt.figure(figsize=(8,5))
sns.countplot(y='LGTCON_IM', data=df, order=df['LGTCON_IM'].value_counts().index, palette='magma')
plt.title('💡 Accidents by Light Condition')
plt.xlabel('Number of Accidents')
plt.ylabel('Light Condition')
plt.tight_layout()
plt.show()

# 4️⃣ Accident Severity Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='MAXSEV_IM', data=df, palette='plasma')
plt.title('⚠️ Distribution of Accident Severity')
plt.xlabel('Severity Level Code')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 5️⃣ Vehicles Involved
plt.figure(figsize=(6,4))
sns.histplot(df['VE_TOTAL'], bins=15, kde=True, color='skyblue')
plt.title('🚘 Distribution of Number of Vehicles per Accident')
plt.xlabel('Number of Vehicles')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 6️⃣ Accidents by U.S. Region
plt.figure(figsize=(7,4))
sns.countplot(x='REGION', data=df, palette='viridis')
plt.title('🗺️ Accidents by U.S. Region')
plt.xlabel('Region')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

print("✅ All Visualizations Complete! 🚦")
