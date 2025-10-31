#Task1

import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files

import numpy as np
np.random.seed(42)
df = pd.DataFrame({
    'Gender': np.random.choice(['Male','Female','Other'], size=300, p=[0.48,0.48,0.04]),
    'Age': np.clip(np.random.normal(loc=30, scale=8, size=300).astype(int), 18, 65)
})

counts = df['Gender'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8,5))
ax.bar(counts.index, counts.values)          
ax.set_title('Distribution of Gender')
ax.set_xlabel('Gender')
ax.set_ylabel('Count')
fig.tight_layout()
fig.savefig('gender_bar_chart.png', dpi=200)  # saved to working dir
plt.show()
files.download('gender_bar_chart.png')

# --- 2) Histogram for a continuous column (e.g., Age) ---
fig, ax = plt.subplots(figsize=(8,5))
ax.hist(df['Age'], bins=10)                  
ax.set_title('Distribution of Age')
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')
fig.tight_layout()
fig.savefig('age_histogram.png', dpi=200)
plt.show()
