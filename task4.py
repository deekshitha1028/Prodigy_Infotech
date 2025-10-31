# Task-04: Sentiment Analysis & Visualization
# Prodigy InfoTech Internship

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import seaborn as sns
import zipfile
import os

# -------------------- STEP 1: Load Dataset --------------------
# Unzip the uploaded file
zip_file_path = 'Twitter_Data.csv.zip'
extract_path = 'extracted_data'
os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Assuming the CSV file inside the zip is named 'Twitter_Data.csv'
csv_file_path = os.path.join(extract_path, 'Twitter_Data.csv')

df = pd.read_csv(csv_file_path, encoding='latin-1')
print("Dataset Loaded Successfully ✅")
print(df.head())

# -------------------- STEP 2: Data Cleaning --------------------
df.dropna(subset=['clean_text'], inplace=True)
df['clean_text'] = df['clean_text'].astype(str)

# -------------------- STEP 3: Sentiment Analysis --------------------
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

df['Sentiment'] = df['clean_text'].apply(get_sentiment)
print(df['Sentiment'].value_counts())

# -------------------- STEP 4: Visualization --------------------

# Sentiment distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Sentiment', data=df, palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Type')
plt.ylabel('Count')
plt.show()

# WordCloud for positive tweets
positive_text = ' '.join(df[df['Sentiment']=='Positive']['clean_text'])
wc = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
plt.figure(figsize=(8,4))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud for Positive Tweets")
plt.show()

# WordCloud for negative tweets
negative_text = ' '.join(df[df['Sentiment']=='Negative']['clean_text'])
wc = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
plt.figure(figsize=(8,4))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud for Negative Tweets")
plt.show()

# WordCloud for neutral tweets
neutral_text = ' '.join(df[df['Sentiment']=='Neutral']['clean_text'])
wc = WordCloud(width=800, height=400, background_color='white').generate(neutral_text)
plt.figure(figsize=(8,4))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud for Neutral Tweets")
plt.show()
