import pandas as pd
import re
import spacy 
from spacy.lang.en.stop_words import STOP_WORDS

data_set_path = 'Task6/article_highlights.csv'

data = pd.read_csv(data_set_path)

# Drop the 'url' column
data = data.drop(columns=['url'])

# Fill NaN values with empty strings to avoid type errors
data['article'] = data['article'].fillna('')
data['highlights'] = data['highlights'].fillna('')

# Ensure all values are strings before processing
data['article'] = data['article'].astype(str)
data['highlights'] = data['highlights'].astype(str)

# Convert text to lowercase
data['article'] = data['article'].str.lower()
data['highlights'] = data['highlights'].str.lower()

# Function to clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^a-zA-Z0-9.,!? ]', '', text)  # Remove special characters
    return text.strip()

data['article'] = data['article'].apply(clean_text)
data['highlights'] = data['highlights'].apply(clean_text)

# Initialize spaCy tokenizer
nlp = spacy.load("en_core_web_sm")

def tokenize_text(text):
    doc = nlp(text)
    return [token.text for token in doc]  # Tokenize into words

data['tokenized_article'] = data['article'].apply(tokenize_text)
data['tokenized_highlights'] = data['highlights'].apply(tokenize_text)

# Function to remove stopwords
def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in STOP_WORDS]

data['filtered_article'] = data['tokenized_article'].apply(remove_stopwords)
data['filtered_highlights'] = data['tokenized_highlights'].apply(remove_stopwords)

# Save cleaned dataset
data.to_csv("Task6/cleaned_dataset.csv", index=False)
print('Dataset is processed and save')
