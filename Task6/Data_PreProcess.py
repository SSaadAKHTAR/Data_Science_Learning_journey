import pandas as pd
import re
import spacy 

data_set_path = 'Task6/article_highlights.csv'

data = pd.read_csv(data_set_path)

data = data.drop(columns=['url'])

data['article'] = data['article'].str.lower()
data['highlights'] = data['highlights'].str.lower()

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^a-zA-Z0-9.,!? ]', '', text)  # Remove special characters
    return text.strip()

data['article'] = data['article'].apply(clean_text)
data['highlights'] = data['highlights'].apply(clean_text)

#spacy for tokenizing
nlp = spacy.load("en_core_web_sm")
def tokenize_text(text):
    doc = nlp(text)
    return [token.text for token in doc]

data['tokenized_article'] = data['article'].apply(tokenize_text)
data['tokenized_highlights'] = data['highlights'].apply(tokenize_text)


