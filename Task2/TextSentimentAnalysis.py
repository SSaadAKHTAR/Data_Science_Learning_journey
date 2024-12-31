import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download necessary NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

dataset_path = 'Task2/MovieReviewTrainingDatabase.csv'
data = pd.read_csv(dataset_path)
print(data.head)


def tokenize_text(text):
    return word_tokenize(text)


stop_words = set(stopwords.words('english'))


def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in stop_words]


lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]


def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text


def preprocess_text(text):
    text = clean_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return ' '.join(tokens)


data['processed_review'] = data['review'].apply(preprocess_text)


print(data[['review', 'processed_review']].head())