# Data pre processing
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download necessary NLTK data
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

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



# Feature engeeniring
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['processed_review']).toarray()
print("Shape of TF-IDF matrix:", X.shape)


y=data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)


model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# for custom inputs
custom_input = "I love this product! It's amazing. but some times its bad as well"
custom_input_transformed = preprocess_text(custom_input)
custom_input_tfidf = tfidf.transform([custom_input_transformed]).toarray()
y_pred_custom = model.predict(custom_input_tfidf)
print(f"Predicted sentiment: {y_pred_custom}")


