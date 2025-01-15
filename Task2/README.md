# Task 2: Text Sentiment Analysis

## Description
This task involves building a **sentiment analysis model** to classify text reviews (e.g., from the IMDB Reviews dataset) into positive or negative sentiments. The focus is on text preprocessing, feature engineering, model training, and evaluation.

---

## Steps

### 1. Text Preprocessing
- **Tokenization:**
  - Split text into individual words for further processing.
- **Stopwords Removal:**
  - Removed common words (e.g., "the", "is") using NLTK’s predefined list.
- **Lemmatization:**
  - Normalized text by converting words to their base forms (e.g., "running" to "run").

### 2. Feature Engineering
- **TF-IDF:**
  - Transformed text into numerical format using Term Frequency-Inverse Document Frequency.

### 3. Model Training
- **Algorithms Used:**
  - Trained models like **Logistic Regression** and **Naive Bayes** to classify sentiments.

### 4. Model Evaluation
- Evaluated performance using:
  - **Precision, Recall, F1-Score:** Key metrics to assess model accuracy.

---

### Model Performance
- **Classification Report:**
  ![Confusion Matrix](./Task2/screenshots/image.png)


## Outcome
- A Python script that:
  - Processes input text for prediction.
  - Classifies sentiments as positive or negative.
  - Outputs evaluation metrics.
