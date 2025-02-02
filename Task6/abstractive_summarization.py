import pandas as pd
from transformers import pipeline

data = pd.read_csv('Task6/cleaned_dataset.csv')

# loading Bart for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def abstractive_summary(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""  # Handle empty or NaN values
    
    input_length = len(text.split())
    max_length = max(30, int(input_length * 0.4))  # Minimum 30 words
    min_length = max(10, int(input_length * 0.2))  # Minimum 10 words
    
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    
    return summary[0]["summary_text"]

data["abstractive_summary"] = data["article"].apply(lambda x: abstractive_summary(x))
data.to_csv('Task6/abstractive_summary.csv')
print("done")