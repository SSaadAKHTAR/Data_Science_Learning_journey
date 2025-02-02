import pandas as pd
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

data = pd.read_csv('Task6/cleaned_dataset.csv')

def extractive_summarization(text, num_sentences = 1):
    
    # print('chk')
    # processed text with spacy
    doc= nlp(text)
    
    # tokenize
    sentences = list(doc.sents)
    
    # If text has fewer sentences than required summary length, return the text
    if len(sentences) <= num_sentences:
        return text
    
    # Compute word frequency (excluding stopwords & punctuation)
    word_frequencies = Counter()
    for token in doc:
        if token.is_stop is False and token.is_punct is False:
                word_frequencies[token.text.lower()] += 1
                
    # Score sentences based on word frequency
    sentence_score = {}
    for sent in sentences:
        sentence_score[sent] = sum(word_frequencies.get(token.text.lower(), 0 ) for token in sent)
        
    # Sort sentences by score & pick the top N
    sorted_sentences = sorted(sentence_score, key = sentence_score.get, reverse=True)
    summary_sentences = sorted(sorted_sentences[:num_sentences], key=lambda s: s.start)
    
    summary = " ".join([str(sent) for sent in summary_sentences[:num_sentences]])
    
    print("2")
    
    
    return summary

data["article"] = data["article"].fillna("")
# Applying function of extractive summarization on all aricles
print("1")
data["extractive_summarization"] = data["article"].apply(lambda x: extractive_summarization(x,num_sentences=1))
print("3")

# saving the summary 
data.to_csv('Task6/Summarized.csv', index=False)
    
    
