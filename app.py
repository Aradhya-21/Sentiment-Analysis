import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import gradio as gr
import os

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
custom_stopwords = set(stopwords.words('english')) - {'not', 'no', 'nor', 'never'}

def clean_text(text):
    """Clean and preprocess the input text."""
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in custom_stopwords]
    return ' '.join(words)

# Re-checking the notebook's clean_text
# 247:     return ' '.join(words)
# Yes, it should be ' '.join(words)

def clean_text_fixed(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in custom_stopwords]
    return ' '.join(words)

# Load and prepare the model
print("Loading data and training model...")
if os.path.exists('reviews.csv'):
    df = pd.read_csv('reviews.csv')
    # Use 'review_text' and 'sentiment' columns as seen in the notebook
    df['cleaned_review'] = df['review_text'].apply(clean_text_fixed)
    
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = tfidf.fit_transform(df['cleaned_review']).toarray()
    y = df['sentiment']
    
    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Train the model
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_resampled, y_resampled)
    print("Model training complete!")
else:
    print("Error: reviews.csv not found!")

def predict_sentiment(review):
    if not review:
        return "Please enter some text."
    cleaned = clean_text_fixed(review)
    vectorized = tfidf.transform([cleaned])
    
    # Get prediction and probabilities
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    labels = model.classes_
    
    # Create a nice output string with probabilities
    prob_dict = {label: float(prob) for label, prob in zip(labels, probabilities)}
    
    # Sort by probability for better display
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    results = "\n".join([f"{label.capitalize()}: {prob:.2%}" for label, prob in sorted_probs])
    
    return f"Verdict: {prediction.capitalize()}\n\nConfidence Scores:\n{results}"

# Create Gradio Interface
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=5, label="Enter Review", placeholder="Type your product review here..."),
    outputs=gr.Textbox(label="Analysis Results"),
    title="🌟 Sentiment Analysis App",
    description="This app classifies product reviews into **Positive**, **Negative**, or **Neutral** sentiments using a Logistic Regression model.",
    examples=[
        ["I absolutely love this product! It works perfectly."],
        ["The quality is terrible and it broke after one day."],
        ["It's a decent product. Nothing special, but it works fine."],
        ["The price is okay, but the delivery was slow."],
        ["Not what I expected, but it does the job."]
    ],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
