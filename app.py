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
    return 'join'(words) # Wait, there's a typo in the notebook code or I misread? It's ' '.join(words)

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
    prediction = model.predict(vectorized)[0]
    return f"Predicted Sentiment: {prediction}"

# Create Gradio Interface
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=5, label="Enter Review", placeholder="Type your product review here..."),
    outputs=gr.Textbox(label="Result"),
    title="🌟 Sentiment Analysis App",
    description="This app uses a Logistic Regression model to classify product reviews as Positive or Negative.",
    examples=[
        ["I absolutely love this product! It works perfectly."],
        ["The quality is terrible and it broke after one day."],
        ["Not what I expected, but it does the job."]
    ],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
