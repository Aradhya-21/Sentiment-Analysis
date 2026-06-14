import re
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set NLTK data path to the local nltk_data folder inside the repository
nltk_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'nltk_data'))
nltk.data.path.insert(0, nltk_data_dir)

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
try:
    custom_stopwords = set(stopwords.words('english')) - {'not', 'no', 'nor', 'never'}
except Exception as e:
    print(f"Warning: Failed to load stopwords from {nltk_data_dir}: {e}")
    custom_stopwords = set()

def clean_text(text):
    """Clean and preprocess the input text."""
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in custom_stopwords]
    return ' '.join(words)
