import re
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set NLTK data path for Vercel (writable /tmp directory)
nltk_data_dir = os.path.join('/tmp', 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Download NLTK resources to the specific path
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir=nltk_data_dir)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', download_dir=nltk_data_dir)

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
try:
    custom_stopwords = set(stopwords.words('english')) - {'not', 'no', 'nor', 'never'}
except Exception:
    custom_stopwords = set() # Fallback if stopwords fail to load

def clean_text(text):
    """Clean and preprocess the input text."""
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in custom_stopwords]
    return ' '.join(words)
