import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import clean_text

df = pd.read_csv('reviews.csv')

# Check how often "good" appears in each sentiment class
print("=== How often 'good' appears in each class ===")
for sentiment in ['positive', 'negative', 'neutral']:
    subset = df[df['sentiment'] == sentiment]['review_text'].str.lower()
    count = subset.str.contains(r'\bgood\b').sum()
    print(f"  {sentiment}: {count} / {len(subset)} reviews ({count/len(subset)*100:.1f}%)")

print()

# Check the top positive/negative words learned by the model
import joblib
import numpy as np

model = joblib.load('model.joblib')
tfidf  = joblib.load('tfidf.joblib')

feature_names = tfidf.get_feature_names_out()
classes = model.classes_

print("=== Top words per class (Logistic Regression coefficients) ===")
for i, cls in enumerate(classes):
    top_indices = np.argsort(model.coef_[i])[-15:]
    top_words = [(feature_names[j], model.coef_[i][j]) for j in top_indices]
    print(f"\n  [{cls.upper()}]")
    for word, coef in reversed(top_words):
        print(f"    {word:<25} {coef:.4f}")

print()
print("=== Coefficient for 'good' across all classes ===")
if 'good' in feature_names:
    idx = list(feature_names).index('good')
    for i, cls in enumerate(classes):
        print(f"  {cls}: {model.coef_[i][idx]:.4f}")
else:
    print("  'good' not in vocabulary!")
