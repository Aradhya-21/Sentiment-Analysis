from preprocessing import clean_text
from model_handler import SentimentModel

sm = SentimentModel()

test_phrases = [
    'good product',
    'great product',
    'bad product',
    'excellent quality',
    'terrible',
    'okay product',
    'not good',
    'amazing',
]

print("\n=== MODEL PREDICTIONS ===")
for phrase in test_phrases:
    cleaned = clean_text(phrase)
    pred, probs = sm.predict(phrase)
    print(f'Input: "{phrase}"')
    print(f'  Cleaned : "{cleaned}"')
    print(f'  Verdict : {pred}')
    print(f'  Probs   : { {k: f"{v:.2%}" for k,v in probs.items()} }')
    print()

print("=== CLASS DISTRIBUTION IN TRAINING DATA ===")
import pandas as pd
df = pd.read_csv('reviews.csv')
print(df['sentiment'].value_counts())

print("\n=== SAMPLE POSITIVE REVIEWS IN DATASET ===")
pos = df[df['sentiment'] == 'positive']['review_text'].head(5)
for r in pos:
    print(' -', r[:80])

print("\n=== SAMPLE NEGATIVE REVIEWS IN DATASET ===")
neg = df[df['sentiment'] == 'negative']['review_text'].head(5)
for r in neg:
    print(' -', r[:80])
