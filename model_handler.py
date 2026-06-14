import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from nltk.sentiment import SentimentIntensityAnalyzer
from preprocessing import clean_text

# VADER maps to our 3 class labels
CLASSES = ['negative', 'neutral', 'positive']

# -------------------------------------------------------------------
# Extend VADER's lexicon with common words it misses or gets wrong.
# Scores use the same -4 to +4 scale as the VADER lexicon.
# -------------------------------------------------------------------
_CUSTOM_LEXICON = {
    # Clearly negative words missing from VADER
    'garbage':       -3.2,
    'trash':         -2.8,
    'junk':          -2.5,
    'useless':       -3.0,
    'worthless':     -3.2,
    'overpriced':    -2.5,
    'defective':     -2.8,
    'broke':         -2.0,
    'broken':        -2.5,
    'mediocre':      -1.8,
    'subpar':        -2.2,
    'disappointing': -2.3,
    'cheap':         -1.5,
    'flimsy':        -2.0,
    'waste':         -2.5,
    # Clearly positive words to reinforce
    'flawless':      +3.2,
    'outstanding':   +3.0,
    'superb':        +3.1,
    'excellent':     +3.0,
    'fantastic':     +3.0,
    'amazing':       +3.0,
    'awesome':       +3.0,
    'reliable':      +2.5,
    'solid':         +2.2,
    # Neutral/hedging phrases — override strong positives that follow them
    'mediocre at':   -2.0,
    'at best':       -1.5,
}

# Singleton VADER instance with extended lexicon
_sia = SentimentIntensityAnalyzer()
_sia.lexicon.update(_CUSTOM_LEXICON)


def _vader_probs(text: str) -> dict:
    """
    Use VADER to get sentiment probabilities.
    VADER compound score:
      >= 0.05  → positive
      <= -0.05 → negative
      else     → neutral
    We convert the raw pos/neg/neu scores into a probability dict.
    """
    scores = _sia.polarity_scores(text)   # keys: neg, neu, pos, compound

    pos = scores['pos']
    neg = scores['neg']
    neu = scores['neu']
    total = pos + neg + neu

    if total == 0:
        return {'negative': 1 / 3, 'neutral': 1 / 3, 'positive': 1 / 3}

    return {
        'positive': pos / total,
        'negative': neg / total,
        'neutral':  neu / total,
    }


class SentimentModel:
    def __init__(self, data_path='reviews.csv', model_dir=None):
        self.data_path = data_path
        self.model = None
        self.tfidf = None
        self.is_trained = False

        if model_dir:
            model_path = os.path.join(model_dir, 'model.joblib')
            tfidf_path = os.path.join(model_dir, 'tfidf.joblib')
            if os.path.exists(model_path) and os.path.exists(tfidf_path):
                import joblib
                self.model = joblib.load(model_path)
                self.tfidf = joblib.load(tfidf_path)
                self.is_trained = True
                print(f"Loaded pre-trained model and tfidf from {model_dir}")
                return

        self.train()


    def train(self):
        print("Loading data and training model...")
        if not os.path.exists(self.data_path):
            print(f"Error: {self.data_path} not found!")
            return

        df = pd.read_csv(self.data_path)
        df['cleaned_review'] = df['review_text'].apply(clean_text)

        # Vectorization
        self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = self.tfidf.fit_transform(df['cleaned_review']).toarray()
        y = df['sentiment']

        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Train model
        self.model = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.model.fit(X_resampled, y_resampled)
        self.is_trained = True
        print("Model training complete!")

    def predict(self, review: str):
        if not self.is_trained:
            return "Model not trained.", {}

        cleaned    = clean_text(review)
        vectorized = self.tfidf.transform([cleaned])

        ml_probs_arr = self.model.predict_proba(vectorized)[0]
        ml_labels    = self.model.classes_
        ml_probs     = {label: float(p) for label, p in zip(ml_labels, ml_probs_arr)}

        # ----------------------------------------------------------------
        # Sparsity check: if the TF-IDF vector is almost entirely zero,
        # the ML model has no real signal — it's just guessing from bias.
        # In that case, trust VADER much more heavily.
        # ----------------------------------------------------------------
        nonzero_ratio = np.count_nonzero(vectorized.toarray()) / vectorized.shape[1]

        vader_scores = _sia.polarity_scores(review)
        vader_probs  = _vader_probs(review)
        vader_confidence = abs(vader_scores['compound'])  # 0.0 (uncertain) → 1.0 (very sure)

        # ----------------------------------------------------------------
        # Dynamic weight:
        #   - Sparse TF-IDF vector → rely mainly on VADER
        #   - VADER compound is strong (|score| > 0.5) → trust VADER more
        #   - VADER compound is weak → trust ML more
        # ----------------------------------------------------------------
        if nonzero_ratio < 0.001:
            vader_weight = 0.85          # almost no vocabulary match
        elif vader_confidence >= 0.5:
            vader_weight = 0.60          # VADER is very confident
        elif vader_confidence >= 0.2:
            vader_weight = 0.40          # VADER is moderately confident
        else:
            vader_weight = 0.20          # VADER is uncertain; trust ML

        ml_weight = 1.0 - vader_weight

        # Blend probabilities
        blended = {}
        for cls in CLASSES:
            blended[cls] = ml_weight * ml_probs.get(cls, 0.0) + \
                           vader_weight * vader_probs.get(cls, 0.0)

        # Normalise so probabilities sum to 1
        total = sum(blended.values())
        blended = {k: v / total for k, v in blended.items()}

        prediction = max(blended, key=blended.get)
        return prediction, blended
