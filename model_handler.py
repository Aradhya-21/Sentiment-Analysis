import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from preprocessing import clean_text

class SentimentModel:
    def __init__(self, data_path='reviews.csv'):
        self.data_path = data_path
        self.model = None
        self.tfidf = None
        self.is_trained = False
        self.train()

    def train(self):
        print("Loading data and training model...")
        if not os.path.exists(self.data_path):
            print(f"Error: {self.data_path} not found!")
            return

        df = pd.read_csv(self.data_path)
        # Preprocess text
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

    def predict(self, review):
        if not self.is_trained:
            return "Model not trained.", {}
        
        cleaned = clean_text(review)
        vectorized = self.tfidf.transform([cleaned])
        
        prediction = self.model.predict(vectorized)[0]
        probabilities = self.model.predict_proba(vectorized)[0]
        labels = self.model.classes_
        
        prob_dict = {label: float(prob) for label, prob in zip(labels, probabilities)}
        return prediction, prob_dict
