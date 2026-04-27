import joblib
from model_handler import SentimentModel

print("Training and saving model for production...")
sm = SentimentModel()

if sm.is_trained:
    # Save the model and tfidf vectorizer
    joblib.dump(sm.model, 'model.joblib')
    joblib.dump(sm.tfidf, 'tfidf.joblib')
    print("Model and Vectorizer saved successfully as .joblib files!")
else:
    print("Failed to train model.")
