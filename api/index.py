from flask import Flask, request, jsonify, send_from_directory
import joblib
import os
from preprocessing import clean_text

app = Flask(__name__, static_folder='../public', static_url_path='')

# Load model and tfidf once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model.joblib')
TFIDF_PATH = os.path.join(os.path.dirname(__file__), '..', 'tfidf.joblib')

model = None
tfidf = None

def load_resources():
    global model, tfidf
    if model is None or tfidf is None:
        model = joblib.load(MODEL_PATH)
        tfidf = joblib.load(TFIDF_PATH)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    load_resources()
    data = request.json
    review = data.get('review', '')
    
    if not review:
        return jsonify({'error': 'No review provided'}), 400
        
    cleaned = clean_text(review)
    vectorized = tfidf.transform([cleaned])
    
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    labels = model.classes_
    
    prob_dict = {label: float(prob) for label, prob in zip(labels, probabilities)}
    
    return jsonify({
        'verdict': prediction,
        'confidence': prob_dict
    })

# For local development
if __name__ == '__main__':
    app.run(debug=True)
