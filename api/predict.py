from flask import Flask, request, jsonify, send_from_directory
import joblib
import os
from preprocessing import clean_text

app = Flask(__name__, static_folder='..', static_url_path='')

# Load model and tfidf once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model.joblib')
TFIDF_PATH = os.path.join(os.path.dirname(__file__), '..', 'tfidf.joblib')

model = None
tfidf = None

def load_resources():
    global model, tfidf
    try:
        if model is None or tfidf is None:
            # Check multiple possible locations for the model files
            current_dir = os.path.dirname(__file__)
            possible_paths = [
                os.path.join(current_dir, '..', 'model.joblib'), # Root from /api/
                os.path.join(current_dir, 'model.joblib'),       # Same dir
                '/var/task/model.joblib'                         # Vercel absolute path
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model = joblib.load(path)
                    print(f"Loaded model from {path}")
                    break
            
            for path in [p.replace('model.joblib', 'tfidf.joblib') for p in possible_paths]:
                if os.path.exists(path):
                    tfidf = joblib.load(path)
                    print(f"Loaded tfidf from {path}")
                    break
                    
            if model is None or tfidf is None:
                raise FileNotFoundError("Could not find model.joblib or tfidf.joblib in any known location.")
    except Exception as e:
        print(f"Error loading resources: {str(e)}")
        raise e

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
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
    except Exception as e:
        import traceback
        error_msg = f"Prediction Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

# For local development
if __name__ == '__main__':
    app.run(debug=True)
