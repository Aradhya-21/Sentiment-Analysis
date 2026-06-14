from flask import Flask, request, jsonify
import joblib
import os
import sys

# --- FIX BUG 1: Add root directory to path so 'preprocessing' can be imported ---
ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(ROOT_DIR))

from preprocessing import clean_text  # noqa: E402  (import after path fix)

app = Flask(__name__)

model = None
tfidf = None


def load_resources():
    global model, tfidf
    if model is not None and tfidf is not None:
        return  # already loaded

    # Vercel deploys all repo files to /var/task; __file__ is inside api/
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(base, 'model.joblib')
    tfidf_path  = os.path.join(base, 'tfidf.joblib')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.joblib not found at {model_path}")
    if not os.path.exists(tfidf_path):
        raise FileNotFoundError(f"tfidf.joblib not found at {tfidf_path}")

    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    print(f"Loaded model from {model_path}")


# --- FIX BUG 3: Removed dead @app.route('/') — Vercel serves index.html statically ---

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        load_resources()
        data = request.get_json(force=True)
        review = (data or {}).get('review', '').strip()

        if not review:
            return jsonify({'error': 'No review text provided'}), 400

        cleaned    = clean_text(review)
        vectorized = tfidf.transform([cleaned])

        prediction   = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]
        labels       = model.classes_

        prob_dict = {label: float(prob) for label, prob in zip(labels, probabilities)}

        return jsonify({'verdict': prediction, 'confidence': prob_dict})

    except Exception as exc:
        import traceback
        return jsonify({'error': str(exc), 'traceback': traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(debug=True)
