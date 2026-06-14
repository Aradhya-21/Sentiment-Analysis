from flask import Flask, request, jsonify
import joblib
import os
import sys

# --- FIX BUG 1: Add root directory to path so 'preprocessing' can be imported ---
ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(ROOT_DIR))

from model_handler import SentimentModel  # noqa: E402

app = Flask(__name__)

model_handler = None


def load_resources():
    global model_handler
    if model_handler is not None:
        return  # already loaded

    # Vercel deploys all repo files to /var/task; __file__ is inside api/
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_handler = SentimentModel(model_dir=base)


# --- FIX BUG 3: Removed dead @app.route('/') — Vercel serves index.html statically ---

@app.route('/api/predict', methods=['POST'])
@app.route('/api', methods=['POST'])
def predict():
    try:
        load_resources()
        data = request.get_json(force=True)
        review = (data or {}).get('review', '').strip()

        if not review:
            return jsonify({'error': 'No review text provided'}), 400

        prediction, prob_dict = model_handler.predict(review)

        return jsonify({'verdict': prediction, 'confidence': prob_dict})

    except Exception as exc:
        import traceback
        return jsonify({'error': str(exc), 'traceback': traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(debug=True)
