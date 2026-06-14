from flask import Flask, request, jsonify
import os
import sys

app = Flask(__name__)

# We'll store any startup error here to display in the response instead of crashing Vercel with a generic 500
startup_error = None
startup_traceback = None

try:
    # --- Add root directory to path so 'preprocessing' and 'model_handler' can be imported ---
    ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    sys.path.insert(0, os.path.abspath(ROOT_DIR))

    from model_handler import SentimentModel  # noqa: E402

    model_handler = None

    def load_resources():
        global model_handler
        if model_handler is not None:
            return  # already loaded

        # Vercel deploys all repo files to /var/task; __file__ is inside api/
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_handler = SentimentModel(model_dir=base)

except Exception as e:
    import traceback
    startup_error = str(e)
    startup_traceback = traceback.format_exc()


@app.route('/api/predict', methods=['POST'])
@app.route('/api', methods=['POST'])
def predict():
    if startup_error:
        return jsonify({
            'error': 'Startup import failed',
            'details': startup_error,
            'traceback': startup_traceback
        }), 500

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

@app.route('/', defaults={'path': ''}, methods=['GET', 'POST'])
@app.route('/<path:path>', methods=['GET', 'POST'])
def catch_all(path):
    # If the request is GET and we're just hitting the API root
    return jsonify({
        "message": "Flask catch-all",
        "received_path": path,
        "request_url": request.url,
        "path_info": request.environ.get('PATH_INFO')
    }), 404

if __name__ == '__main__':
    app.run(debug=True)

