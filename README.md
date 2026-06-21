## 📊 Sentiment Analysis of Product Reviews

### 🧠 Technology
- **Domain:** Artificial Intelligence / Natural Language Processing
- **Level:** Intermediate
- **Tech Stack:** Python, Flask, Scikit-learn, NLTK, Pandas, HTML/Vanilla JS
- **Deployment:** Vercel

---

### 📌 Project Description

This project performs **Sentiment Analysis** on customer product reviews. It determines whether a given review is **positive**, **neutral**, or **negative**.

It uses a **Hybrid Approach** combining:
1. **Rule-Based Sentiment (VADER):** Extended with a custom lexicon for domain-specific terminology.
2. **Machine Learning:** A Logistic Regression model trained on TF-IDF features with SMOTE to handle class imbalances.

The system dynamically balances predictions between the ML model and the VADER lexicon based on confidence scores and vocabulary sparsity. The application is packaged as a **Flask REST API** backend with a lightweight **Vanilla HTML/JS** frontend, specifically optimized for **Vercel Deployment**.

---

### 🔧 Features

- Custom Text Preprocessing (Tokenization, Stopword Removal, Stemming)
- TF-IDF Vectorization
- SMOTE for dataset balancing
- Logistic Regression Model Training
- Extended VADER Lexicon Integration
- Beautiful, responsive Web UI with dynamic animations
- RESTful Flask API (`/api/predict`)

---

### 🚀 Deployment

This project is strictly configured for deployment on **Vercel**. 

Vercel handles the routing between the static frontend and the Python Serverless Functions backend via the explicit `vercel.json` configuration.

#### Deploy to Vercel (Recommended)
1. Push this repository to GitHub.
2. Go to your [Vercel Dashboard](https://vercel.com/dashboard) and click **Add New Project**.
3. Import your GitHub repository.
4. Leave all build settings as default. The explicit `vercel.json` file in the root directory will automatically configure the builds:
   - `api/index.py` will be built as a Python Serverless Function.
   - `index.html` will be built as a Static asset.
5. Click **Deploy**.

#### How Vercel Routing Works in this Project
- The frontend fetches data from the `/api/predict` endpoint.
- `vercel.json` routes the root URL (`/`) strictly to `index.html`.
- `vercel.json` routes all `/api/*` traffic to the Flask application (`api/index.py`).
- The NLTK data (vader lexicon) is read directly from the unzipped file within the `nltk_data` directory, preventing Serverless Function startup crashes.

---

### 🖥️ Local Run

To test the API locally:
```bash
pip install -r requirements.txt
python api/index.py
```
This will start the Flask server locally. You can then open `index.html` in your browser (using Live Server or simply opening the file) and change the fetch URL in `index.html` to `http://localhost:5000/api/predict` (ensure you revert it to `/api/predict` before pushing to Vercel).

---

### 🤝 Contributing
Contributions, issues and feature requests are welcome!

### 📄 License
This project is for educational purposes. Feel free to modify and use it as needed.
