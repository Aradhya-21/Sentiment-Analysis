import gradio as gr
from model_handler import SentimentModel

# Initialize the model handler
sentiment_model = SentimentModel()

def analyze_sentiment(review):
    if not review:
        return "Please enter some text."
    
    prediction, prob_dict = sentiment_model.predict(review)
    
    # Sort by probability for better display
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    results = "\n".join([f"{label.capitalize()}: {prob:.2%}" for label, prob in sorted_probs])
    
    return f"Verdict: {prediction.capitalize()}\n\nConfidence Scores:\n{results}"

# Create Gradio Interface
demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=5, label="Enter Review", placeholder="Type your product review here..."),
    outputs=gr.Textbox(label="Analysis Results"),
    title="🌟 Sentiment Analysis App",
    description="This app classifies product reviews into **Positive**, **Negative**, or **Neutral** sentiments using a Logistic Regression model.",
    examples=[
        ["I absolutely love this product! It works perfectly."],
        ["The quality is terrible and it broke after one day."],
        ["It's a decent product. Nothing special, but it works fine."],
        ["The price is okay, but the delivery was slow."],
        ["Not what I expected, but it does the job."]
    ],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
