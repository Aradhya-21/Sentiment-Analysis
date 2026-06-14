from model_handler import SentimentModel

sm = SentimentModel()

extra_tests = [
    "good product",
    "it's okay",
    "not great not terrible",
    "love it",
    "hate it",
    "works fine",
    "absolute garbage",
    "best purchase ever",
    "waste of money",
    "mediocre at best",
]

print("=== FINAL VERIFICATION ===")
for phrase in extra_tests:
    pred, probs = sm.predict(phrase)
    bar = {'positive': '[+]', 'negative': '[-]', 'neutral': '[~]'}
    print(f"{bar[pred]} [{pred.upper():8}]  \"{phrase}\"")
    print(f"           Pos:{probs['positive']:.0%}  Neu:{probs['neutral']:.0%}  Neg:{probs['negative']:.0%}")
