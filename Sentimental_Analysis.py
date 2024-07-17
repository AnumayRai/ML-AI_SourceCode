import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def sentiment_analysis(text, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)

    logits = outputs.logits
    logits = logits.squeeze().detach().cpu().numpy()
    predicted_label = logits.argmax()

    sentiment_map = {0: 'Negative', 1: 'Positive'}
    sentiment = sentiment_map[predicted_label]

    return sentiment

# Test the function
text = "I love this movie, it's so amazing!"
result = sentiment_analysis(text)
print(f"Sentiment: {result}")
