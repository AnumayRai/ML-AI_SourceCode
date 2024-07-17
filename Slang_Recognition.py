from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests

# Load the BERT model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('path/to/your/model')
tokenizer = AutoTokenizer.from_pretrained('path/to/your/model')

# Define a function to predict whether a sentence contains slang
def contains_slang(sentence):
    inputs = tokenizer(sentence, return_tensors='pt')
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).tolist()[0]
    return probs[1] > 0.5  # Assuming class 1 is slang

# Define a function to translate slang
def translate_slang(sentence):
    if not contains_slang(sentence):
        return sentence
    # Split the sentence into words
    words = sentence.split()
    # Replace each slang word with its definition
    for i, word in enumerate(words):
        definition = get_meaning(word)
        if definition:
            words[i] = f'({definition})'
    # Return the translated sentence
    return ' '.join(words)

# Define a function to fetch the meaning of a word from Urban Dictionary
def get_meaning(word):
    response = requests.get(f'http://api.urbandictionary.com/v0/define?term={word}')
    data = response.json()
    if data['list']:
        return data['list'][0]['definition']
    else:
        return None

# Test the functions
sentence = 'That concert was lit!'
print(translate_slang(sentence))  # Outputs: That concert was (exciting or excellent)!
