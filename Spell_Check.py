import torch
import enchant
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load the pre-trained language model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# Initialize the spell checker
spell_checker = enchant.Dict("en_US")

def contextual_spell_check(text):
    # Tokenize the input text
    tokens = tokenizer.tokenize(text)

    # Add special tokens and create input tensors
    input_ids = tokenizer.encode(["[CLS]"] + tokens + ["[SEP]"], return_tensors="pt")
    input_mask = torch.ones_like(input_ids)

    # Replace each token with the [MASK] token and predict the correct spelling
    corrected_tokens = []
    for i in range(1, len(input_ids[0]) - 1):
        # Replace the current token with the [MASK] token
        masked_input_ids = input_ids.clone()
        masked_input_ids[0, i] = tokenizer.mask_token_id

        # Predict the correct spelling using the language model
        with torch.no_grad():
            outputs = model(masked_input_ids, attention_mask=input_mask)
            predicted_token_id = torch.argmax(outputs[0][0, i]).item()
            predicted_token = tokenizer.decode(predicted_token_id)

        # Check if the predicted token is spelled correctly
        if not spell_checker.check(predicted_token):
            # If the predicted token is misspelled, use the original token
            predicted_token = tokens[i - 1]

        corrected_tokens.append(predicted_token)

    # Join the corrected tokens to form the corrected text
    corrected_text = "".join(corrected_tokens)

    return corrected_text
