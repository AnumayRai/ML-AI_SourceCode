from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset

# Load pre-trained model and tokenizer
model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load a dataset that reflects the user's preferences
# In a real system, you would create this dataset based on the user's preferences
dataset = load_dataset('my_dataset')  # replace 'my_dataset' with your dataset

# Preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples['source'], truncation=True, padding='longest')

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Define the trainer
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {'accuracy': (predictions == labels).sum().item() / len(predictions)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./my_fine_tuned_model')

# Function to personalize translation
def personalize_translation(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# Example usage
text = "Hello, world!"
model = AutoModelForSeq2SeqLM.from_pretrained('./my_fine_tuned_model')
translation = personalize_translation(model, tokenizer, text)
print(translation)
