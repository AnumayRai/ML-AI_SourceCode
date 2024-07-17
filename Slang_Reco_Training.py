from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load the BERT model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Load the dataset
dataset = load_dataset('path/to/your/dataset')

# Tokenize the dataset
tokenized_dataset = dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding=True), batched=True)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='path/to/your/output',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='path/to/your/logs',
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
)

# Train the model
trainer.train()
