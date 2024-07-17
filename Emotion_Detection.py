from pyAudioAnalysis import audioFeatureExtraction as aF

def extract_audio_features(audio_file):
    F, f_names = aF.stFeatureExtraction(audio_file, 1.0, 1.0, 0.05 * 44100, 0.025 * 44100)
    return F, f_names
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def fine_tune_text_emotion_detection_model(model_name, dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(dataset["labels"]))

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    trainer.train()
