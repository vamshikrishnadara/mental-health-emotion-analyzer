from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch

# Load dataset
dataset = load_dataset("go_emotions")

# Select small subset for faster training
dataset = dataset["train"].shuffle(seed=42).select(range(8000))

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

dataset = dataset.map(tokenize)

def convert_label(example):
    example["label"] = example["labels"][0] if len(example["labels"]) > 0 else 0
    return example

dataset = dataset.map(convert_label)
dataset = dataset.remove_columns(["labels"])
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=28
)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

model.save_pretrained("models/emotion_transformer")
tokenizer.save_pretrained("models/emotion_transformer")

print("Transformer model trained and saved!")
