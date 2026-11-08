"""
Advanced Emotion Detection using DistilBERT
Dataset: dair-ai/emotion
Task: Six-class text emotion classification
"""

import os
import json
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_NAME = "distilbert/distilbert-base-uncased"

OUTPUT_DIR = "models/emotion-distilbert"
ARTIFACT_DIR = "artifacts"

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5

LABELS = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
]

set_seed(42)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

print("Emotion Detection Training Started")


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("Loading Emotion Dataset...")

dataset = load_dataset("dair-ai/emotion")

print(dataset)

print("Training examples:", len(dataset["train"]))
print("Validation examples:", len(dataset["validation"]))
print("Test examples:", len(dataset["test"]))


# ==========================================
# 3. LOAD TOKENIZER
# ==========================================

print("Loading DistilBERT Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ==========================================
# 4. TEXT TOKENIZATION
# ==========================================

def preprocess_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH
    )


print("Tokenizing Dataset...")

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True
)


# ==========================================
# 5. MODEL CONFIGURATION
# ==========================================

id2label = {
    index: label
    for index, label in enumerate(LABELS)
}

label2id = {
    label: index
    for index, label in id2label.items()
}

print("Loading DistilBERT Classification Model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id
)


# ==========================================
# 6. DATA COLLATOR
# ==========================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ==========================================
# 7. EVALUATION METRICS
# ==========================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": f1
    }


# ==========================================
# 8. TRAINING ARGUMENTS
# ==========================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=32,

    num_train_epochs=EPOCHS,

    weight_decay=0.01,
    warmup_ratio=0.1,

    load_best_model_at_end=True,

    metric_for_best_model="macro_f1",
    greater_is_better=True,

    logging_dir="logs",
    logging_steps=50,

    save_total_limit=2,

    report_to="none",

    fp16=torch.cuda.is_available(),

    seed=42
)


# ==========================================
# 9. INITIALIZE TRAINER
# ==========================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_dataset["train"],

    eval_dataset=tokenized_dataset["validation"],

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)


# ==========================================
# 10. TRAIN MODEL
# ==========================================

print("Starting Model Training...")

train_result = trainer.train()

print("Training Completed!")


# ==========================================
# 11. SAVE TRAINED MODEL
# ==========================================

print("Saving Model...")

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)


# ==========================================
# 12. TEST DATASET EVALUATION
# ==========================================

print("Evaluating on Test Dataset...")

test_results = trainer.evaluate(
    tokenized_dataset["test"],
    metric_key_prefix="test"
)

print("\nTEST RESULTS")

for key, value in test_results.items():

    print(f"{key}: {value}")


# ==========================================
# 13. PREDICTIONS
# ==========================================

predictions_output = trainer.predict(
    tokenized_dataset["test"]
)

y_pred = np.argmax(
    predictions_output.predictions,
    axis=-1
)

y_true = predictions_output.label_ids


# ==========================================
# 14. CLASSIFICATION REPORT
# ==========================================

report = classification_report(

    y_true,

    y_pred,

    labels=list(range(len(LABELS))),

    target_names=LABELS,

    zero_division=0,

    output_dict=True
)

with open(
    os.path.join(
        ARTIFACT_DIR,
        "classification_report.json"
    ),
    "w"
) as file:

    json.dump(
        report,
        file,
        indent=4
    )


# ==========================================
# 15. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(LABELS)))
)

np.savetxt(

    os.path.join(
        ARTIFACT_DIR,
        "confusion_matrix.csv"
    ),

    cm,

    delimiter=",",

    fmt="%d"
)


# ==========================================
# 16. SAVE TEST METRICS
# ==========================================

with open(
    os.path.join(
        ARTIFACT_DIR,
        "test_metrics.json"
    ),
    "w"
) as file:

    json.dump(
        test_results,
        file,
        indent=4
    )


# ==========================================
# 17. FINAL OUTPUT
# ==========================================

print("\n" + "=" * 50)

print("EMOTION DETECTION TRAINING FINISHED")

print("=" * 50)

print("Model saved:", OUTPUT_DIR)

print("Reports saved:", ARTIFACT_DIR)

print("Emotion Classes:", LABELS)

print("Device:", trainer.args.device)

print("All tasks completed successfully!")
  
