"""Fine-tune XLM-RoBERTa for Arabic/English sentiment classification.

Input CSV format:
    text,label
    "الخدمة ممتازة",2
    "المنتج سيء",0
    "السعر عادي",1

Labels:
    0 = negative
    1 = neutral
    2 = positive

Install:
    pip install -U torch transformers datasets accelerate scikit-learn

Run:
    python train_sentiment_mbert.py --data sentiment_train.csv
"""

from __future__ import annotations

import argparse
import os
import numpy as np
from datasets import ClassLabel, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

BASE_CHECKPOINT = "FacebookAI/xlm-roberta-base"
DEFAULT_OUTPUT = "models/xlmr-arabic-english-sentiment"
ID2LABEL = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
LABEL2ID = {name: idx for idx, name in ID2LABEL.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="sentiment_train.csv")
    parser.add_argument("--base-model", default=BASE_CHECKPOINT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Dataset not found: {args.data}. Create a CSV with columns text,label."
        )

    dataset = load_dataset("csv", data_files=args.data)["train"]
    required = {"text", "label"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    dataset = dataset.filter(
        lambda row: isinstance(row["text"], str)
        and row["text"].strip()
        and int(row["label"]) in (0, 1, 2)
    )
    dataset = dataset.cast_column("label", ClassLabel(names=["NEGATIVE", "NEUTRAL", "POSITIVE"]))
    split = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    tokenized = split.map(tokenize, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = float((predictions == labels).mean())
        macro_f1 = 0.0
        try:
            from sklearn.metrics import f1_score
            macro_f1 = float(f1_score(labels, predictions, average="macro", zero_division=0))
        except ImportError:
            pass
        return {"accuracy": accuracy, "macro_f1": macro_f1}

    training_args = TrainingArguments(
        output_dir=args.output,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved fine-tuned sentiment model to: {args.output}")


if __name__ == "__main__":
    main()
