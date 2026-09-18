"""Phase 3: fine-tune WangchanBERTa on Wisesight Sentiment, raw vs. normalized text.

Usage:
    python -m scripts.train_sentiment --variant raw
    python -m scripts.train_sentiment --variant normalized
"""

import argparse
import json
import os

import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.normalize import normalize_thai

MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
LABEL_NAMES = ["pos", "neu", "neg", "q"]


def build_texts(raw_texts: list[str], variant: str) -> list[str]:
    if variant == "raw":
        return raw_texts
    return [normalize_thai(t)["normalized_text"] for t in raw_texts]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["raw", "normalized"], required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--limit", type=int, default=None, help="truncate each split, for smoke-testing")
    parser.add_argument("--save-model-dir", default=None, help="if set, save the trained model+tokenizer here")
    parser.add_argument("--warmup-ratio", type=float, default=0.0, help="fraction of steps for linear LR warmup")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_sentiment] variant={args.variant} device={device}", flush=True)

    ds = load_dataset("pythainlp/wisesight_sentiment")
    if args.limit:
        ds = {split: ds[split].select(range(min(args.limit, len(ds[split])))) for split in ds}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_split(split):
        texts = build_texts(list(ds[split]["texts"]), args.variant)
        enc = tokenizer(texts, truncation=True, max_length=args.max_length, padding="max_length")
        return Dataset.from_dict(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": list(ds[split]["category"]),
                "text": texts,
            }
        )

    train_ds = tokenize_split("train")
    val_ds = tokenize_split("validation")
    test_ds = tokenize_split("test")
    print(f"[train_sentiment] tokenized train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
    print("[train_sentiment] model loaded, starting Trainer.train()", flush=True)

    steps_per_epoch = -(-len(train_ds) // args.batch_size)  # ceil div
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"[train_sentiment] total_steps={total_steps} warmup_steps={warmup_steps}", flush=True)

    training_args = TrainingArguments(
        output_dir=f"/tmp/wangchanberta-{args.variant}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        disable_tqdm=True,
        report_to=[],
        seed=42,
        warmup_steps=warmup_steps,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds.remove_columns(["text"]),
        eval_dataset=val_ds.remove_columns(["text"]),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    if args.save_model_dir:
        trainer.save_model(args.save_model_dir)
        tokenizer.save_pretrained(args.save_model_dir)
        print(f"[train_sentiment] saved model to {args.save_model_dir}", flush=True)

    test_output = trainer.predict(test_ds.remove_columns(["text"]))
    preds = np.argmax(test_output.predictions, axis=-1)
    labels = test_output.label_ids

    metrics = {
        "variant": args.variant,
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "per_class_f1": dict(
            zip(LABEL_NAMES, f1_score(labels, preds, average=None, labels=range(4)))
        ),
        "confusion_matrix": confusion_matrix(labels, preds, labels=range(4)).tolist(),
        "label_names": LABEL_NAMES,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/{args.variant}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(f"{args.output_dir}/{args.variant}_predictions.jsonl", "w", encoding="utf-8") as f:
        for text, true_label, pred_label in zip(test_ds["text"], labels, preds):
            f.write(
                json.dumps(
                    {
                        "text": text,
                        "true_label": LABEL_NAMES[int(true_label)],
                        "pred_label": LABEL_NAMES[int(pred_label)],
                        "correct": bool(true_label == pred_label),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
