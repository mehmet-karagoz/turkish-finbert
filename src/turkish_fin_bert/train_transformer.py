import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .console import configure_console
from .train_model import LABEL_ORDER, temporal_split


LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}


def train_transformer(input_csv: Path, model_name: str, output_dir: Path, epochs: int, batch_size: int, max_length: int) -> dict:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("BERT fine-tuning için opsiyonel bağımlılıkları kurun: uv sync --extra nlp") from exc

    class SentimentDataset(torch.utils.data.Dataset):
        def __init__(self, texts: list[str], labels: list[str], tokenizer):
            self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
            self.labels = [LABEL_TO_ID[label] for label in labels]

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    df = pd.read_csv(input_csv)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df.dropna(subset=["date", "text", "label"])
    df = df[df["label"].isin(LABEL_ORDER)].copy()

    train_df, test_df = temporal_split(df, test_size=0.2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_ORDER),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    train_ds = SentimentDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer)
    eval_ds = SentimentDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        }

    training_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "logging_steps": 20,
    }
    strategy_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"
    training_kwargs[strategy_key] = "epoch"
    args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BERTurk benzeri sentiment modeli fine-tune eder.")
    parser.add_argument("--input", type=Path, required=True, help="Temizlenmiş ve etiketli CSV.")
    parser.add_argument("--model-name", default="dbmdz/bert-base-turkish-cased", help="Hugging Face model adı.")
    parser.add_argument("--output-dir", type=Path, default=Path("models/berturk_sentiment"), help="Model çıktı klasörü.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    return parser


def main() -> None:
    configure_console()
    args = build_parser().parse_args()
    metrics = train_transformer(args.input, args.model_name, args.output_dir, args.epochs, args.batch_size, args.max_length)
    print(f"BERT modeli kaydedildi: {args.output_dir}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
