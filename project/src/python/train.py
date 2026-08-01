import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


DEFAULT_MODEL_ID = "google/flan-t5-base"
DEFAULT_DATASET_PATH = Path("./dataset.json")
DEFAULT_OUTPUT_DIR = Path("./checkpoints")
DEFAULT_BEST_MODEL_DIR = Path("./best_model")
DEFAULT_PREFIX = (
    "" #PREFIX NEEDS TO BE CHANGED IN FRONTEND
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Flan-T5 on input/output pairs.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Base model name or local path.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Path to a JSON dataset file.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for checkpoints.")
    parser.add_argument("--best-model-dir", default=str(DEFAULT_BEST_MODEL_DIR), help="Directory to save the final best model.")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Optional instruction prefix added to each input.")
    parser.add_argument("--input-column", default="input", help="Name of the source text field.")
    parser.add_argument("--target-column", default="output", help="Name of the target text field.")
    parser.add_argument("--max-input-length", type=int, default=128, help="Maximum token length for encoder inputs.")
    parser.add_argument("--max-target-length", type=int, default=384, help="Maximum token length for decoder targets.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate for fine-tuning.")
    parser.add_argument("--epochs", type=float, default=20.0, help="Number of training epochs.")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Per-device training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Per-device evaluation batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio for the scheduler.")
    parser.add_argument("--logging-steps", type=int, default=25, help="Logging frequency in steps.")
    parser.add_argument("--save-total-limit", type=int, default=2, help="Maximum number of checkpoints to keep.")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience in evaluation rounds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Optional cap on training samples after split.")
    parser.add_argument("--max-eval-samples", type=int, default=0, help="Optional cap on validation samples after split.")
    return parser.parse_args()


def load_json_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with dataset_path.open(encoding="utf8") as file_handle:
        data = json.load(file_handle)

    if isinstance(data, dict):
        if "dataset" in data:
            data = data["dataset"]
        elif "data" in data:
            data = data["data"]
        else:
            raise ValueError("JSON dataset must be a list or contain a 'dataset'/'data' key.")

    if not isinstance(data, list):
        raise ValueError("Dataset JSON must resolve to a list of samples.")

    if not data:
        raise ValueError("Dataset is empty.")

    return data


def ensure_columns(example: Dict[str, Any], input_column: str, target_column: str) -> Tuple[str, str]:
    if input_column not in example:
        raise KeyError(f"Missing input column '{input_column}'. Available keys: {sorted(example.keys())}")
    if target_column not in example:
        raise KeyError(f"Missing target column '{target_column}'. Available keys: {sorted(example.keys())}")

    source_text = str(example[input_column]).strip()
    target_text = str(example[target_column]).strip()
    return source_text, target_text


def build_preprocess_fn(
    tokenizer: AutoTokenizer,
    prefix: str,
    input_column: str,
    target_column: str,
    max_input_length: int,
    max_target_length: int,
):
    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        source_text, target_text = ensure_columns(example, input_column, target_column)

        model_inputs = tokenizer(
            prefix + source_text,
            truncation=True,
            padding=False,
            max_length=max_input_length,
        )

        labels = tokenizer(
            text_target=target_text,
            truncation=True,
            padding=False,
            max_length=max_target_length,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess


def make_validation_split(dataset: Dataset, test_size: float, seed: int) -> Tuple[Dataset, Dataset]:
    if len(dataset) < 2:
        raise ValueError("Need at least 2 samples to create a train/validation split.")

    split = dataset.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]


def limit_dataset(dataset: Dataset, max_samples: int) -> Dataset:
    if max_samples and max_samples > 0:
        return dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def compute_exact_match_metrics(tokenizer: AutoTokenizer):
    pad_token_id = tokenizer.pad_token_id

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # predictions can be a tuple
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Replace ignored labels
        labels = labels.astype(np.int64)
        labels[labels == -100] = tokenizer.pad_token_id
        labels[labels >= tokenizer.vocab_size] = tokenizer.pad_token_id

        # If predictions are logits instead of ids
        if predictions.ndim == 3:
            predictions = predictions.argmax(axis=-1)

        predictions = predictions.astype(np.int64)

        predictions[predictions < 0] = tokenizer.pad_token_id
        predictions[predictions >= tokenizer.vocab_size] = tokenizer.pad_token_id

        decoded_predictions = tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
        )

        decoded_labels = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
        )

        decoded_predictions = [text.strip() for text in decoded_predictions]
        decoded_labels = [text.strip() for text in decoded_labels]

        exact_matches = sum(pred == label for pred, label in zip(decoded_predictions, decoded_labels))
        total = max(len(decoded_labels), 1)

        return {
            "exact_match": exact_matches / total,
            "avg_prediction_length": sum(len(text) for text in decoded_predictions) / total,
        }

    return compute_metrics


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    best_model_dir = Path(args.best_model_dir)

    set_seed(args.seed)

    print("Loading dataset...")
    data = load_json_dataset(dataset_path)
    dataset = Dataset.from_list(data)

    train_ds, eval_ds = make_validation_split(dataset, test_size=args.test_size, seed=args.seed)
    train_ds = limit_dataset(train_ds, args.max_train_samples)
    eval_ds = limit_dataset(eval_ds, args.max_eval_samples)

    print(f"Train samples: {len(train_ds)}")
    print(f"Eval samples : {len(eval_ds)}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    print("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    model.config.use_cache = False

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    print("Original sample:")
    print(train_ds[0])
    print()

    preprocess = build_preprocess_fn(
        tokenizer=tokenizer,
        prefix=args.prefix,
        input_column=args.input_column,
        target_column=args.target_column,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    )

    # Tokenize datasets
    train_ds = train_ds.map(
        preprocess,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train dataset",
    )

    eval_ds = eval_ds.map(
        preprocess,
        remove_columns=eval_ds.column_names,
        desc="Tokenizing validation dataset",
    )

    print("Tokenized sample:")
    print(train_ds[0])
    print("Input length:", len(train_ds[0]["input_ids"]))
    print("Label length:", len(train_ds[0]["labels"]))

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        predict_with_generate=True,
        generation_max_length=512,
        generation_num_beams=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,
        report_to="none",
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        compute_metrics=compute_exact_match_metrics(tokenizer),
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()

    print("Saving best model...")
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    print("Done!")


if __name__ == "__main__":
    main()