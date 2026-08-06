import os
import random
from typing import Dict, Any

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_NAME = "./models/klue-roberta-base"

DATA_FILES = {
    "train": "./data/train.csv",
    "validation": "./data/valid.csv",
    "test": "./data/test.csv",
}

OUTPUT_DIR = "./outputs/checkpoints"
BEST_MODEL_DIR = "./outputs/best_classifier"

MAX_LENGTH = 256
NUM_LABELS = 5
SEED = 42

LABEL2ID = {
    "schedule": 0,
    "shopping": 1,
    "place": 2,
    "memo": 3,
    # "trash": 4,
    "unknown": 4,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)


def load_csv_dataset():
    for split_name, path in DATA_FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{split_name} 파일이 없습니다: {path}")
    return load_dataset("csv", data_files=DATA_FILES)


def preprocess_examples(example: Dict[str, Any]) -> Dict[str, Any]:
    text = str(example.get("text", "")).strip()
    if not text:
        text = "[EMPTY]"

    if "label_id" in example and example["label_id"] not in [None, ""]:
        label_id = int(example["label_id"])
    elif "label" in example and example["label"] not in [None, ""]:
        label_str = str(example["label"]).strip()
        if label_str not in LABEL2ID:
            raise ValueError(f"알 수 없는 라벨: {label_str}")
        label_id = LABEL2ID[label_str]
    else:
        raise ValueError("CSV에 label_id 또는 label 컬럼이 필요합니다.")

    return {
        "text": text,
        "labels": label_id,
    }


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": precision,
        "macro_recall": recall,
    }


def main():
    seed_everything(SEED)

    print("1) 데이터 로드")
    dataset = load_csv_dataset()

    print("2) 토크나이저 로드")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("3) 전처리")
    dataset = dataset.map(preprocess_examples)

    print("4) 토크나이징")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
    )

    keep_columns = {"input_ids", "attention_mask", "labels"}
    remove_columns = [
        col for col in tokenized_dataset["train"].column_names
        if col not in keep_columns
    ]
    tokenized_dataset = tokenized_dataset.remove_columns(remove_columns)

    print("5) 분류 모델 로드")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("6) 학습 설정")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("7) 학습 시작")
    trainer.train()

    print("8) validation 평가")
    print(trainer.evaluate(tokenized_dataset["validation"]))

    print("9) test 평가")
    print(trainer.evaluate(tokenized_dataset["test"]))

    print("10) 모델 저장")
    trainer.save_model(BEST_MODEL_DIR)
    tokenizer.save_pretrained(BEST_MODEL_DIR)

    print(f"완료: {BEST_MODEL_DIR}")


if __name__ == "__main__":
    main()