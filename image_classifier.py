import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput


# ==========================================
# 기본 설정
# ==========================================

IMAGE_MODEL_NAME = "openai/clip-vit-base-patch32"

DATA_FILES = {
    "train": "./data/train.csv",
    "validation": "./data/valid.csv",
    "test": "./data/test.csv",
}

OUTPUT_DIR = "./outputs/image_checkpoints"
BEST_MODEL_DIR = "./outputs/best_image_classifier"

NUM_LABELS = 5
SEED = 42

ENCODER_LEARNING_RATE = 1e-5
CLASSIFIER_LEARNING_RATE = 1e-3

LABEL2ID = {
    "schedule": 0,
    "shopping": 1,
    "place": 2,
    "memo": 3,
    "unknown": 4,
}

ID2LABEL = {
    value: key
    for key, value in LABEL2ID.items()
}

PROJECT_ROOT = Path(__file__).resolve().parent


# ==========================================
# Seed
# ==========================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)


# ==========================================
# CSV 로드
# ==========================================

def load_csv_dataset():
    for split_name, path in DATA_FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{split_name} 파일이 없습니다: {path}"
            )

    dataset = load_dataset(
        "csv",
        data_files=DATA_FILES,
    )

    for split_name, split_data in dataset.items():
        if "image_path" not in split_data.column_names:
            raise ValueError(
                f"{split_name} CSV에 image_path 컬럼이 없습니다."
            )

        if (
            "label" not in split_data.column_names
            and "label_id" not in split_data.column_names
        ):
            raise ValueError(
                f"{split_name} CSV에 label 또는 "
                "label_id 컬럼이 필요합니다."
            )

    # LABEL2ID에 정의된 5개 라벨만 유지
    # schedule / shopping / place / memo / unknown
    for split_name in dataset:
        dataset[split_name] = dataset[split_name].filter(
            lambda example: str(
                example.get("label", "")
            ).strip() in LABEL2ID
        )

    return dataset


# ==========================================
# 라벨 처리
# ==========================================

def get_label_id(example: Dict[str, Any]) -> int:
    label_id = example.get("label_id")

    if label_id not in [None, ""]:
        label_id = int(label_id)

        if label_id not in ID2LABEL:
            raise ValueError(
                f"알 수 없는 label_id: {label_id}"
            )

        return label_id

    label = str(
        example.get("label", "")
    ).strip()

    if label not in LABEL2ID:
        raise ValueError(
            f"알 수 없는 라벨: {label}"
        )

    return LABEL2ID[label]


# ==========================================
# 이미지 Dataset
# ==========================================

class CaptureMateImageDataset(Dataset):

    def __init__(
        self,
        dataset,
        image_processor,
    ):
        self.dataset = dataset
        self.image_processor = image_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]

        image_path = Path(
            str(example["image_path"]).strip()
        )

        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        if not image_path.exists():
            raise FileNotFoundError(
                f"이미지 파일이 없습니다: {image_path}"
            )

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")

                image_inputs = self.image_processor(
                    images=image,
                    return_tensors="pt",
                )

        except Exception as error:
            raise RuntimeError(
                f"이미지 로드 실패: {image_path}"
            ) from error

        return {
            "pixel_values": image_inputs[
                "pixel_values"
            ].squeeze(0),
            "labels": get_label_id(example),
        }


# ==========================================
# Data Collator
# ==========================================

class ImageDataCollator:

    def __call__(self, features):
        pixel_values = torch.stack(
            [
                feature["pixel_values"]
                for feature in features
            ]
        )

        labels = torch.tensor(
            [
                feature["labels"]
                for feature in features
            ],
            dtype=torch.long,
        )

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


# ==========================================
# CLIP UI 분류 모델
# ==========================================

class CLIPUIClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.image_model = (
            CLIPVisionModelWithProjection.from_pretrained(
                IMAGE_MODEL_NAME
            )
        )

        # CLIP 전체 고정
        for parameter in self.image_model.parameters():
            parameter.requires_grad = False

        # 마지막 Vision Transformer Block만 학습
        last_vision_layer = (
            self.image_model
            .vision_model
            .encoder
            .layers[-1]
        )

        for parameter in last_vision_layer.parameters():
            parameter.requires_grad = True

        # CLIP image projection도 학습
        for parameter in (
            self.image_model.visual_projection.parameters()
        ):
            parameter.requires_grad = True

        image_feature_size = (
            self.image_model.config.projection_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                image_feature_size,
                256,
            ),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(
                256,
                128,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(
                128,
                NUM_LABELS,
            ),
        )

    def forward(
        self,
        pixel_values=None,
        labels=None,
    ):
        image_output = self.image_model(
            pixel_values=pixel_values,
        )

        image_feature = image_output.image_embeds

        image_feature = F.normalize(
            image_feature,
            dim=1,
        )

        logits = self.classifier(
            image_feature
        )

        loss = None

        if labels is not None:
            loss = F.cross_entropy(
                logits,
                labels,
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


# ==========================================
# 학습률 분리 Trainer
# ==========================================

class ImageTrainer(Trainer):

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        encoder_parameters = [
            parameter
            for parameter
            in self.model.image_model.parameters()
            if parameter.requires_grad
        ]

        classifier_parameters = [
            parameter
            for parameter
            in self.model.classifier.parameters()
            if parameter.requires_grad
        ]

        optimizer_groups = [
            {
                "params": encoder_parameters,
                "lr": ENCODER_LEARNING_RATE,
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": classifier_parameters,
                "lr": CLASSIFIER_LEARNING_RATE,
                "weight_decay": self.args.weight_decay,
            },
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_groups
        )

        return self.optimizer


# ==========================================
# 평가 지표
# ==========================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
    )

    precision, recall, _, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": precision,
        "macro_recall": recall,
    }


# ==========================================
# 실행
# ==========================================

def main():
    seed_everything(SEED)

    print("1) 데이터 로드")
    dataset = load_csv_dataset()

    print(
        "train:",
        len(dataset["train"]),
        "validation:",
        len(dataset["validation"]),
        "test:",
        len(dataset["test"]),
    )

    print("2) CLIP 이미지 Processor 로드")
    image_processor = (
        CLIPImageProcessor.from_pretrained(
            IMAGE_MODEL_NAME
        )
    )

    print("3) 이미지 Dataset 생성")
    train_dataset = CaptureMateImageDataset(
        dataset=dataset["train"],
        image_processor=image_processor,
    )

    valid_dataset = CaptureMateImageDataset(
        dataset=dataset["validation"],
        image_processor=image_processor,
    )

    test_dataset = CaptureMateImageDataset(
        dataset=dataset["test"],
        image_processor=image_processor,
    )

    print("4) CLIP UI 분류 모델 로드")
    model = CLIPUIClassifier()

    data_collator = ImageDataCollator()

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"학습 파라미터: "
        f"{trainable_parameters:,} / {total_parameters:,}"
    )

    print("5) 학습 설정")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",

        # 실제 학습률은 ImageTrainer에서 분리 적용
        learning_rate=ENCODER_LEARNING_RATE,

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,

        num_train_epochs=10,
        weight_decay=0.01,

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        save_total_limit=2,
        report_to="none",
        seed=SEED,

        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = ImageTrainer(
        model=model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=valid_dataset,

        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("6) 이미지 분류 학습 시작")
    trainer.train()

    print("7) validation 평가")
    validation_result = trainer.evaluate(
        valid_dataset
    )
    print(validation_result)

    print("8) test 평가")
    test_result = trainer.evaluate(
        test_dataset
    )
    print(test_result)

    print("9) 이미지 모델 저장")
    os.makedirs(
        BEST_MODEL_DIR,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": trainer.model.state_dict(),
            "label2id": LABEL2ID,
            "id2label": ID2LABEL,
            "image_model_name": IMAGE_MODEL_NAME,
            "num_labels": NUM_LABELS,
        },
        os.path.join(
            BEST_MODEL_DIR,
            "image_classifier.pt",
        ),
    )

    image_processor.save_pretrained(
        BEST_MODEL_DIR
    )

    print(f"완료: {BEST_MODEL_DIR}")


if __name__ == "__main__":
    main()