import argparse
import contextlib
import csv
import json
import os
from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)
from huggingface_hub import snapshot_download


# ==========================================
# 설정
# ==========================================

TEXT_MODEL_ID = os.getenv(
    "CAPTUREMATE_TEXT_MODEL_ID",
    "hur03/capturemate-category-classifier-v1-5class",
)

IMAGE_MODEL_DIR = os.getenv(
    "CAPTUREMATE_IMAGE_MODEL_DIR",
    "",
)

IMAGE_MODEL_ID = os.getenv(
    "CAPTUREMATE_IMAGE_MODEL_ID",
    "hur03/capturemate-image-classifier-v1-5class",
)

IMAGE_MODEL_CACHE_DIR = os.getenv(
    "CAPTUREMATE_IMAGE_MODEL_CACHE_DIR",
    "",
)

TEST_CSV_PATH = "./data/test.csv"

TEXT_WEIGHT = 0.75
IMAGE_WEIGHT = 0.25
FUSION_WEIGHT_GRID = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.9,
    1.0,
]

VALID_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
}

MAX_LENGTH = 256
NUM_LABELS = 5

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

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==========================================
# CLIP 이미지 분류 모델
# Hugging Face에 저장된 커스텀 체크포인트 구조와 같아야 함
# ==========================================

class CLIPUIClassifier(nn.Module):

    def __init__(
        self,
        image_model_name: str,
        num_labels: int,
    ):
        super().__init__()

        self.image_model = (
            CLIPVisionModelWithProjection.from_pretrained(
                image_model_name
            )
        )

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
                num_labels,
            ),
        )

    def forward(
        self,
        pixel_values=None,
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

        return logits


# ==========================================
# 텍스트 모델 로드
# ==========================================

def load_text_model():
    print(f"텍스트 모델: {TEXT_MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(
        TEXT_MODEL_ID
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(TEXT_MODEL_ID)
        .to(DEVICE)
    )

    model.eval()

    return tokenizer, model


# ==========================================
# 이미지 모델 로드
# ==========================================

def load_image_model():
    if IMAGE_MODEL_DIR:
        image_model_dir = Path(
            IMAGE_MODEL_DIR
        )
        print(f"이미지 모델 로컬 경로: {image_model_dir}")
    else:
        print(f"이미지 모델: {IMAGE_MODEL_ID}")
        image_model_dir = Path(
            snapshot_download(
                repo_id=IMAGE_MODEL_ID,
                repo_type="model",
                cache_dir=(
                    IMAGE_MODEL_CACHE_DIR
                    or None
                ),
                allow_patterns=[
                    "image_classifier.pt",
                    "preprocessor_config.json",
                ],
            )
        )

    checkpoint_path = (
        image_model_dir
        / "image_classifier.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"이미지 모델 파일이 없습니다: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    image_model_name = checkpoint.get(
        "image_model_name",
        "openai/clip-vit-base-patch32",
    )

    num_labels = checkpoint.get(
        "num_labels",
        NUM_LABELS,
    )

    image_processor = (
        CLIPImageProcessor.from_pretrained(
            image_model_dir
        )
    )

    model = CLIPUIClassifier(
        image_model_name=image_model_name,
        num_labels=num_labels,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    return image_processor, model


# ==========================================
# 텍스트 예측
# ==========================================

@torch.inference_mode()
def predict_text(
    text: str,
    tokenizer,
    model,
):
    clean_text = str(text).strip()

    if not clean_text or clean_text.lower() == "nan":
        clean_text = "[EMPTY]"

    inputs = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    logits = model(
        **inputs
    ).logits

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )[0]

    return {
        ID2LABEL[index]: probability.item()
        for index, probability
        in enumerate(probabilities)
    }


# ==========================================
# 이미지 예측
# ==========================================

@torch.inference_mode()
def predict_image(
    image_path: str,
    image_processor,
    model,
):
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"이미지 파일이 없습니다: {path}"
        )

    with Image.open(path) as image:
        image = image.convert("RGB")

        inputs = image_processor(
            images=image,
            return_tensors="pt",
        )

    pixel_values = inputs[
        "pixel_values"
    ].to(DEVICE)

    logits = model(
        pixel_values=pixel_values
    )

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )[0]

    return {
        ID2LABEL[index]: probability.item()
        for index, probability
        in enumerate(probabilities)
    }


# ==========================================
# 결과 결합
# ==========================================

def get_top_score(scores):
    sorted_scores = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_label, top_score = sorted_scores[0]
    second_label, second_score = sorted_scores[1]

    return {
        "label": top_label,
        "confidence": top_score,
        "second_label": second_label,
        "second_confidence": second_score,
        "margin": top_score - second_score,
    }


def fuse_predictions(
    text_scores,
    image_scores,
    text_weight: float = TEXT_WEIGHT,
    image_weight: float = IMAGE_WEIGHT,
):
    if abs(
        text_weight + image_weight - 1.0
    ) > 1e-6:
        raise ValueError(
            "텍스트 가중치와 이미지 가중치의 합은 "
            "1이어야 합니다."
        )

    final_scores = {}

    for label in LABEL2ID:
        final_scores[label] = (
            text_scores[label] * text_weight
            + image_scores[label] * image_weight
        )

    predicted_label = max(
        final_scores,
        key=final_scores.get,
    )

    confidence = final_scores[
        predicted_label
    ]

    return predicted_label, confidence, final_scores


def choose_adaptive_weights(
    text_scores,
    image_scores,
):
    text_top = get_top_score(
        text_scores
    )

    image_top = get_top_score(
        image_scores
    )

    text_confidence = text_top[
        "confidence"
    ]

    text_margin = text_top[
        "margin"
    ]

    image_confidence = image_top[
        "confidence"
    ]

    image_margin = image_top[
        "margin"
    ]

    # 텍스트 모델이 확실할수록 텍스트 비중을 높이고,
    # 텍스트가 애매하면 이미지 모델이 더 개입하게 한다.
    if (
        text_confidence >= 0.85
        and text_margin >= 0.50
    ):
        text_weight = 0.80
    elif (
        text_confidence >= 0.70
        and text_margin >= 0.30
    ):
        text_weight = 0.65
    elif text_confidence >= 0.55:
        text_weight = 0.50
    else:
        text_weight = 0.35

    # 이미지 모델이 매우 확신하고 텍스트 모델은 덜 확실하면
    # 이미지 비중을 조금 더 키운다.
    if (
        image_confidence >= 0.85
        and image_margin >= 0.50
        and text_confidence < 0.80
    ):
        text_weight -= 0.15

    # 두 모델이 같은 라벨을 고르면 결합 결과는 안정적이므로,
    # 과하게 한쪽으로 쏠리지 않게 중간값으로 둔다.
    if text_top["label"] == image_top["label"]:
        text_weight = max(
            0.45,
            min(text_weight, 0.70),
        )

    text_weight = max(
        0.20,
        min(text_weight, 0.85),
    )

    image_weight = 1.0 - text_weight

    return text_weight, image_weight


def fuse_predictions_adaptive(
    text_scores,
    image_scores,
):
    text_weight, image_weight = (
        choose_adaptive_weights(
            text_scores=text_scores,
            image_scores=image_scores,
        )
    )

    (
        predicted_label,
        confidence,
        final_scores,
    ) = fuse_predictions(
        text_scores=text_scores,
        image_scores=image_scores,
        text_weight=text_weight,
        image_weight=image_weight,
    )

    return (
        predicted_label,
        confidence,
        final_scores,
        text_weight,
        image_weight,
    )


def calculate_metrics(
    targets,
    predictions,
):
    return {
        "accuracy": accuracy_score(
            targets,
            predictions,
        ),
        "macro_f1": f1_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_precision": precision_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }


# ==========================================
# 확률 출력
# ==========================================

def print_scores(
    title: str,
    scores,
):
    print(f"\n[{title}]")

    sorted_scores = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for label, score in sorted_scores:
        print(
            f"{label:10s}: {score:.4f}"
        )


def shorten_text(text: str, max_length: int = 80) -> str:
    text = " ".join(str(text).split())

    if len(text) <= max_length:
        return text

    return text[: max_length - 3] + "..."


def find_image_paths(path: str):
    target = Path(path)

    if not target.exists():
        raise FileNotFoundError(
            f"이미지 경로가 없습니다: {target}"
        )

    if target.is_file():
        if target.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            raise ValueError(
                f"지원하지 않는 이미지 확장자입니다: {target}"
            )

        return [target]

    return sorted(
        child
        for child in target.rglob("*")
        if child.is_file()
        and child.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )


def predict_single_image_result(
    image_path: Path,
    tokenizer,
    text_model,
    image_processor,
    image_model,
):
    from extract_ocr_dataset import extract_text

    ocr_text = extract_text(
        str(image_path)
    )

    text_scores = predict_text(
        text=ocr_text,
        tokenizer=tokenizer,
        model=text_model,
    )

    image_scores = predict_image(
        image_path=str(image_path),
        image_processor=image_processor,
        model=image_model,
    )

    (
        final_label,
        final_confidence,
        final_scores,
    ) = fuse_predictions(
        text_scores=text_scores,
        image_scores=image_scores,
        text_weight=TEXT_WEIGHT,
        image_weight=IMAGE_WEIGHT,
    )

    text_top = get_top_score(
        text_scores
    )
    image_top = get_top_score(
        image_scores
    )

    return {
        "filename": image_path.name,
        "image_path": str(image_path),
        "ocr_text": ocr_text,
        "text_label": text_top["label"],
        "text_confidence": text_top["confidence"],
        "image_label": image_top["label"],
        "image_confidence": image_top["confidence"],
        "final_label": final_label,
        "final_confidence": final_confidence,
        "final_scores": final_scores,
        "text_weight": TEXT_WEIGHT,
        "image_weight": IMAGE_WEIGHT,
        "error": "",
    }


def run_batch_prediction(
    image_input: str,
    output_csv: str,
    tokenizer,
    text_model,
    image_processor,
    image_model,
):
    image_paths = find_image_paths(
        image_input
    )

    if not image_paths:
        raise RuntimeError(
            f"이미지 파일이 없습니다: {image_input}"
        )

    print(
        f"Fusion: Text {TEXT_WEIGHT:.2f} / "
        f"Image {IMAGE_WEIGHT:.2f}"
    )
    print(f"이미지 개수: {len(image_paths)}")

    rows = []

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        try:
            result = predict_single_image_result(
                image_path=image_path,
                tokenizer=tokenizer,
                text_model=text_model,
                image_processor=image_processor,
                image_model=image_model,
            )

            rows.append(result)

            print(
                f"[{index}/{len(image_paths)}] "
                f"{result['filename']} -> "
                f"{result['final_label']} "
                f"({result['final_confidence']:.4f}) "
                f"| text={result['text_label']} "
                f"| image={result['image_label']}"
            )
            print(
                f"  OCR: {shorten_text(result['ocr_text'])}"
            )

        except Exception as error:
            row = {
                "filename": image_path.name,
                "image_path": str(image_path),
                "ocr_text": "",
                "text_label": "",
                "text_confidence": "",
                "image_label": "",
                "image_confidence": "",
                "final_label": "ERROR",
                "final_confidence": "",
                "text_weight": TEXT_WEIGHT,
                "image_weight": IMAGE_WEIGHT,
                "error": str(error),
            }
            rows.append(row)
            print(
                f"[{index}/{len(image_paths)}] "
                f"{image_path.name} -> ERROR: {error}"
            )

    if output_csv:
        output_path = Path(
            output_csv
        )
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = [
            "filename",
            "image_path",
            "ocr_text",
            "text_label",
            "text_confidence",
            "image_label",
            "image_confidence",
            "final_label",
            "final_confidence",
            "text_weight",
            "image_weight",
            "error",
        ]

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"CSV 저장: {output_path}")


def build_api_result(result):
    return {
        "filename": result["filename"],
        "category": result["final_label"],
        "confidence": result["final_confidence"],
        "text_weight": result["text_weight"],
        "image_weight": result["image_weight"],
    }


def run_api_prediction(
    image_input: str,
    tokenizer,
    text_model,
    image_processor,
    image_model,
):
    image_paths = find_image_paths(
        image_input
    )

    if not image_paths:
        raise RuntimeError(
            f"이미지 파일이 없습니다: {image_input}"
        )

    results = []

    for image_path in image_paths:
        try:
            result = predict_single_image_result(
                image_path=image_path,
                tokenizer=tokenizer,
                text_model=text_model,
                image_processor=image_processor,
                image_model=image_model,
            )
            results.append(
                build_api_result(result)
            )

        except Exception as error:
            results.append(
                {
                    "filename": image_path.name,
                    "category": "ERROR",
                    "confidence": None,
                    "text_weight": TEXT_WEIGHT,
                    "image_weight": IMAGE_WEIGHT,
                    "error": str(error),
                }
            )

    if len(results) == 1:
        return results[0]

    return {
        "results": results
    }


# ==========================================
# 이미지 한 장 전체 예측
# ==========================================

def predict_multimodal(
    image_path: str,
    tokenizer,
    text_model,
    image_processor,
    image_model,
):
    print("\n1) OCR 텍스트 추출")

    from extract_ocr_dataset import extract_text

    ocr_text = extract_text(
        image_path
    )

    print("\n[추출된 OCR 텍스트]")
    print(
        ocr_text
        if ocr_text
        else "추출된 텍스트가 없습니다."
    )

    print("\n2) RoBERTa 텍스트 분류")

    text_scores = predict_text(
        text=ocr_text,
        tokenizer=tokenizer,
        model=text_model,
    )

    print("3) CLIP 이미지 분류")

    image_scores = predict_image(
        image_path=image_path,
        image_processor=image_processor,
        model=image_model,
    )

    print("4) 결과 결합")

    (
        final_label,
        final_confidence,
        final_scores,
    ) = fuse_predictions(
        text_scores=text_scores,
        image_scores=image_scores,
        text_weight=TEXT_WEIGHT,
        image_weight=IMAGE_WEIGHT,
    )

    return {
        "image_path": image_path,
        "ocr_text": ocr_text,
        "text_scores": text_scores,
        "image_scores": image_scores,
        "final_scores": final_scores,
        "category": final_label,
        "confidence": final_confidence,
        "text_weight": TEXT_WEIGHT,
        "image_weight": IMAGE_WEIGHT,
    }


# ==========================================
# test.csv 전체 평가
# ==========================================

def evaluate_multimodal(
    tokenizer,
    text_model,
    image_processor,
    image_model,
    test_csv: str = TEST_CSV_PATH,
):
    test_path = Path(test_csv)

    if not test_path.exists():
        raise FileNotFoundError(
            f"테스트 CSV가 없습니다: {test_path}"
        )

    test_df = pd.read_csv(
        test_path,
        encoding="utf-8-sig",
    )

    required_columns = {
        "image_path",
        "text",
        "label",
    }

    missing_columns = (
        required_columns - set(test_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"테스트 CSV에 필요한 컬럼이 없습니다: "
            f"{sorted(missing_columns)}"
        )

    targets = []
    text_predictions = []
    image_predictions = []
    default_fusion_predictions = []
    fixed_predictions_by_weight = {
        text_weight: []
        for text_weight in FUSION_WEIGHT_GRID
    }

    failed_count = 0
    total_count = len(test_df)

    print("\n" + "=" * 60)
    print(f"멀티모달 전체 평가 시작: {total_count}개")
    print(
        f"기본 고정 Fusion 비율: "
        f"Text {TEXT_WEIGHT} / Image {IMAGE_WEIGHT}"
    )
    print("기본 고정 Fusion을 최종값으로 사용하고, 가중치 스윕은 참고용으로 평가합니다.")
    print("=" * 60)

    for position, (_, row) in enumerate(
        test_df.iterrows(),
        start=1,
    ):
        image_path = str(
            row["image_path"]
        ).strip()

        ocr_text = str(
            row["text"]
        ).strip()

        target_label = str(
            row["label"]
        ).strip()

        if target_label not in LABEL2ID:
            print(
                f"[{position}/{total_count}] "
                f"지원하지 않는 라벨: {target_label}"
            )
            failed_count += 1
            continue

        try:
            text_scores = predict_text(
                text=ocr_text,
                tokenizer=tokenizer,
                model=text_model,
            )

            image_scores = predict_image(
                image_path=image_path,
                image_processor=image_processor,
                model=image_model,
            )

            (
                default_fusion_label,
                _,
                _,
            ) = fuse_predictions(
                text_scores=text_scores,
                image_scores=image_scores,
                text_weight=TEXT_WEIGHT,
                image_weight=IMAGE_WEIGHT,
            )

            text_label = max(
                text_scores,
                key=text_scores.get,
            )

            image_label = max(
                image_scores,
                key=image_scores.get,
            )

            targets.append(target_label)

            text_predictions.append(
                text_label
            )

            image_predictions.append(
                image_label
            )

            default_fusion_predictions.append(
                default_fusion_label
            )

            for text_weight in FUSION_WEIGHT_GRID:
                image_weight = 1.0 - text_weight

                (
                    fixed_label,
                    _,
                    _,
                ) = fuse_predictions(
                    text_scores=text_scores,
                    image_scores=image_scores,
                    text_weight=text_weight,
                    image_weight=image_weight,
                )

                fixed_predictions_by_weight[
                    text_weight
                ].append(fixed_label)

            mark = (
                "O"
                if target_label == default_fusion_label
                else "X"
            )

            print(
                f"[{position}/{total_count}] "
                f"{mark} "
                f"정답={target_label}, "
                f"Text={text_label}, "
                f"Image={image_label}, "
                f"Fusion={default_fusion_label}, "
                f"W=({TEXT_WEIGHT:.2f}/{IMAGE_WEIGHT:.2f})"
            )

        except Exception as error:
            failed_count += 1

            print(
                f"[{position}/{total_count}] "
                f"처리 실패: {image_path}"
            )
            print(f"오류: {error}")

    if not default_fusion_predictions:
        print("\n평가 가능한 데이터가 없습니다.")
        return

    label_names = list(
        LABEL2ID.keys()
    )

    default_fusion_metrics = calculate_metrics(
        targets,
        default_fusion_predictions,
    )

    text_metrics = calculate_metrics(
        targets,
        text_predictions,
    )

    image_metrics = calculate_metrics(
        targets,
        image_predictions,
    )

    fixed_results = []

    for text_weight, weight_predictions in (
        fixed_predictions_by_weight.items()
    ):
        metrics = calculate_metrics(
            targets,
            weight_predictions,
        )

        fixed_results.append(
            {
                "text_weight": text_weight,
                "image_weight": 1.0 - text_weight,
                "predictions": weight_predictions,
                **metrics,
            }
        )

    fixed_results = sorted(
        fixed_results,
        key=lambda item: (
            item["macro_f1"],
            item["accuracy"],
        ),
        reverse=True,
    )

    print("\n" + "=" * 60)
    print("모델별 성능 비교")
    print("=" * 60)

    print("\n[RoBERTa Text]")
    print(
        f"Accuracy : {text_metrics['accuracy']:.4f}"
    )
    print(
        f"Macro F1 : {text_metrics['macro_f1']:.4f}"
    )

    print("\n[CLIP Image]")
    print(
        f"Accuracy : {image_metrics['accuracy']:.4f}"
    )
    print(
        f"Macro F1 : {image_metrics['macro_f1']:.4f}"
    )

    print("\n[Fixed Weight Sweep]")
    print("Text/Image | Accuracy | Macro F1")

    for result in fixed_results:
        print(
            f"{result['text_weight']:.1f}/"
            f"{result['image_weight']:.1f}"
            f"     | {result['accuracy']:.4f}"
            f"   | {result['macro_f1']:.4f}"
        )

    best_fixed = fixed_results[0]

    print("\n[Default Fixed Fusion]")
    print(
        f"Weight          : Text {TEXT_WEIGHT:.1f} / "
        f"Image {IMAGE_WEIGHT:.1f}"
    )
    print(
        f"Accuracy        : "
        f"{default_fusion_metrics['accuracy']:.4f}"
    )
    print(
        f"Macro F1        : "
        f"{default_fusion_metrics['macro_f1']:.4f}"
    )
    print(
        f"Macro Precision : "
        f"{default_fusion_metrics['macro_precision']:.4f}"
    )
    print(
        f"Macro Recall    : "
        f"{default_fusion_metrics['macro_recall']:.4f}"
    )

    print("\n[Best Fixed Fusion in Sweep]")
    print(
        f"Weight          : Text "
        f"{best_fixed['text_weight']:.1f} / Image "
        f"{best_fixed['image_weight']:.1f}"
    )
    print(
        f"Accuracy        : {best_fixed['accuracy']:.4f}"
    )
    print(
        f"Macro F1        : {best_fixed['macro_f1']:.4f}"
    )
    print(
        f"Macro Precision : "
        f"{best_fixed['macro_precision']:.4f}"
    )
    print(
        f"Macro Recall    : "
        f"{best_fixed['macro_recall']:.4f}"
    )

    print("\n[Default Fixed Fusion Classification Report]")

    print(
        classification_report(
            targets,
            default_fusion_predictions,
            labels=label_names,
            target_names=label_names,
            digits=4,
            zero_division=0,
        )
    )

    print("[Default Fixed Fusion Confusion Matrix]")

    print(
        confusion_matrix(
            targets,
            default_fusion_predictions,
            labels=label_names,
        )
    )

    print(
        f"\n평가 성공: {len(default_fusion_predictions)}개"
    )
    print(
        f"평가 실패: {failed_count}개"
    )


# ==========================================
# 이미지 한 장 테스트 출력
# ==========================================

def run_single_test(
    tokenizer,
    text_model,
    image_processor,
    image_model,
):
    image_path = input(
        "\n이미지 경로: "
    ).strip()

    if not Path(image_path).exists():
        print("이미지 파일이 존재하지 않습니다.")
        return

    try:
        result = predict_multimodal(
            image_path=image_path,
            tokenizer=tokenizer,
            text_model=text_model,
            image_processor=image_processor,
            image_model=image_model,
        )

        print_scores(
            "RoBERTa 텍스트 확률",
            result["text_scores"],
        )

        print_scores(
            "CLIP 이미지 확률",
            result["image_scores"],
        )

        print_scores(
            "최종 결합 확률",
            result["final_scores"],
        )

        print("\n" + "=" * 50)
        print(
            f"최종 카테고리 : "
            f"{result['category']}"
        )
        print(
            f"최종 신뢰도   : "
            f"{result['confidence']:.4f}"
        )
        print(
            f"적용 가중치   : "
            f"Text {result['text_weight']:.2f} / "
            f"Image {result['image_weight']:.2f}"
        )
        print("=" * 50)

    except Exception as error:
        print(f"\n오류 발생: {error}")


# ==========================================
# 실행
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "CaptureMate multimodal screenshot classifier"
        )
    )
    parser.add_argument(
        "--image",
        help="추론할 이미지 파일 경로",
    )
    parser.add_argument(
        "--image-dir",
        help="추론할 이미지들이 들어있는 폴더 경로",
    )
    parser.add_argument(
        "--output-csv",
        default="./outputs/predictions.csv",
        help="--image 또는 --image-dir 결과를 저장할 CSV 경로",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="CSV 저장 없이 터미널에만 출력",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="API에서 파싱하기 좋은 최종 분류 JSON만 stdout으로 출력",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    image_input = args.image or args.image_dir

    if args.json and not image_input:
        raise ValueError(
            "--json은 --image 또는 --image-dir와 함께 사용해야 합니다."
        )

    if args.json:
        with contextlib.redirect_stdout(sys.stderr):
            tokenizer, text_model = (
                load_text_model()
            )
            image_processor, image_model = (
                load_image_model()
            )
            output = run_api_prediction(
                image_input=image_input,
                tokenizer=tokenizer,
                text_model=text_model,
                image_processor=image_processor,
                image_model=image_model,
            )

        print(
            json.dumps(
                output,
                ensure_ascii=False,
            )
        )
        return

    print(f"사용 장치: {DEVICE}")

    print("텍스트 모델 로드 중...")
    tokenizer, text_model = (
        load_text_model()
    )

    print("이미지 모델 로드 중...")
    image_processor, image_model = (
        load_image_model()
    )

    if image_input:
        run_batch_prediction(
            image_input=image_input,
            output_csv=(
                ""
                if args.no_csv
                else args.output_csv
            ),
            tokenizer=tokenizer,
            text_model=text_model,
            image_processor=image_processor,
            image_model=image_model,
        )
        return

    while True:
        print("\n" + "=" * 50)
        print("1. 이미지 한 장 테스트")
        print("2. test.csv 전체 성능 평가")
        print("q. 종료")
        print("=" * 50)

        menu = input(
            "선택: "
        ).strip().lower()

        if menu in {
            "q",
            "quit",
            "exit",
        }:
            print("프로그램을 종료합니다.")
            break

        if menu == "1":
            run_single_test(
                tokenizer=tokenizer,
                text_model=text_model,
                image_processor=image_processor,
                image_model=image_model,
            )

        elif menu == "2":
            try:
                evaluate_multimodal(
                    tokenizer=tokenizer,
                    text_model=text_model,
                    image_processor=image_processor,
                    image_model=image_model,
                )

            except Exception as error:
                print(
                    f"\n평가 중 오류 발생: {error}"
                )

        else:
            print(
                "1, 2 또는 q를 입력해주세요."
            )


if __name__ == "__main__":
    main()
