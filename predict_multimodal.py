from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)

from extract_ocr_dataset import extract_text


# ==========================================
# 설정
# ==========================================

TEXT_MODEL_DIR = "./outputs/best_classifier"

IMAGE_MODEL_DIR = "./outputs/best_image_classifier"
IMAGE_CHECKPOINT_PATH = (
    "./outputs/best_image_classifier/image_classifier.pt"
)

TEXT_WEIGHT = 0.7
IMAGE_WEIGHT = 0.3

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
# image_classifier.py와 구조가 같아야 함
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
    model_path = Path(TEXT_MODEL_DIR)

    if not model_path.exists():
        raise FileNotFoundError(
            f"텍스트 모델 폴더가 없습니다: {model_path}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        TEXT_MODEL_DIR
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(TEXT_MODEL_DIR)
        .to(DEVICE)
    )

    model.eval()

    return tokenizer, model


# ==========================================
# 이미지 모델 로드
# ==========================================

def load_image_model():
    checkpoint_path = Path(
        IMAGE_CHECKPOINT_PATH
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
            IMAGE_MODEL_DIR
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
    clean_text = text.strip()

    if not clean_text:
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

    scores = {
        ID2LABEL[index]: probability.item()
        for index, probability
        in enumerate(probabilities)
    }

    return scores


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

    scores = {
        ID2LABEL[index]: probability.item()
        for index, probability
        in enumerate(probabilities)
    }

    return scores


# ==========================================
# 결과 결합
# ==========================================

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


# ==========================================
# 전체 예측
# ==========================================

def predict_multimodal(
    image_path: str,
    tokenizer,
    text_model,
    image_processor,
    image_model,
):
    print("\n1) OCR 텍스트 추출")

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
    )

    return {
        "image_path": image_path,
        "ocr_text": ocr_text,
        "text_scores": text_scores,
        "image_scores": image_scores,
        "final_scores": final_scores,
        "category": final_label,
        "confidence": final_confidence,
    }


# ==========================================
# 실행
# ==========================================

def main():
    print(f"사용 장치: {DEVICE}")

    print("텍스트 모델 로드 중...")
    tokenizer, text_model = load_text_model()

    print("이미지 모델 로드 중...")
    image_processor, image_model = load_image_model()

    while True:
        image_path = input(
            "\n이미지 경로 (종료: q): "
        ).strip()

        if image_path.lower() in {"q", "quit", "exit"}:
            print("프로그램을 종료합니다.")
            break

        if not Path(image_path).exists():
            print("이미지 파일이 존재하지 않습니다.")
            continue

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
            print(f"최종 카테고리 : {result['category']}")
            print(f"최종 신뢰도   : {result['confidence']:.4f}")
            print("=" * 50)

        except Exception as e:
            print(f"\n오류 발생: {e}")
            
            
if __name__ == "__main__":
    main()