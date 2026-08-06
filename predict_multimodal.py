from pathlib import Path

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

from extract_ocr_dataset import extract_text


# ==========================================
# 설정
# ==========================================

TEXT_MODEL_DIR = "./outputs/best_classifier"

IMAGE_MODEL_DIR = "./outputs/best_image_classifier"
IMAGE_CHECKPOINT_PATH = (
    "./outputs/best_image_classifier/image_classifier.pt"
)

TEST_CSV_PATH = "./data/test.csv"

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
    predictions = []

    text_predictions = []
    image_predictions = []

    failed_count = 0
    total_count = len(test_df)

    print("\n" + "=" * 60)
    print(f"멀티모달 전체 평가 시작: {total_count}개")
    print(
        f"Fusion 비율: "
        f"Text {TEXT_WEIGHT} / Image {IMAGE_WEIGHT}"
    )
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
                predicted_label,
                _,
                _,
            ) = fuse_predictions(
                text_scores=text_scores,
                image_scores=image_scores,
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
            predictions.append(predicted_label)

            text_predictions.append(
                text_label
            )

            image_predictions.append(
                image_label
            )

            mark = (
                "O"
                if target_label == predicted_label
                else "X"
            )

            print(
                f"[{position}/{total_count}] "
                f"{mark} "
                f"정답={target_label}, "
                f"Text={text_label}, "
                f"Image={image_label}, "
                f"Fusion={predicted_label}"
            )

        except Exception as error:
            failed_count += 1

            print(
                f"[{position}/{total_count}] "
                f"처리 실패: {image_path}"
            )
            print(f"오류: {error}")

    if not predictions:
        print("\n평가 가능한 데이터가 없습니다.")
        return

    label_names = list(
        LABEL2ID.keys()
    )

    fusion_accuracy = accuracy_score(
        targets,
        predictions,
    )

    fusion_macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    fusion_precision = precision_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    fusion_recall = recall_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    text_accuracy = accuracy_score(
        targets,
        text_predictions,
    )

    text_macro_f1 = f1_score(
        targets,
        text_predictions,
        average="macro",
        zero_division=0,
    )

    image_accuracy = accuracy_score(
        targets,
        image_predictions,
    )

    image_macro_f1 = f1_score(
        targets,
        image_predictions,
        average="macro",
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("모델별 성능 비교")
    print("=" * 60)

    print("\n[RoBERTa Text]")
    print(f"Accuracy : {text_accuracy:.4f}")
    print(f"Macro F1 : {text_macro_f1:.4f}")

    print("\n[CLIP Image]")
    print(f"Accuracy : {image_accuracy:.4f}")
    print(f"Macro F1 : {image_macro_f1:.4f}")

    print("\n[Multimodal Fusion]")
    print(f"Accuracy        : {fusion_accuracy:.4f}")
    print(f"Macro F1        : {fusion_macro_f1:.4f}")
    print(f"Macro Precision : {fusion_precision:.4f}")
    print(f"Macro Recall    : {fusion_recall:.4f}")

    print("\n[Multimodal Classification Report]")

    print(
        classification_report(
            targets,
            predictions,
            labels=label_names,
            target_names=label_names,
            digits=4,
            zero_division=0,
        )
    )

    print("[Multimodal Confusion Matrix]")

    print(
        confusion_matrix(
            targets,
            predictions,
            labels=label_names,
        )
    )

    print(
        f"\n평가 성공: {len(predictions)}개"
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
        print("=" * 50)

    except Exception as error:
        print(f"\n오류 발생: {error}")


# ==========================================
# 실행
# ==========================================

def main():
    print(f"사용 장치: {DEVICE}")

    print("텍스트 모델 로드 중...")
    tokenizer, text_model = (
        load_text_model()
    )

    print("이미지 모델 로드 중...")
    image_processor, image_model = (
        load_image_model()
    )

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