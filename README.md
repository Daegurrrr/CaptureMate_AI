# CaptureMate AI

CaptureMate AI는 캡쳐 이미지에서 OCR 텍스트를 추출하고, OCR 텍스트 분류 모델과 CLIP 이미지 분류 모델을 함께 사용해 캡쳐 카테고리를 예측하는 멀티모달 추론 파이프라인입니다.

이 레포는 학습된 모델 weight를 직접 포함하지 않습니다. 최종 모델은 Hugging Face Hub에서 다운로드해 사용합니다.

## Overview

```text
Image
  ├─ PaddleOCR → OCR Text → RoBERTa Text Classifier
  └─ Original Image → CLIP Image Classifier
                         │
                         ▼
        Multimodal Fusion
        final_score = text_score * 0.75 + image_score * 0.25
                         │
                         ▼
              Final Category
```

## Categories

| Label | Description |
|---|---|
| `schedule` | 일정, 예약, 티켓, 캘린더 관련 캡쳐 |
| `shopping` | 쇼핑, 상품, 결제, 주문 관련 캡쳐 |
| `place` | 장소, 지도, 매장, 위치 관련 캡쳐 |
| `memo` | 메모, 글, 저장용 텍스트 캡쳐 |
| `unknown` | 추천 액션을 연결하지 않는 기타 캡쳐 |

## Models

| Model | Hugging Face Repo | Purpose |
|---|---|---|
| Text model | `hur03/capturemate-category-classifier-v1-5class` | OCR 텍스트 기반 5-class 분류 |
| Image model | `hur03/capturemate-image-classifier-v1-5class` | 이미지 기반 5-class 분류 |

## Fusion Strategy

최종 분류는 텍스트 모델과 이미지 모델의 softmax 확률을 고정 비율로 결합해 계산합니다.

```text
Text weight  = 0.75
Image weight = 0.25
```

계산식:

```text
final_score = text_score * 0.75 + image_score * 0.25
```

각 카테고리별 `final_score`를 계산한 뒤, 가장 높은 점수를 가진 label을 최종 카테고리로 반환합니다.

## Performance

`data/test.csv` 기준 성능 비교 결과입니다.

### Overall Performance

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Text only | 91.92% | 91.38% |
| Image only | 80.81% | 78.71% |
| Multimodal Fusion | 93.94% | 93.30% |

Text only 대비 Multimodal Fusion은 Accuracy 기준 `+2.02%p`, Macro F1 기준 `+1.92%p` 향상되었습니다.

### Category-wise F1 Comparison

| Category | Text only F1 | Fusion F1 | Change |
|---|---:|---:|---:|
| `schedule` | 88.89% | 88.89% | 0.00%p |
| `shopping` | 98.18% | 100.00% | +1.82%p |
| `place` | 97.14% | 97.14% | 0.00%p |
| `memo` | 92.68% | 92.68% | 0.00%p |
| `unknown` | 80.00% | 87.80% | +7.80%p |

### Why Multimodal Fusion?

CaptureMate에서 `unknown`은 단순한 기타 라벨이 아니라, 추천 액션을 연결하지 않아야 하는 캡쳐를 걸러내는 역할을 합니다.

텍스트 모델만 사용할 경우 OCR 텍스트 일부가 장소, 쇼핑, 일정처럼 보이면 실제로는 추천 액션이 필요 없는 캡쳐도 특정 카테고리로 잘못 분류될 수 있습니다. 이런 경우 사용자는 불필요한 추천 액션을 보게 됩니다.

Multimodal Fusion은 이미지 모델의 시각 정보를 함께 반영해 이러한 오분류를 줄이는 것을 목표로 했습니다. 실험 결과 전체 성능도 소폭 향상되었지만, 특히 `unknown` 카테고리의 F1이 `80.00%`에서 `87.80%`로 개선되어 불필요한 추천 액션을 줄이는 데 효과가 있었습니다.

최종적으로 `Text 75% / Image 25%` 조합을 기본 fusion 비율로 적용했습니다.

## Setup

```bash
pip install -r requirements.txt
```

## Run

기본 실행:

```bash
python predict_multimodal.py
```

실행하면 다음 메뉴가 표시됩니다.

```text
1. 이미지 한 장 테스트
2. test.csv 전체 성능 평가
q. 종료
```

로컬에서 직접 테스트할 때는 보통 `1. 이미지 한 장 테스트`를 사용합니다. 
`2. test.csv 전체 성능 평가`는 로컬에 `data/test.csv`와 평가 이미지가 있을 때만 사용합니다.

## Predict With Image File

사진 파일 하나를 입력해 추론합니다.

```bash
python predict_multimodal.py --image ./new_images/IMG_0001.jpeg
```

이미지 폴더 전체를 입력해 여러 장을 한 번에 추론합니다.

```bash
python predict_multimodal.py --image-dir ./new_images
```

기본 결과 CSV는 아래 경로에 저장됩니다.

```text
./outputs/predictions.csv
```

터미널 출력만 보고 싶으면 `--no-csv` 옵션을 사용합니다.

```bash
python predict_multimodal.py --image-dir ./new_images --no-csv
```

## JSON Output For API

API 연동에 바로 사용할 수 있는 최종 분류 결과만 필요하면 `--json` 옵션을 사용합니다.

```bash
python predict_multimodal.py --image ./new_images/IMG_0001.jpeg --json
```

응답 예시:

```json
{
  "filename": "IMG_0001.jpeg",
  "category": "shopping",
  "confidence": 0.9231,
  "text_weight": 0.75,
  "image_weight": 0.25
}
```

## Backend Integration

현재 앱 API 구조에서는 클라이언트가 이미지를 백엔드로 전달하고, 백엔드에서 OCR 및 멀티모달 분류를 수행합니다.

```text
Client
  → Backend API
    → PaddleOCR
    → Text model
    → Image model
    → Fusion
    → Final category
```

이 AI 레포는 추론 로직과 실험 코드를 관리하는 레포이며, 백엔드 API 서버는 동일한 모델 ID와 fusion 방식을 서버용 코드에 반영해 사용합니다.

즉, API 호출 시 클라이언트가 AI 레포를 직접 실행하는 구조는 아닙니다.

## Environment Variables

기본값을 그대로 쓰면 Hugging Face에서 모델을 자동으로 다운로드합니다.

| Variable | Default | Description |
|---|---|---|
| `CAPTUREMATE_TEXT_MODEL_ID` | `hur03/capturemate-category-classifier-v1-5class` | OCR 텍스트 분류 모델 repo id |
| `CAPTUREMATE_IMAGE_MODEL_ID` | `hur03/capturemate-image-classifier-v1-5class` | CLIP 이미지 분류 모델 repo id |
| `CAPTUREMATE_IMAGE_MODEL_CACHE_DIR` | Hugging Face default cache | 이미지 모델 다운로드 캐시 경로 |
| `CAPTUREMATE_IMAGE_MODEL_DIR` | empty | 로컬 이미지 모델 폴더를 강제로 사용할 때만 지정 |

로컬 실험 모델을 쓰고 싶을 때만 아래처럼 지정합니다.

```bash
CAPTUREMATE_IMAGE_MODEL_DIR=./outputs/best_image_classifier python predict_multimodal.py
```

## Project Structure

```text
CaptureMate_AI
├── predict_multimodal.py          # Hugging Face 모델 다운로드 + 멀티모달 추론
├── extract_ocr_dataset.py         # PaddleOCR 텍스트 추출 유틸
├── requirements.txt
└── README.md
```

## Git Policy

다음 파일은 GitHub에 올리지 않습니다.

| Path | Reason |
|---|---|
| `data/` | 학습/평가 CSV 데이터 |
| `images/` | 원본 캡쳐 이미지 |
| `outputs/` | 로컬 학습 결과와 모델 weight |
| `models/` | 로컬 사전학습 모델 |
| `paddle_ocr_env/`, `.venv/`, `venv/` | Python 가상환경 |

## Notes

- 텍스트 모델은 `transformers`의 `AutoTokenizer`, `AutoModelForSequenceClassification`으로 로드합니다.
- 이미지 모델은 커스텀 `CLIPUIClassifier` 체크포인트입니다.
- Hugging Face에서는 모델 weight만 다운로드합니다.
- 추론 시 입력 이미지는 Hugging Face로 전송하지 않습니다.
- 원본 이미지는 로컬 또는 서버 메모리에서 OCR과 이미지 분류에 사용됩니다.
- Hugging Face 모델 구조나 label 순서를 변경해 새로 업로드하는 경우, 백엔드 추론 코드도 함께 확인해야 합니다.
