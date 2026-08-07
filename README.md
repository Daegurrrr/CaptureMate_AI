# CaptureMate AI

CaptureMate AI는 캡쳐 이미지에서 OCR 텍스트를 추출하고, OCR 텍스트 분류 모델과 CLIP 이미지 분류 모델을 함께 사용해 캡쳐 카테고리를 예측하는 추론용 파이프라인입니다.

이 GitHub 버전은 학습된 모델 파일을 포함하지 않습니다. 모델은 Hugging Face Hub에서 다운로드해 사용합니다.

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
| OCR text classifier | `hur03/capturemate-category-classifier-v1-5class` | OCR 텍스트 기반 5-class 분류 |
| CLIP image classifier | `hur03/capturemate-image-classifier-v1-5class` | 이미지 기반 5-class 분류 |

현재 멀티모달 결합 비율은 다음과 같습니다.

```txt
Text  = 0.7
Image = 0.3
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python predict_multimodal.py
```

실행하면 다음 메뉴가 표시됩니다.

```txt
1. 이미지 한 장 테스트
2. test.csv 전체 성능 평가
q. 종료
```

GitHub 배포/추론 환경에서는 보통 `1. 이미지 한 장 테스트`를 사용합니다. `2. test.csv 전체 성능 평가`는 로컬에 `data/test.csv`와 평가 이미지가 있을 때만 사용합니다.

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

```txt
CaptureMate_AI
├── predict_multimodal.py          # Hugging Face 모델 다운로드 + 멀티모달 추론
├── extract_ocr_dataset.py         # PaddleOCR 텍스트 추출 유틸
└── requirements.txt
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

- 텍스트 모델은 `transformers`의 `AutoTokenizer`, `AutoModelForSequenceClassification`으로 바로 로드합니다.
- 이미지 모델은 커스텀 `CLIPUIClassifier` 체크포인트입니다. Hugging Face에서 `image_classifier.pt`와 `preprocessor_config.json`을 다운로드한 뒤, CaptureMate 코드의 모델 클래스로 로드합니다.
- 원본 이미지는 Hugging Face나 외부 서버로 업로드하지 않습니다. 추론 시 로컬/서버 메모리에서 OCR과 이미지 분류를 수행합니다.
