# CaptureMate_AI
CaptureMate AI 레포입니다!
CaptureMate AI는 캡쳐 이미지에서 OCR 텍스트를 추출하고, 추출된 텍스트 분석을 통해 사용자의 캡쳐를 자동으로 카테고리 분류하기 위한 AI 파이프라인입니다.

## 🧱 기술 스택

| 구분 | 기술 |
|---|---|
| 언어 | Python |
| OCR | PaddleOCR |
| 분류 모델 | KLUE-RoBERTa |
| Deep Learning Framework | Transformers |
| Dataset Processing | Pandas / Datasets / Scikit-learn |
| Model Training | Hugging Face Trainer |
| Model Hub | Hugging Face Hub |
| Numerical Computing | NumPy |


## 🔄 전체 파이프라인

```txt
캡쳐 이미지 수집
↓
run_pipeline.py 실행
↓
사전학습 모델 다운로드
↓
OCR 텍스트 추출 및 정제
↓
full_dataset.csv 생성
↓
train / validation / test 데이터 분리
↓
분류 모델 학습 및 평가
↓
best classifier 저장
↓
Hugging Face Hub 업로드
↓
백엔드에서 모델 로드 후 OCR 텍스트 분류
```

## 🧩 분류 카테고리

현재 모델은 다음 5개 카테고리를 분류합니다.

| Label | Description |
|---|---|
| schedule | 일정 관련 캡쳐 |
| shopping | 쇼핑/상품/결제 관련 캡쳐 |
| place | 장소/지도/매장 정보 관련 캡쳐 |
| memo | 메모/텍스트 저장용 캡쳐 |
| trash | 분류 가치가 낮거나 불필요한 캡쳐 |

## 📂 프로젝트 구조

```txt
CaptureMate_AI
├── data/                       # 학습용 CSV 데이터 저장
├── images/                     # 원본 캡쳐 이미지 데이터
├── models/                     # 사전학습 모델 저장

├── outputs/
│   ├── checkpoints/            # 학습 중간 checkpoint 저장
│   └── best_classifier/        # 최종 저장된 best model

├── run_pipeline.py             # 전체 AI 파이프라인 실행
├── download_base_model.py      # KLUE-RoBERTa 사전학습 모델 다운로드
├── extract_ocr_dataset.py      # OCR 추출 및 CSV 데이터셋 생성
├── split_dataset.py            # train / validation / test 데이터 분리
├── train_classifier.py         # 카테고리 분류 모델 학습 및 평가
├── upload_model_to_hub.py      # Hugging Face Hub 업로드
└── requirements.txt            # 실행에 필요한 Python 패키지 목록
```


## 🚫 Git에 포함하지 않는 파일

다음 폴더 및 파일은 용량이 크거나 로컬 환경에 의존하므로 Git에 올리지 않습니다.

| Path | Reason |
|---|---|
| `data/` | 학습용 CSV 데이터 |
| `images/` | 원본 캡쳐 이미지 데이터 |
| `models/` | 로컬 사전학습 모델 |
| `outputs/checkpoints/` | 학습 중간 checkpoint |
| `outputs/**/*.safetensors` | 모델 weight 파일 |
| `paddle_ocr_env/` | Python 가상환경 |



## 🧪 학습 결과 예시

### Dataset Size
| Split | Samples |
|---|---:|
| Train | 350 |
| Validation | 75 |
| Test | 75 |

현재 학습 결과는 다음과 같습니다.

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 86.57% | 92.54% |
| Macro F1 | 85.08% | 92.82% |
| Macro Precision | 86.19% | 93.24% |
| Macro Recall | 85.20% | 92.87% |


## 🤗 Hugging Face Model

학습된 모델은 Hugging Face Hub에서 관리합니다.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "capturemate-category-classifier"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
```


## 🚀 모델 사용 방식

백엔드에서는 OCR로 추출한 텍스트를 모델에 입력하여 카테고리를 예측합니다.

```txt
이미지 업로드
↓
PaddleOCR로 OCR 텍스트 추출
↓
학습된 분류 모델에 OCR 텍스트 입력
↓
category / confidence 반환
↓
DB 저장 및 앱에 전달
```

## ▶️ 실행 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 전체 파이프라인 실행

```bash
python run_pipeline.py
```


## 📌 참고

- `data/`, `images/`, `models/`, `outputs/`는 Git에 포함하지 않습니다.
- 모델 공유는 Hugging Face Hub를 통해 진행합니다.
- 백엔드에는 학습 코드가 아니라 업로드된 모델을 로드하는 추론 코드만 연결합니다.