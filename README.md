# CaptureMate_AI
CaptureMate AI 레포입니다!
CaptureMate AI는 캡쳐 이미지에서 OCR 텍스트를 추출하고, 추출된 텍스트를 기반으로 캡쳐 카테고리를 분류하는 모델 학습 및 업로드 파이프라인입니다.

---

## 📌 현재 상태

현재 AI 파이프라인은 다음 단계까지 구현되어 있습니다.

1. PaddleOCR 기반 캡쳐 이미지 OCR 텍스트 추출
2. OCR 결과 정제 및 CSV 데이터셋 생성
3. train / validation / test 데이터 분리
4. KLUE-RoBERTa 기반 카테고리 분류 모델 학습
5. 학습된 best model 저장
6. Hugging Face Hub 업로드 스크립트 작성

---

## 🧩 분류 카테고리

현재 모델은 다음 5개 카테고리를 분류합니다.

| Label | Description |
|---|---|
| schedule | 일정 관련 캡쳐 |
| shopping | 쇼핑/상품/결제 관련 캡쳐 |
| place | 장소/지도/매장 정보 관련 캡쳐 |
| memo | 메모/텍스트 저장용 캡쳐 |
| unknown | 분류 가치가 낮거나 불필요한 캡쳐 |

---

## 📁 파일 설명

| File | Description |
|---|---|
| `ocr.py` | 이미지에서 OCR 텍스트를 추출하고 정제하여 CSV 생성 |
| `prepare_dataset_for_train.py` | 전체 CSV 데이터를 train / validation / test로 분리 |
| `test_train.py` | KLUE-RoBERTa 기반 분류 모델 학습 및 평가 |
| `download_model.py` | 사전학습 모델 다운로드용 스크립트 |
| `upload_to_huggingface.py` | 학습 완료된 모델을 Hugging Face Hub에 업로드 |
| `requirements.txt` | 실행에 필요한 Python 패키지 목록 |

---

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

---

## 🔄 전체 파이프라인

```txt
캡쳐 이미지 수집
↓
ocr.py 실행
↓
OCR 텍스트 추출 및 정제
↓
full_dataset.csv 생성
↓
prepare_dataset_for_train.py 실행
↓
train / validation / test 데이터 분리
↓
test_train.py 실행
↓
분류 모델 학습 및 평가
↓
outputs/best_classifier 저장
↓
upload_to_huggingface.py 실행
↓
Hugging Face Hub에 모델 업로드
↓
백엔드에서 모델 로드 후 OCR 텍스트 분류
```

---

## 🧪 학습 결과 예시

현재 학습 결과는 다음과 같습니다.

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 86.57% | 92.54% |
| Macro F1 | 85.08% | 92.82% |
| Macro Precision | 86.19% | 93.24% |
| Macro Recall | 85.20% | 92.87% |

---

## 🤗 Hugging Face Model

학습된 모델은 Hugging Face Hub에서 관리합니다.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "HUGGINGFACE_USERNAME/capturemate-category-classifier"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
```

---

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

## ▶️ 실행 순서

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. OCR 데이터셋 생성

```bash
python ocr.py
```

### 3. 데이터셋 분리

```bash
python prepare_dataset_for_train.py
```

### 4. 모델 학습

```bash
python test_train.py
```

### 5. Hugging Face 업로드

```bash
python upload_to_huggingface.py
```

---

```md
## 📌 참고

- `data/`, `images/`, `models/`, `outputs/`는 Git에 포함하지 않습니다.
- 모델 공유는 Hugging Face Hub를 통해 진행합니다.
- 백엔드에는 학습 코드가 아니라 업로드된 모델을 로드하는 추론 코드만 연결합니다.
```