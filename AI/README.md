# OCR Classification Pipeline

## 기능
- PaddleOCR 기반 OCR
- OCR 전처리
- 장소 / 일정 / 쇼핑 / 메모 / 기타 분류

## 실행 방법

```bash
pip install -r requirements.txt
python ocr_pipeline.py
```

## 사용 방법

```python
from ocr_pipeline import process_image

result = process_image("images/test.jpg")

print(result)
```

## 반환 형식

```python
{
  "text": "전처리된 OCR 텍스트",
  "category": "일정",
  "confidence": "100.0%",
  "scores": {
    "기타": 0,
    "장소": 0,
    "메모": 0,
    "일정": 10,
    "쇼핑": 0
  }
}
```