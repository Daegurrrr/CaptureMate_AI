from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

MODEL_PATH = "./outputs/best_classifier"

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model.push_to_hub("capturemate-category-classifier")
tokenizer.push_to_hub("capturemate-category-classifier")

print("업로드 완료")