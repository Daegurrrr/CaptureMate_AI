from pathlib import Path
from transformers import AutoTokenizer, AutoModel

MODEL_ID = "klue/roberta-base"
SAVE_DIR = Path("./models/klue-roberta-base")

def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("1) 토크나이저 다운로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("2) 모델 다운로드 중...")
    model = AutoModel.from_pretrained(MODEL_ID)

    print("3) 로컬 폴더에 저장 중...")
    tokenizer.save_pretrained(SAVE_DIR)
    model.save_pretrained(SAVE_DIR)

    print(f"완료: {SAVE_DIR.resolve()}")

if __name__ == "__main__":
    main()