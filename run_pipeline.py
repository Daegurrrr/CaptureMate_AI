import subprocess
import sys
from pathlib import Path


STEPS = [
    ("1. KLUE-RoBERTa 사전학습 모델 다운로드", "download_base_model.py"),
    ("2. OCR 텍스트 추출 및 full_dataset.csv 생성", "extract_ocr_dataset.py"),
    ("3. train / valid / test 데이터셋 분리", "split_dataset.py"),
    ("4. 카테고리 분류 모델 학습 및 평가", "train_classifier.py"),
    ("5. 학습된 모델 Hugging Face Hub 업로드", "upload_model_to_hub.py"),
]


def run_step(description: str, script_name: str):
    script_path = Path(script_name)

    if not script_path.exists():
        raise FileNotFoundError(f"실행 파일을 찾을 수 없습니다: {script_name}")

    print("\n" + "=" * 70)
    print(description)
    print("=" * 70)

    result = subprocess.run([sys.executable, script_name])

    if result.returncode != 0:
        raise RuntimeError(f"{script_name} 실행 중 오류가 발생했습니다.")


def main():
    print("CaptureMate AI 전체 파이프라인을 시작합니다.")

    for description, script_name in STEPS:
        run_step(description, script_name)

    print("\n" + "=" * 70)
    print("전체 AI 파이프라인 실행이 완료되었습니다.")
    print("=" * 70)


if __name__ == "__main__":
    main()