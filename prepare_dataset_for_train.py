import os
import pandas as pd
from sklearn.model_selection import train_test_split


# -----------------------------------
# 설정
# -----------------------------------
INPUT_CSV = "./data/full_dataset.csv"   # 원본 전체 CSV
OUTPUT_DIR = "./data"

LABEL2ID = {
    "schedule": 0,
    "shopping": 1,
    "place": 2,
    "memo": 3,
    "trash": 4,
}

RANDOM_STATE = 42


# -----------------------------------
# 1. 데이터 로드
# -----------------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    return df


# -----------------------------------
# 2. 컬럼 정리
# -----------------------------------
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "image_path",
        "filename",
        "initial_label",
        "final_label",
        "layout_type",
        "ocr_text",
        "note",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"필수 컬럼이 없습니다: {col}")

    # 필요한 컬럼만 복사
    data = df.copy()

    # text 컬럼 생성
    data["text"] = data["ocr_text"].astype(str).str.strip()

    # label 컬럼 생성
    data["label"] = data["final_label"].astype(str).str.strip()

    # NaN 문자열 처리
    data["text"] = data["text"].replace("nan", "")
    data["label"] = data["label"].replace("nan", "")

    # 빈 텍스트 제거
    data = data[data["text"] != ""].copy()

    # 빈 라벨 제거
    data = data[data["label"] != ""].copy()

    # 정의된 라벨만 남기기
    data = data[data["label"].isin(LABEL2ID.keys())].copy()

    # label_id 생성
    data["label_id"] = data["label"].map(LABEL2ID)

    # note 결측 처리
    data["note"] = data["note"].fillna("")

    # 학습 및 분석용으로 남길 컬럼
    data = data[
        [
            "image_path",
            "filename",
            "layout_type",
            "note",
            "text",
            "label",
            "label_id",
        ]
    ].copy()

    return data


# -----------------------------------
# 3. train / valid / test 분리
# -----------------------------------
def split_dataset(df: pd.DataFrame):
    """
    비율:
    - train: 70%
    - valid: 15%
    - test : 15%
    """

    # 1차: train 70 / temp 30
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["label_id"],
    )

    # 2차: temp를 valid 15 / test 15로 반반 분리
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["label_id"],
    )

    return train_df, valid_df, test_df


# -----------------------------------
# 4. 저장
# -----------------------------------
def save_splits(train_df, valid_df, test_df, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    valid_path = os.path.join(output_dir, "valid.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    valid_df.to_csv(valid_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    print(f"저장 완료:")
    print(f"  - {train_path}")
    print(f"  - {valid_path}")
    print(f"  - {test_path}")


# -----------------------------------
# 5. 분포 확인
# -----------------------------------
def print_label_distribution(df: pd.DataFrame, name: str):
    print(f"\n[{name}] 데이터 수: {len(df)}")
    print(df["label"].value_counts())
    print(df["label_id"].value_counts().sort_index())


# -----------------------------------
# 6. 실행
# -----------------------------------
def main():
    print("1) 원본 CSV 로드")
    df = load_data(INPUT_CSV)

    print(f"원본 데이터 수: {len(df)}")

    print("2) 데이터 전처리")
    processed_df = preprocess_dataframe(df)

    print(f"전처리 후 데이터 수: {len(processed_df)}")

    print_label_distribution(processed_df, "전체")

    print("3) train / valid / test 분할")
    train_df, valid_df, test_df = split_dataset(processed_df)

    print_label_distribution(train_df, "TRAIN")
    print_label_distribution(valid_df, "VALID")
    print_label_distribution(test_df, "TEST")

    print("4) 저장")
    save_splits(train_df, valid_df, test_df, OUTPUT_DIR)

    print("\n완료되었습니다.")


if __name__ == "__main__":
    main()