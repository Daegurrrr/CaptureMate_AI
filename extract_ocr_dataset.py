import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import re
import csv
from pathlib import Path
from collections import Counter
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image


# --------------------------------------------------
# 0. OCR 초기화
# --------------------------------------------------
ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

IMAGE_FOLDER = "images"
OUTPUT_CSV = "./data/full_dataset.csv"
VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


# --------------------------------------------------
# 1. 상수
# --------------------------------------------------
IMPORTANT_PATTERNS = [
    r"\d{1,2}:\d{2}",
    r"\d{1,4}[./-]\d{1,2}([./-]\d{1,4})?",
    r"\d[\d,]*원",
    r"\d+\s*%",
    r"\d+\s*박",
    r"\d+\s*일",
    r"\d{2,4}-\d{3,4}-\d{4}",
    r"http[s]?://",
    r"www\.",
    r"naver\.me",
    r"map\.naver",
    r"booking\.naver",
]

MEANINGFUL_SINGLE_CHARS = {"%", "층", "원", "시", "분", "월", "일", "박"}

CHAR_CORRECTION_IN_NUMBER = str.maketrans({
    "이": "0",
    "일": "1",
    "오": "5",
    "O": "0",
    "l": "1",
    "I": "1",
    "ㅇ": "0",
})


# --------------------------------------------------
# 2. 텍스트 정규화
# --------------------------------------------------
def has_important_pattern(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in IMPORTANT_PATTERNS)


def normalize_price(text: str) -> str:
    text = re.sub(r"[₩\\](?=\d)", "원", text)
    text = re.sub(r"\bW(?=\d[\d,]*)", "원", text)
    text = re.sub(r"원\s*(\d[\d,]*)", r"\1원", text)
    return text


def fix_date_ocr_errors(text: str) -> str:
    date_pattern = (
        r"\d{1,4}[./\-]\d{1,2}[이일오O0ㅇlI\d]*"
        r"(?:[./\-~]\d{1,2}[이일오O0ㅇlI\d]*)?"
    )

    def _correct(m: re.Match) -> str:
        return m.group(0).translate(CHAR_CORRECTION_IN_NUMBER)

    return re.sub(date_pattern, _correct, text)


def normalize_line(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\-{2,}", "-", text)
    text = re.sub(r"\|{2,}", "|", text)

    text = normalize_price(text)
    text = fix_date_ocr_errors(text)

    text = re.sub(r"\b[lI1]\s*5G\b", "5G", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[lI1]\s*LTE\b", "LTE", text, flags=re.IGNORECASE)
    text = re.sub(r"\blLTE\b", "LTE", text, flags=re.IGNORECASE)

    text = re.sub(r"[^\w\s\.\,\/\-\:\~\(\)원%+#@&!]", "", text)

    return text.strip()


# --------------------------------------------------
# 3. bbox 유틸
# --------------------------------------------------
def get_box_xyxy(box):
    arr = np.array(box, dtype=np.float32)

    if arr.ndim == 2 and arr.shape[0] >= 4:
        xs = arr[:, 0]
        ys = arr[:, 1]
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    if arr.ndim == 1 and arr.shape[0] == 4:
        x1, y1, x2, y2 = arr.tolist()
        return float(x1), float(y1), float(x2), float(y2)

    return 0.0, 0.0, 0.0, 0.0


def get_box_center_y(box) -> float:
    _, y1, _, y2 = get_box_xyxy(box)
    return (y1 + y2) / 2.0


def get_text_height(box) -> float:
    _, y1, _, y2 = get_box_xyxy(box)
    return y2 - y1


def calculate_iou(box1, box2) -> float:
    x1_min, y1_min, x1_max, y1_max = get_box_xyxy(box1)
    x2_min, y2_min, x2_max, y2_max = get_box_xyxy(box2)

    inter_x = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_y = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection = inter_x * inter_y

    area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


# --------------------------------------------------
# 4. 필터링 규칙
# --------------------------------------------------
def classify_text_tier(box, image_height: int) -> str:
    ratio = get_text_height(box) / image_height if image_height else 0

    if ratio >= 0.040:
        return "TITLE"
    elif ratio >= 0.020:
        return "BODY"
    elif ratio >= 0.016:
        return "CAPTION"
    else:
        return "NOISE"


def is_statusbar_text(text: str) -> bool:
    text = text.strip()

    patterns = [
        r"^\d{1,2}:\d{2}$",
        r"^\d{1,3}%$",
        r"^(SKT|KT|LG U\+|5G|LTE)$",
        r"^오전\s*\d{1,2}:\d{2}$",
        r"^오후\s*\d{1,2}:\d{2}$",
        r"^[lI1]\s*5G$",
        r"^[lI1]\s*LTE$",
        r"^\d{1,2}:\d{2}\s*\d*\s*(SKT|KT|LG U\+|5G|LTE)?$",
    ]

    return any(re.match(p, text, re.IGNORECASE) for p in patterns)


def is_ui_text(text: str) -> bool:
    text = text.strip()

    exact_matches = {
        "대화를",
        "대화를 시작해보세요",
        "답글 1개 더 보기",
        "메뉴판 이미지로 보기",
        "정보 더보기",
        "답글 달기",
        "댓글",
        "Instagram",
    }

    contains_matches = [
        "대화를 시작해보세요",
        "답글 1개 더 보기",
        "메뉴판 이미지로 보기",
        "정보 더보기",
        "답글 달기",
    ]

    if text in exact_matches:
        return True

    if any(x in text for x in contains_matches):
        return True

    return False


def is_noise_token(text: str) -> bool:
    text = text.strip()

    if not text:
        return True

    if has_important_pattern(text):
        return False

    if re.fullmatch(r"[\W_]+", text):
        return True

    if len(text) == 1 and text not in MEANINGFUL_SINGLE_CHARS:
        return True

    return False


def should_keep_item(item: dict) -> bool:
    text = item["text"].strip()
    score = item["score"]
    tier = item["tier"]

    if not text:
        return False

    if is_statusbar_text(text):
        return False

    if is_ui_text(text):
        return False

    if is_noise_token(text):
        return False

    if tier == "TITLE" and score < 0.50 and not has_important_pattern(text):
        return False

    if tier == "BODY" and score < 0.45 and not has_important_pattern(text):
        return False

    if tier == "CAPTION" and score < 0.55 and not has_important_pattern(text):
        return False

    if tier == "NOISE" and score < 0.65 and not has_important_pattern(text):
        return False

    if text in {"@", "#", "*", "_", "|"}:
        return False

    return True


# --------------------------------------------------
# 5. OCR item 처리
# --------------------------------------------------
def remove_duplicate_boxes(ocr_items: list[dict], iou_threshold: float = 0.4) -> list[dict]:
    results = sorted(ocr_items, key=lambda x: x["score"], reverse=True)
    kept = []

    for candidate in results:
        overlap = any(
            calculate_iou(candidate["box"], k["box"]) >= iou_threshold
            for k in kept
        )
        if not overlap:
            kept.append(candidate)

    return kept


def group_items_by_y(items: list[dict], y_gap_threshold: float = 18.0) -> list[dict]:
    if not items:
        return []

    sorted_items = sorted(items, key=lambda x: get_box_center_y(x["box"]))
    groups = []
    current_group = [sorted_items[0]]

    for item in sorted_items[1:]:
        prev_y = get_box_center_y(current_group[-1]["box"])
        curr_y = get_box_center_y(item["box"])

        if abs(curr_y - prev_y) <= y_gap_threshold:
            current_group.append(item)
        else:
            groups.append(current_group)
            current_group = [item]

    groups.append(current_group)

    merged_items = []

    for group in groups:
        group_sorted = sorted(group, key=lambda x: get_box_xyxy(x["box"])[0])
        line = " ".join(x["text"] for x in group_sorted)
        line = normalize_line(line)

        if not line:
            continue

        if is_statusbar_text(line):
            continue

        if re.fullmatch(r"\d{1,4}", line):
            continue

        if re.fullmatch(r"\d{1,2}:\d{2}", line):
            continue

        dominant_tier = max(
            (x["tier"] for x in group_sorted),
            key=lambda t: {"TITLE": 3, "BODY": 2, "CAPTION": 1, "NOISE": 0}.get(t, 0),
        )

        merged_items.append({
            "text": line,
            "tier": dominant_tier,
        })

    return merged_items


# --------------------------------------------------
# 6. 후처리 및 출력 구조
# --------------------------------------------------
def dedupe_lines(lines: list[str]) -> list[str]:
    seen = set()
    result = []

    for line in lines:
        key = line.strip()
        if key and key not in seen:
            seen.add(key)
            result.append(line)

    return result


def dedupe_short_lines(lines: list[str]) -> list[str]:
    counter = Counter()
    result = []

    for line in lines:
        key = line.strip()
        if len(key.split()) <= 2:
            if counter[key] >= 1:
                continue
            counter[key] += 1

        result.append(line)

    return result


def clean_text_block(lines: list[str]) -> str:
    lines = dedupe_lines(lines)
    lines = dedupe_short_lines(lines)

    text = "\n".join(lines)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def build_structured_output(merged_items: list[dict]) -> dict:
    output = {
        "title": [],
        "body": [],
        "caption": [],
    }

    tier_map = {
        "TITLE": "title",
        "BODY": "body",
        "CAPTION": "caption",
    }

    for item in merged_items:
        key = tier_map.get(item["tier"], "body")
        output[key].append(item["text"])

    output["summary_text"] = " ".join(output["title"] + output["body"])

    return output


# --------------------------------------------------
# 7. OCR 결과 파싱
# --------------------------------------------------
def extract_ocr_fields(result_page: dict):
    texts = result_page.get("rec_texts", [])
    scores = result_page.get("rec_scores", [])
    boxes = (
        result_page.get("dt_polys")
        or result_page.get("rec_boxes")
        or result_page.get("text_boxes")
        or []
    )

    texts = [] if texts is None else list(texts)
    scores = [] if scores is None else list(scores)
    boxes = [] if boxes is None else list(boxes)

    return texts, scores, boxes


def process_image(image_path: str) -> dict:
    with Image.open(image_path) as img:
        _, image_height = img.size

    result = ocr.predict(image_path)

    label = Path(image_path).parent.name

    if not result or not result[0]:
        return {
            "filename": os.path.basename(image_path),
            "image_path": image_path,
            "label": label,
            "layout_type": "classification",
            "structured": {
                "title": [],
                "body": [],
                "caption": [],
                "summary_text": "",
            },
            "cleaned_text": "",
            "debug_items": [],
        }

    texts, scores, boxes = extract_ocr_fields(result[0])

    ocr_items = []

    for text, score, box in zip(texts, scores, boxes):
        text = normalize_line(text)

        if not text:
            continue

        tier = classify_text_tier(box, image_height)

        ocr_items.append({
            "text": text,
            "score": float(score),
            "box": box,
            "tier": tier,
        })

    ocr_items = remove_duplicate_boxes(ocr_items, iou_threshold=0.4)

    filtered_items = [
        item for item in ocr_items
        if should_keep_item(item)
    ]

    merged_items = group_items_by_y(filtered_items, y_gap_threshold=18.0)

    structured = build_structured_output(merged_items)

    all_lines = [item["text"] for item in merged_items]
    cleaned_text = clean_text_block(all_lines)

    return {
        "filename": os.path.basename(image_path),
        "image_path": image_path,
        "label": label,
        "layout_type": "classification",
        "structured": structured,
        "cleaned_text": cleaned_text,
        "debug_items": ocr_items,
    }


def extract_text(image_path: str) -> str:
    output = process_image(image_path)
    return output["cleaned_text"]


# --------------------------------------------------
# 8. CSV 저장
# --------------------------------------------------
def save_results_to_csv(results: list[dict], output_csv: str):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "filename",
                "initial_label",
                "final_label",
                "layout_type",
                "ocr_text",
                "note",
            ]
        )

        writer.writeheader()

        for r in results:
            writer.writerow({
                "image_path": r["image_path"],
                "filename": r["filename"],
                "initial_label": r["label"],
                "final_label": r["label"],
                "layout_type": r["layout_type"],
                "ocr_text": r["cleaned_text"],
                "note": "",
            })


# --------------------------------------------------
# 9. 실행
# --------------------------------------------------
def main():
    image_root = Path(IMAGE_FOLDER)
    all_image_paths = []

    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_EXTS:
            all_image_paths.append(path)

    all_image_paths = sorted(all_image_paths)

    results = []

    for image_path in all_image_paths:
        output = process_image(str(image_path))
        results.append(output)

        print("\n" + "=" * 60)
        print(f"파일명     : {output['filename']}")
        print(f"이미지경로 : {output['image_path']}")
        print(f"자동라벨   : {output['label']}")
        print(f"레이아웃   : {output['layout_type']}")
        print("=" * 60)

        s = output["structured"]

        if s["title"]:
            print("\n[TITLE]")
            for t in s["title"]:
                print(f"  {t}")

        if s["body"]:
            print("\n[BODY]")
            for t in s["body"]:
                print(f"  {t}")

        if s["caption"]:
            print("\n[CAPTION]")
            for t in s["caption"]:
                print(f"  {t}")

        print("\n[최종 정제 텍스트]")
        print(output["cleaned_text"] if output["cleaned_text"] else "남은 텍스트가 없습니다.")

    save_results_to_csv(results, OUTPUT_CSV)

    print("\n" + "=" * 60)
    print(f"CSV 저장 완료: {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()