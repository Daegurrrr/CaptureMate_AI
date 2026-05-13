import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import re
from collections import Counter
from paddleocr import PaddleOCR
from PIL import Image

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

IMAGE_FOLDER = "images"

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


def normalize_line_light(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\-{2,}", "-", text)
    text = re.sub(r"\|{2,}", "|", text)

    text = normalize_price(text)
    text = fix_date_ocr_errors(text)

    # 상태바 OCR 오인식 보정
    text = re.sub(r"\b[lI1]\s*5G\b", "5G", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[lI1]\s*LTE\b", "LTE", text, flags=re.IGNORECASE)
    text = re.sub(r"\blLTE\b", "LTE", text, flags=re.IGNORECASE)

    text = re.sub(r"[^\w\s\.\,\/\-\:\~\(\)원%+#@&!]", "", text)

    return text.strip()


def get_box_center_y(box: list) -> float:
    return (box[0][1] + box[2][1]) / 2.0


def get_text_height(box: list) -> float:
    y_coords = [pt[1] for pt in box]
    return max(y_coords) - min(y_coords)


def calculate_iou(box1: list, box2: list) -> float:
    def to_rect(box):
        xs = [int(p[0]) for p in box]
        ys = [int(p[1]) for p in box]
        return min(xs), min(ys), max(xs), max(ys)

    x1_min, y1_min, x1_max, y1_max = to_rect(box1)
    x2_min, y2_min, x2_max, y2_max = to_rect(box2)

    inter_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection = inter_x * inter_y

    area1 = max(0, x1_max - x1_min) * max(0, y1_max - y1_min)
    area2 = max(0, x2_max - x2_min) * max(0, y2_max - y2_min)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def classify_text_tier(box: list, image_height: int) -> str:
    ratio = get_text_height(box) / image_height

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


# 함수명 통일: should_keep_for_classification에서 이 이름을 사용함
def is_noise_token_light(text: str) -> bool:
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


def remove_duplicate_boxes(ocr_results: list[dict], iou_threshold: float = 0.4) -> list[dict]:
    results = sorted(ocr_results, key=lambda x: x["score"], reverse=True)
    kept = []

    for candidate in results:
        overlap = any(
            calculate_iou(candidate["box"], k["box"]) >= iou_threshold
            for k in kept
        )
        if not overlap:
            kept.append(candidate)

    return kept


def should_keep_for_classification(item: dict) -> bool:
    text = item["text"].strip()
    score = item["score"]
    tier = item["tier"]

    if not text:
        return False

    if is_statusbar_text(text):
        return False

    if is_ui_text(text):
        return False

    if is_noise_token_light(text):
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


def group_lines_by_y(items: list[dict], y_gap_threshold: float = 18.0) -> list[str]:
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

    merged_lines = []
    for group in groups:
        group_sorted = sorted(group, key=lambda x: x["box"][0][0])
        line = " ".join(x["text"] for x in group_sorted)
        line = normalize_line_light(line)
        if line:
            merged_lines.append(line)

    return merged_lines


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


def build_classification_text(lines: list[str]) -> str:
    lines = dedupe_lines(lines)
    lines = dedupe_short_lines(lines)

    text = "\n".join(lines)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def extract_ocr_fields(result_page: dict):
    texts = result_page.get("rec_texts", [])
    scores = result_page.get("rec_scores", [])
    boxes = (
        result_page.get("dt_polys")
        or result_page.get("rec_boxes")
        or result_page.get("text_boxes")
        or []
    )
    return texts, scores, boxes


def process_image_for_classification(image_path: str) -> dict:
    with Image.open(image_path) as img:
        _, image_height = img.size

    result = ocr.predict(image_path)

    if not result or not result[0]:
        return {
            "filename": os.path.basename(image_path),
            "classification_text": "",
            "raw_lines": [],
        }

    texts, scores, boxes = extract_ocr_fields(result[0])

    ocr_items = []
    for text, score, box in zip(texts, scores, boxes):
        text = normalize_line_light(text)
        if not text:
            continue

        tier = classify_text_tier(box, image_height)

        ocr_items.append({
            "text": text,
            "score": score,
            "box": box,
            "tier": tier,
        })

    ocr_items = remove_duplicate_boxes(ocr_items, iou_threshold=0.4)

    filtered_items = [
        item for item in ocr_items
        if should_keep_for_classification(item)
    ]

    raw_lines = group_lines_by_y(filtered_items, y_gap_threshold=18.0)

    # 줄 단계에서 한 번 더 제거
    cleaned_lines = []
    for line in raw_lines:
        line_stripped = line.strip()

        if is_statusbar_text(line_stripped):
            continue

        # 단독 숫자 줄 제거
        if re.fullmatch(r"\d{1,4}", line_stripped):
            continue

        # 단독 시간 제거
        if re.fullmatch(r"\d{1,2}:\d{2}", line_stripped):
            continue

        cleaned_lines.append(line)

    classification_text = build_classification_text(cleaned_lines)

    return {
        "filename": os.path.basename(image_path),
        "classification_text": classification_text,
        "raw_lines": cleaned_lines,
    }

def extract_text(image_path):
    output = process_image_for_classification(image_path)
    return output["classification_text"]

def main():
    for filename in sorted(os.listdir(IMAGE_FOLDER)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_path = os.path.join(IMAGE_FOLDER, filename)
        output = process_image_for_classification(image_path)

        print("\n" + "=" * 60)
        print(f"파일명 : {output['filename']}")
        print("=" * 60)

        print("\n[분류 전 텍스트]")
        print(output["classification_text"] if output["classification_text"] else "남은 텍스트가 없습니다.")

        save_name = os.path.join("outputs", f"ocr_{filename}.txt")
        
        with open(save_name, "w", encoding="utf-8") as f:
            f.write(output["classification_text"])

        print(f"[저장 완료] {save_name}")


if __name__ == "__main__":
    main()