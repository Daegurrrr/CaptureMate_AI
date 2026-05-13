from test_ocr import extract_text
from classify_test import final_classify


def process_image(image_path):
    text = extract_text(image_path)

    category, scores = final_classify(text, return_scores=True)

    total = sum(scores.values())

    if total > 0:
        confidence = round((max(scores.values()) / total) * 100, 2)
    else:
        confidence = 0

    return {
        "text": text,
        "category": category,
        "confidence": f"{confidence}%",
        "scores": scores
    }


if __name__ == "__main__":
    result = process_image("images/IMG_0039.JPG")
    print(result)