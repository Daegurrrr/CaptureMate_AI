from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
import os


model = SentenceTransformer("jhgan/ko-sbert-sts")


categories = {
    "기타": "일반 텍스트 화면 캡처 분류가 애매한 내용 알 수 없는 정보 이모티콘 앱 화면 UI",
    "장소": "맛집 카페 음식점 식당 지도 위치 주소 방문 리뷰 영업시간 길찾기",
    "메모": "메모 필기 공부 정리 개념 설명 자료 노트 해야할일 기록 요약 정보 정리 혜택 정리",
    "일정": "날짜 시간 일정 예약 기간 마감 종료 오픈 팝업 행사 이벤트 콘서트 페스티벌",
    "쇼핑": "상품 구매 주문 가격 옵션 배송 판매 쇼핑몰 브랜드 사이즈 색상 후기 재고 결제"
}

category_names = list(categories.keys())
category_texts = list(categories.values())
category_embeddings = model.encode(category_texts)


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^\w가-힣\s%:/\-~.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sbert_classify(text: str):
    query_embedding = model.encode([text])
    scores = cosine_similarity(query_embedding, category_embeddings)[0]
    best_idx = scores.argmax()
    return category_names[best_idx], scores


PLACE_KEYWORDS = [
    "맛집", "카페", "음식점", "식당", "주소", "위치", "지도", "길찾기",
    "포장", "예약", "단체", "방문", "리뷰", "영업시간", "매장", "지점",
    "브런치", "디저트", "영업종료", "역", "출구","층",
    "본점", "2호점", "역점", "밥집", "숙소", "술집", "호텔", "영업 중",
    "커피", "병원", "의원", "영업 전", "캠프", "캠핑장", "주점", "휴무",
    "카라반", "이비인후과", "내과", "정형외과", "성형외과", "내원","펜션","글램핑","카라반","hotel",
    "camp","pension"
]

MEMO_KEYWORDS = [
    "메모", "정리", "필기", "공부", "개념", "설명", "자료", "노트",
    "기록", "요약", "중요", "체크", "해야", "할일", "todo",
    "안녕하세요", "안내드립니다", "드림", "됩니다", "기자", "기사",
    "조합", "돌리기", "방법", "시술", "레시피", "알려줍니다", "알려",
    "이유", "표현", "화법", "궁금", "꿀팁", "안내", "서비스", "소개",
    "Tip", "tip", "축하", "목록", "번역", "제작", "운동영상","추천하는"
    "다이어트영상", "붓기", "매월", "매주", "인증", "필요", "교육",
    "가능", "증빙", "수령", "운임", "청소년", "대학생", "학생", "관리"
    "혜택", "할인 정보", "정보", "인증 필요", "구매 가능", "학생 카드",
    "교육 할인", "최대","추천","갓성비","갓성비 시술","병원 추천","학생 할인","대학생 할인",
    "소개해","놀이기구"
]

SCHEDULE_KEYWORDS = [
    "일정", "예약", "날짜", "시간", "기간", "마감", "종료", "오픈",
    "팝업", "행사", "이벤트", "프로모션", "부터", "운영기간", "선착순",
    "진행중", "대기", "특가", "까지", "사전예약", "현장대기", "신청접수",
    "공개", "콘서트", "페스티벌", "블랙프라이데이", "블프", "하루만","1매","구매 가능"
    "supersale", "sale", "쿠폰팩","할인정보","내한","내한 콘서트","식사권","쿠폰","유효기간","쿠폰 유효기간",
    "할인 쿠폰"
]

SHOPPING_KEYWORDS = [
    "상품", "구매", "주문", "가격", "옵션", "배송", "판매", "쇼핑",
    "쇼핑몰", "브랜드", "사이즈", "색상", "후기", "재고", "장바구니",
    "결제", "원", "ml", "환불", "쿠폰", "디카", "다이소", "스토어",
    "구매처", "쿠팡", "상의", "하의", "패션", "구매하기", "하울", "리필"
    "won", "폰케이스", "shop", "가방", "아우터", "제품", "화장품",
    "코디", "즉시 구매가", "신상", "무료배송", "무료반품", "오늘드림","제품","제품 정보",
    "정보","구매처","스토어","리필","쇼핑","상의","하의","신발","코디"
]


def default_scores():
    return {
        "기타": 1,
        "장소": 0,
        "메모": 0,
        "일정": 0,
        "쇼핑": 0
    }


def count_keywords(text: str, keywords: list[str]) -> int:
    return sum(1 for k in keywords if k in text)


def has_date_or_time(text: str) -> bool:

    if re.search(r"\d+\s?(분|시간|일|주)\s?전", text):
        return False
    
    # 타임세일/구매 카운트다운은 일정 시간 아님
    if re.search(
        r"(타임세일|구매|남은시간|남은 시간|카운트다운).{0,10}\d{1,2}:\d{2}",
        text
    ):
        return False

    patterns = [
        r"\d{1,2}월\s?\d{1,2}일",
        r"\d{4}[./-]\d{1,2}[./-]\d{1,2}",
        r"\d{1,2}/\d{1,2}",   # 추가
        r"\d{1,2}:\d{2}",
        r"(오전|오후)\s?\d{1,2}:\d{2}",
        r"\d{1,2}시",
        r"\d+일간"
    ]

    return any(re.search(p, text) for p in patterns)


def has_discount_or_event(text: str) -> bool:

    # 배터리 퍼센트 제외
    battery_patterns = [
        r"배터리",
        r"충전",
        r"LTE",
        r"SKT",
        r"5G"
    ]

    if any(k in text for k in battery_patterns):
        text = re.sub(r"\d+\s?%", "", text)

    patterns = [
        r"\d+\s?%\s?(할인|쿠폰|적립|세일|OFF)",
        r"\d[\d,]*\s?원\s?할인",
        r"할인",
        r"혜택",
        r"쿠폰",
        r"프로모션",
        r"선착순",
        r"세일"
    ]

    return any(re.search(p, text, re.IGNORECASE) for p in patterns)

def has_price(text: str) -> bool:
    return bool(re.search(r"\d[\d,]*\s?원", text))


def looks_like_place(text: str) -> bool:
    if any(k in text for k in ["식사권", "쿠폰", "할인권", "유효기간","시술","갓성비"]):
        return False
    
    return any(k in text for k in [
        "카페", "맛집", "식당", "음식점", "주소", "위치", "지도", "길찾기",
        "영업시간", "매장", "병원", "의원", "숙소", "휴무", "주점",
        "라스트오더", "카라반", "캠핑장", "호텔", "이비인후과", "내과",
        "정형외과", "내원"
    ])


def looks_like_memo(text: str) -> bool:
    return any(k in text for k in [
        "메모", "정리", "필기", "공부", "개념", "설명", "자료", "노트",
        "요약", "안녕하세요", "안내드립니다", "드림", "기자", "뉴스",
        "기사", "레시피", "방법", "팁", "꿀팁", "영상", "유튜브",
        "매월", "매주", "인증", "교육", "증빙", "수령", "운임",
        "청소년", "대학생", "학생", "혜택","추천","갓성비","소개해","놀이기구"
        "갓성비 시술","병원 추천","학생 할인","대학생 할인","관리","tip","Tip","추천하는"
    ])


def looks_like_popup_or_schedule(text: str) -> bool:
    return any(k in text for k in [
        "팝업", "팝업스토어", "행사", "이벤트", "운영기간", "예약",
        "마감", "오픈", "특가", "프로모션", "페스티벌",
        "쿠폰팩", "콘서트", "내한","쿠폰","유효기간","할인","할인정보","까지",
        "내한 콘서트","식사권","쿠폰 유효기간","쿠폰 유효기간"
    ])


def looks_like_shopping(text: str) -> bool:
    return has_price(text) or any(k in text for k in [
        "상품", "구매", "주문", "옵션", "배송", "쇼핑몰", "사이즈",
        "색상", "결제", "환불", "구매하기", "무료배송", "무료반품",
        "하울", "화장품", "즉시 구매가","제품","제품 정보","정보","구매처",
        "스토어","추천","리필","패션","상의","하의","신발","코디"
    ])


def is_too_short_text(text: str) -> bool:
    text = text.strip()

    # 장소 정보는 짧아도 기타 처리하지 않음
    if any(k in text for k in [
        "호텔", "캠핑", "캠핑장", "글램핑", "펜션", "카라반",
        "숙소", "카페", "맛집", "식당", "주소", "위치",
        "역", "공원"
    ]):
        return False

    # 쇼핑 정보도 짧아도 기타 처리하지 않음
    if has_price(text) or any(k in text for k in [
        "상품", "제품", "신발", "스니커즈", "상의", "하의",
        "구매", "쿠폰받기", "장바구니", "사이즈"
    ]):
        return False

    # 메모/정보글도 짧아도 기타 처리하지 않음
    if any(k in text for k in [
        "게시판", "Tip", "tip", "방법", "공부", "경험",
        "정리", "필기", "개념", "설명", "자료", "노트",
        "추천", "비교", "시술", "정보"
    ]):
        return False

    if has_date_or_time(text):
        return False

    if len(text) < 5:
        return True

    if len(text.split()) <= 3 and len(text) < 15:
        return True

    return False


def looks_like_ui_screen(text: str) -> bool:
    ui_keywords = [
        "kakaoemoticon", "이모티콘", "채팅방에서 써보기",
        "하나만 사기", "플러스로 다 써보기", "상품 안내",
        "해지 및 환불 안내", "추천", "인기", "스타일",
        "더보기", "답글", "로그인", "회원가입","공유 앨범",
        "배경화면 제안","중복된 항목","최근 삭제된 항목","가려진 항목",
        "시작하기","다음에 하기"
    ]

    count = sum(1 for k in ui_keywords if k in text)
    return count >= 2


def final_classify(text: str, return_scores: bool = False):
    text = clean_text(text)

    if not text:
        if return_scores:
            return "기타", default_scores()
        return "기타"

    if is_too_short_text(text):
        if return_scores:
            return "기타", default_scores()
        return "기타"
    

    # 인스타/스토리/짧은 일상 캡처는 기타
    if any(k in text for k in ["메시지 보내기", "스토리", "Instagram", "instagram"]):
        if not any(k in text for k in [
            "할인", "쿠폰", "가격", "원", "주소", "위치",
            "예약", "기간", "마감", "이벤트", "팝업",
            "운동", "방법", "정리", "추천", "후기"
        ]):
            if return_scores:
                return "기타", default_scores()
            return "기타"
    

    # -------------------------
    # 1. 진짜 UI 화면은 기타
    # -------------------------
    if looks_like_ui_screen(text):
        if return_scores:
            return "기타", default_scores()
        return "기타"
    

    # -------------------------
    # 2. 콘서트/공연 일정표 우선 처리
    # -------------------------
    date_count = len(re.findall(
        r"\d{1,2}월\s?\d{1,2}일|\d{1,2}/\d{1,2}|\d{4}[./-]\d{1,2}[./-]\d{1,2}",
        text
    ))

    if date_count >= 2 and any(k in text for k in ["콘서트", "라인업", "공연", "내한"]):
        if return_scores:
            scores = default_scores()
            scores["기타"] = 0
            scores["일정"] = 10
            return "일정", scores
        return "일정"

    # -------------------------
    # 3. 할인/이벤트 기간 우선 처리
    # -------------------------
    if (
        has_date_or_time(text)
        and ("까지" in text or "기간" in text or "이벤트 기간" in text or "유효기간" in text)
        and ("할인" in text or "이벤트" in text or "쿠폰" in text or "혜택" in text)
    ):
        if return_scores:
            scores = default_scores()
            scores["기타"] = 0
            scores["일정"] = 10
            return "일정", scores
        return "일정"

    # -------------------------
    # 4. 예매/티켓오픈 일정 우선 처리
    # -------------------------
    if (
        ("티켓오픈" in text or "예매" in text or "예약" in text)
        and ("기간" in text or has_date_or_time(text))
    ):
        if return_scores:
            scores = default_scores()
            scores["기타"] = 0
            scores["일정"] = 10
            return "일정", scores
        return "일정"

    # -------------------------
    # 5. 장소 + 일정이면 일정 우선
    # -------------------------
    if (
        looks_like_place(text)
        and has_date_or_time(text)
        and any(k in text for k in [
            "기간", "까지", "예매", "예약", "티켓오픈",
            "행사", "이벤트", "할인", "쿠폰",
            "콘서트", "공연", "축제"
        ])
    ):
        if return_scores:
            scores = default_scores()
            scores["기타"] = 0
            scores["일정"] = 10
            return "일정", scores
        return "일정"

    # -------------------------
    # 6. 장소만 있으면 장소
    # -------------------------
    if looks_like_place(text):
        scores = default_scores()
        scores["기타"] = 0
        scores["장소"] = 10
        if return_scores:
            return "장소", scores
        return "장소"

    # 여기 아래는 기존 SBERT/점수 계산 그대로 두면 됨

    sbert_result, _ = sbert_classify(text)

    scores = {
        "기타": 0,
        "장소": 0,
        "메모": 0,
        "일정": 0,
        "쇼핑": 0
    }

    scores["장소"] += count_keywords(text, PLACE_KEYWORDS)
    scores["메모"] += count_keywords(text, MEMO_KEYWORDS)
    scores["일정"] += count_keywords(text, SCHEDULE_KEYWORDS)
    scores["쇼핑"] += count_keywords(text, SHOPPING_KEYWORDS)

    if looks_like_place(text):
        scores["장소"] += 3

    if looks_like_memo(text):
        scores["메모"] += 3

    if has_date_or_time(text) and looks_like_popup_or_schedule(text):
        if looks_like_shopping(text):
            scores["일정"] += 6
            scores["쇼핑"] += 2
        else:
            scores["일정"] += 6
    elif has_date_or_time(text):
        scores["일정"] += 1

    # 할인/혜택만 있다고 무조건 일정으로 보내지 않음
    if has_discount_or_event(text):
        scores["메모"] += 3

        if has_price(text) or looks_like_shopping(text):
            scores["쇼핑"] += 1

    if looks_like_popup_or_schedule(text):
        scores["일정"] += 3

    if looks_like_shopping(text):
        scores["쇼핑"] += 3

    # 장소 + 일정 보정
    if scores["장소"] > 0 and scores["일정"] > 0:
        if has_date_or_time(text) or looks_like_popup_or_schedule(text):
            scores["일정"] += 5

    # 쇼핑 + 일정 보정
    if scores["쇼핑"] > 0 and scores["일정"] > 0:
        if has_discount_or_event(text) and has_date_or_time(text):
            scores["일정"] += 6
        # 할인 정보만 있고 기간 없으면 메모
        if has_discount_or_event(text) and not has_date_or_time(text):
            scores["메모"] += 2

    # 쇼핑 + 장소 보정
    if scores["쇼핑"] > 0 and scores["장소"] > 0:
        if has_price(text):
            scores["쇼핑"] += 2

    # 정보성 혜택 정리 보정
    if looks_like_memo(text) and "혜택" in text and not has_price(text):
        scores["메모"] += 4

    # 이모티콘/앱 UI 보정
    if "이모티콘" in text:
        scores["기타"] += 4
        scores["쇼핑"] -= 2
    
    if has_price(text) and not has_date_or_time(text):
        scores["쇼핑"] += 3

    # 상품 상세 화면 신호
    if has_price(text) and any(k in text for k in ["리뷰", "쿠폰받기", "장바구니", "구매 가능 가격", "일시품절","타임세일"]):
        scores["쇼핑"] += 5
        scores["일정"] = max(0,scores["일정"]-3)

    scores[sbert_result] += 2

    best_category = choose_best_category(scores, text)
    best_score = scores[best_category]

    if best_score <= 1:
        best_category = "기타"

    if return_scores:
        return best_category, scores

    return best_category

def choose_best_category(scores, text):
    max_score = max(scores.values())
    candidates = [k for k, v in scores.items() if v == max_score]

    if len(candidates) == 1:
        return candidates[0]

    # 동점일 때만 우선순위 처리
    if "쇼핑" in candidates and "메모" in candidates:
        if looks_like_shopping(text):
            return "쇼핑"

    if "일정" in candidates and "쇼핑" in candidates:
        if has_date_or_time(text) and not has_price(text):
            return "일정"

        if has_price(text):
            return "쇼핑"

    if "기타" in candidates:
        return "기타"

    return candidates[0]

def normalize_scores(scores: dict) -> dict:
    values = np.array(list(scores.values()), dtype=float)
    values = values - np.max(values)
    exp_values = np.exp(values)
    probs = exp_values / exp_values.sum()
    return dict(zip(scores.keys(), probs))


def confidence_level(confidence: float) -> str:
    if confidence >= 0.75:
        return "높음"
    elif confidence >= 0.45:
        return "보통"
    else:
        return "낮음"


def final_classify_with_confidence(text: str):
    category, scores = final_classify(text, return_scores=True)
    probs = normalize_scores(scores)
    confidence = probs[category]
    level = confidence_level(confidence)
    return category, confidence, level, probs


def extract_entities(text: str):
    result = {}

    discounts = re.findall(r"\d+\s?%", text)
    if discounts:
        result["discount"] = discounts

    dates = re.findall(r"\d{1,2}월\s?\d{1,2}일", text)
    if dates:
        result["date"] = dates

    times = re.findall(r"(?:오전|오후)?\s?\d{1,2}:\d{2}", text)
    if times:
        result["time"] = times

    prices = re.findall(r"\d[\d,]*\s?원", text)
    if prices:
        result["price"] = prices

    return result


if __name__ == "__main__":
    for file in sorted(os.listdir()):
        if not (file.startswith("ocr_") and file.endswith(".txt")):
            continue

        with open(file, "r", encoding="utf-8") as f:
            text = f.read()

        print("\n" + "=" * 50)
        print(f"파일: {file}")

        if not text.strip():
            print("결과: 텍스트 없음")
            continue

        category, confidence, level, probs = final_classify_with_confidence(text)
        entities = extract_entities(text)

        print(f"카테고리: {category}")
        print(f"신뢰도: {confidence:.2%} ({level})")

        print("[카테고리별 점수]")
        for k, v in probs.items():
            print(f"- {k}: {v:.2%}")

        print("[추출 정보]")
        print("할인:", entities.get("discount", []))
        print("날짜:", entities.get("date", []))
        print("시간:", entities.get("time", []))
        print("가격:", entities.get("price", []))