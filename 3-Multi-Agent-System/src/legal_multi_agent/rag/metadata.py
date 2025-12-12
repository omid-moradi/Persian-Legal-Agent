from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


def normalize_for_match(text: str) -> str:
    """
    Temporary normalization just for matching:
    - half-space -> space
    - multiple spaces -> single space
    - lowercase
    """
    t = text.replace("\u200c", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()


# ===========================
# Persian word -> number
# ===========================
FARSI_ONES = {
    "صفر": 0,
    "یک": 1,
    "دو": 2,
    "سه": 3,
    "چهار": 4,
    "پنج": 5,
    "شش": 6,
    "هفت": 7,
    "هشت": 8,
    "نه": 9,
}

FARSI_TENS = {
    "ده": 10,
    "یازده": 11,
    "دوازده": 12,
    "سیزده": 13,
    "چهارده": 14,
    "پانزده": 15,
    "شانزده": 16,
    "هفده": 17,
    "هجده": 18,
    "نوزده": 19,
    "بیست": 20,
    "سی": 30,
    "چهل": 40,
    "پنجاه": 50,
    "شصت": 60,
    "هفتاد": 70,
    "هشتاد": 80,
    "نود": 90,
}

FARSI_HUNDREDS = {
    "یکصد": 100,
    "صد": 100,
    "دویست": 200,
    "سیصد": 300,
    "چهارصد": 400,
    "پانصد": 500,
    "ششصد": 600,
    "هفتصد": 700,
    "هشتصد": 800,
    "نهصد": 900,
}

FARSI_ORDINAL_SIMPLE = {
    "اول": "1",
    "دوم": "2",
    "سوم": "3",
    "چهارم": "4",
    "پنجم": "5",
    "ششم": "6",
    "هفتم": "7",
    "هشتم": "8",
    "نهم": "9",
    "دهم": "10",
    "یازدهم": "11",
    "دوازدهم": "12",
    "سیزدهم": "13",
    "چهاردهم": "14",
    "پانزدهم": "15",
    "شانزدهم": "16",
    "هفدهم": "17",
    "هجدهم": "18",
    "نوزدهم": "19",
    "بیستم": "20",
    "نخست": "1",
    "سی‌ام": "30",
    "چهلم": "40",
    "پنجاهم": "50",
    "شصتم": "60",
    "هفتادم": "70",
    "هشتادم": "80",
    "نودم": "90",
    "یکصدم": "100",
}


def farsi_word_to_number(text: str) -> Optional[int]:
    text = text.strip().replace("‌", " ")  # half-space -> space

    if text.isdigit():
        return int(text)

    if text in FARSI_ORDINAL_SIMPLE:
        return int(FARSI_ORDINAL_SIMPLE[text])

    parts = text.split()
    total = 0
    current = 0

    for part in parts:
        if part == "و":
            continue
        if part in FARSI_HUNDREDS:
            current += FARSI_HUNDREDS[part]
        elif part in FARSI_TENS:
            current += FARSI_TENS[part]
        elif part in FARSI_ONES:
            current += FARSI_ONES[part]
        elif part.endswith("م") or part.endswith("مین"):
            base = part.rstrip("مین").rstrip("م")
            if base in FARSI_TENS:
                current += FARSI_TENS[base]
            elif base in FARSI_ONES:
                current += FARSI_ONES[base]

    total += current
    return total if total > 0 else None


def normalize_article_number(text: str) -> str:
    text = text.strip()
    if text.isdigit():
        return text
    try:
        number = farsi_word_to_number(text)
        return str(number) if number is not None else text
    except Exception:
        return text


@dataclass
class ParsedMetadata:
    law_name: Optional[str] = None
    article_number: Optional[str] = None
    article_type: Optional[str] = None  # "اصل" or "ماده"
    has_metadata: bool = False


def load_qavanin_list(qavanin_text: str) -> List[str]:
    lines = [line.strip() for line in qavanin_text.split("\n") if line.strip()]
    return [
        line
        for line in lines
        if line.startswith(("قانون", "لایحه", "آیین‌نامه", "آیین نامه", "دستورالعمل", "احکام"))
    ]


def parse_question_metadata_simple(question: str, qavanin_list: List[str]) -> Dict:
    """
    Extract:
    - law_name (exact/fuzzy)
    - article/principle number (simple regex)
    """
    q = question.strip()
    q_norm = normalize_for_match(q)

    result = ParsedMetadata().__dict__

    # 1) law name exact
    for law in qavanin_list:
        if normalize_for_match(law) in q_norm:
            result["law_name"] = law
            result["has_metadata"] = True
            break

    # 1b) law name fuzzy (lightweight)
    if not result["law_name"]:
        candidates = []
        for law in qavanin_list:
            law_clean = (
                law.replace("لایحه قانونی", "")
                .replace("قانون", "")
                .replace("آیین‌نامه", "")
                .strip()
            )
            law_keywords_raw = [w for w in law_clean.split() if len(w) > 3][:4]
            if not law_keywords_raw:
                continue

            law_keywords_norm = [normalize_for_match(kw) for kw in law_keywords_raw]
            matches = sum(1 for kw_norm in law_keywords_norm if kw_norm in q_norm)

            if matches >= 1:
                match_ratio = matches / len(law_keywords_norm)
                score = (match_ratio * 100) + len(law_keywords_norm)
                candidates.append({"law": law, "matches": matches, "score": score})

        if candidates:
            candidates.sort(key=lambda x: (x["matches"], x["score"]), reverse=True)
            best = candidates[0]
            # heuristic copied from your pipeline
            if best["matches"] >= 2 or len([w for w in best["law"].split() if len(w) > 3]) == 1:
                result["law_name"] = best["law"]
                result["has_metadata"] = True

    # 2) principle/article number
    patterns = [
        (r"اصل\s+([آ-ی\s]+(?:م|مین)?|\d+)", "اصل"),
        (r"(?:ماده|مواد)\s+(\d+(?:\s*مکرر)?(?:\s*تا\s*\d+)?(?:\s*و\s*\d+)*)", "ماده"),
    ]
    for pattern, article_type in patterns:
        match = re.search(pattern, q)
        if match:
            raw_number = match.group(1).strip()
            result["article_number"] = normalize_article_number(raw_number)
            result["article_type"] = article_type
            result["has_metadata"] = True
            break

    return result
