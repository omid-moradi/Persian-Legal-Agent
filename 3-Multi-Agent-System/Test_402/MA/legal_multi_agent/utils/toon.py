from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
_CODE_FENCE_RE = re.compile(r"^\s*```")
_WS_RE = re.compile(r"\s+")


def _strip_code_fences(text: str) -> str:
    """Remove markdown ``` fences if present."""
    t = (text or "").strip()
    if not t.startswith("```"):
        return t
    lines = t.splitlines()
    if lines and _CODE_FENCE_RE.match(lines):
        lines = lines[1:]
    if lines and _CODE_FENCE_RE.match(lines[-1]):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _normalize_lines(text: str) -> List[str]:
    """Split into non-empty lines, trimmed."""
    t = _strip_code_fences(text)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return lines


def _schema_to_regex(schema: List[str]) -> re.Pattern:
    fields_pat = r"\s*,\s*".join([re.escape(f) for f in schema])
    pat = rf"^results\s*\{{\s*{fields_pat}\s*\}}\s*:.*$"
    return re.compile(pat, re.IGNORECASE)


def _split_csv_n(text: str, n: int) -> Optional[List[str]]:
    if text is None:
        return None
    parts = [p.strip() for p in text.strip().split(",", n - 1)]
    if len(parts) != n:
        return None
    return parts


def _to_bool(x: str) -> Optional[bool]:
    if x is None:
        return None
    v = x.strip().strip('"').strip("'").lower()
    if v in ("true", "1", "yes", "y"):
        return True
    if v in ("false", "0", "no", "n"):
        return False
    return None


def _to_int(x: str) -> Optional[int]:
    if x is None:
        return None
    v = x.strip().strip('"').strip("'")
    try:
        return int(float(v))
    except Exception:
        return None


# ─────────────────────────────────────────────
# Generic parsers
# ─────────────────────────────────────────────
@dataclass
class ToonParseResult:
    schema: List[str]
    row: Dict[str, str]
    header_line_index: int


@dataclass
class ToonMultiRowResult:
    schema: List[str]
    rows: List[Dict[str, str]]
    header_line_index: int


def extract_toon_single_row(
    content: str,
    schema: List[str],
    verbose: bool = False,
) -> Optional[ToonParseResult]:
    """
    Extract a single-row TOON table with a specific schema.

    Expected pattern (surrounding text ignored):
      results{field1,field2,...}:
      v1,v2,...

    Returns:
      ToonParseResult or None
    """
    lines = _normalize_lines(content)
    if not lines:
        if verbose:
            print("⚠️ TOON: empty content")
        return None

    header_re = _schema_to_regex(schema)
    header_idx = None
    for i, ln in enumerate(lines):
        if header_re.match(ln):
            header_idx = i
            break

    if header_idx is None:
        if verbose:
            print(f"⚠️ TOON: header not found for schema={schema}")
        return None

    if header_idx + 1 >= len(lines):
        if verbose:
            print("⚠️ TOON: no data row after header")
        return None

    data_line = lines[header_idx + 1]
    parts = _split_csv_n(data_line, n=len(schema))
    if parts is None:
        if verbose:
            print(f"⚠️ TOON: could not split data row into {len(schema)} fields: {data_line!r}")
        return None

    row = {schema[i]: parts[i].strip().strip('"').strip("'") for i in range(len(schema))}
    return ToonParseResult(schema=schema, row=row, header_line_index=header_idx)


def extract_toon_multi_row(
    content: str,
    schema: List[str],
    max_rows: int = 10,
    verbose: bool = False,
) -> Optional[ToonMultiRowResult]:
    """
    Extract a multi-row TOON table with a specific schema.

    Expected pattern:
      results{field1,field2,...}:
      v1,v2,...
      v1,v2,...

    Returns:
      ToonMultiRowResult or None
    """
    lines = _normalize_lines(content)
    if not lines:
        if verbose:
            print("⚠️ TOON: empty content")
        return None

    header_re = _schema_to_regex(schema)
    header_idx = None
    for i, ln in enumerate(lines):
        if header_re.match(ln):
            header_idx = i
            break

    if header_idx is None:
        if verbose:
            print(f"⚠️ TOON: header not found for schema={schema}")
        return None

    rows = []
    for row_offset in range(1, max_rows + 1):
        idx = header_idx + row_offset
        if idx >= len(lines):
            break
        line = lines[idx]
        if line.lower().startswith("results{"):
            break
        parts = _split_csv_n(line, n=len(schema))
        if parts is None:
            break
        row = {schema[i]: parts[i].strip().strip('"').strip("'") for i in range(len(schema))}
        rows.append(row)

    if not rows:
        if verbose:
            print("⚠️ TOON: no valid data rows found")
        return None

    return ToonMultiRowResult(schema=schema, rows=rows, header_line_index=header_idx)


# ─────────────────────────────────────────────
# Specific parsers
# ─────────────────────────────────────────────

def extract_toon_answer(content: str, verbose: bool = False) -> Optional[Dict]:
    """
    Parse:
      results{explanation,answer}:
      <توضیح>,<1-4>

    Returns:
      {"explanation": str, "answer": str}
    """
    schema = ["explanation", "answer"]
    parsed = extract_toon_single_row(content, schema=schema, verbose=verbose)

    if parsed:
        explanation = parsed.row["explanation"].strip()
        answer_raw  = parsed.row["answer"]
        m = re.search(r"([1-4])\s*$", answer_raw.strip(), re.MULTILINE)
        answer = m.group(1) if m else None
        if answer is None:
            if verbose:
                print(f"⚠️ TOON(answer): invalid answer: {answer_raw!r}")
            return None
        return {"explanation": explanation, "answer": answer}

    if verbose:
        print("⚠️ TOON(answer): standard format failed, trying fallback...")

    fallback_pattern = r'results\s*\{([^}]+)\}\s*:\s*\n?\s*(.+)'
    match = re.search(fallback_pattern, content, re.IGNORECASE | re.DOTALL)
    if not match:
        if verbose:
            print("⚠️ TOON(answer): fallback also failed")
        return None

    # ✅ اول خط اول را بگیر، بعد split کن
    data_line = match.group(2).strip().splitlines()[0].strip()

    parts = _split_csv_n(data_line, n=2)
    if not parts:
        parts3 = _split_csv_n(data_line, n=3)
        if parts3:
            parts = [parts3[0], parts3[1]]
        else:
            if verbose:
                print(f"⚠️ TOON(answer): fallback couldn't split: {data_line[:100]}")
            return None

    # ✅ parts یک list است — با ایندکس دسترسی پیدا کن
    explanation = parts[0].strip()
    answer_raw  = parts[1].strip()

    m = re.search(r'([1-4])', answer_raw)
    answer = m.group(1) if m else None
    if answer is None:
        if verbose:
            print(f"⚠️ TOON(answer): fallback invalid answer: {answer_raw!r}")
        return None
    if verbose:
        print(f"✅ TOON(answer): fallback successful - answer={answer}")
    return {"explanation": explanation, "answer": answer}


# ── regex ثابت برای critic: هم one-liner و هم two-liner ─────────────────────
_CRITIC_ONELINER_RE = re.compile(
    r"results\s*\{\s*needs_revision\s*,\s*issue\s*,\s*action\s*\}\s*:\s*"
    r"(true|false)\s*,\s*(.*?)\s*,\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_issue_from_verdict(text: str) -> str:
    """جمله خلاصه مشکل را از گام ۴ critic استخراج می‌کند."""
    m = re.search(
        r"ERROR\s*FOUND\s*[:\-–]?\s*(.{10,300}?)(?:\n|$)",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()[:300]
    return "خطای صریح شناسایی شد — جزئیات در متن critic موجود است"


def extract_toon_critic(content: str, verbose: bool = False) -> Optional[Dict]:
    """
    Parse critic TOON.

    سه روش به‌ترتیب اولویت:

    ① One-liner (فرمت اصلی خروجی critic):
       results{needs_revision,issue,action}:false,خطا,اصلاح

    ② Two-liner (فرمت سنتی):
       results{needs_revision,issue,action}:
       false,خطا,اصلاح

    ③ Keyword fallback از گام ۴:
       اگر TOON نوشته نشد ولی «ERROR FOUND» یا «NO ERROR» در متن بود.

    Returns:
        {"needs_revision": bool, "issue": str, "action": str}
        یا None اگر هیچ‌کدام موفق نشد.
    """
    if not content:
        return None

    # ── روش ۱: one-liner regex — مستقیم‌ترین و مطمئن‌ترین ──────────────
    m = _CRITIC_ONELINER_RE.search(content)
    if m:
        needs_revision = _to_bool(m.group(1))
        if needs_revision is None:
            if verbose:
                print(f"⚠️ TOON(critic): invalid bool: {m.group(1)!r}")
            return None
        issue  = m.group(2).strip() or "خطای صریحی یافت نشد"
        action = m.group(3).strip() or "پاسخ قابل قبول است و نیازی به تغییر ندارد"
        if verbose:
            print(f"✅ TOON(critic) [روش ۱ — one-liner]: needs_revision={needs_revision}")
        return {"needs_revision": needs_revision, "issue": issue, "action": action}

    # ── روش ۲: two-liner از طریق extract_toon_single_row ─────────────────
    schema = ["needs_revision", "issue", "action"]
    parsed = extract_toon_single_row(content, schema=schema, verbose=verbose)
    if parsed:
        needs_revision = _to_bool(parsed.row["needs_revision"])
        if needs_revision is None:
            if verbose:
                print(f"⚠️ TOON(critic): invalid bool: {parsed.row['needs_revision']!r}")
            return None
        issue  = parsed.row["issue"].strip()  or "خطای صریحی یافت نشد"
        action = parsed.row["action"].strip() or "پاسخ قابل قبول است و نیازی به تغییر ندارد"
        if verbose:
            print(f"✅ TOON(critic) [روش ۲ — two-liner]: needs_revision={needs_revision}")
        return {"needs_revision": needs_revision, "issue": issue, "action": action}

    # ── روش ۳: keyword fallback از گام ۴ ────────────────────────────────
    error_found = bool(re.search(r"\bERROR\s*FOUND\b", content, re.IGNORECASE))
    no_error    = bool(re.search(r"\bNO\s*ERROR\b",    content, re.IGNORECASE))

    if error_found and not no_error:
        if verbose:
            print("⚠️ TOON(critic) [روش ۳ — keyword]: ERROR FOUND بدون TOON")
        return {
            "needs_revision": True,
            "issue":  _extract_issue_from_verdict(content),
            "action": "بازبینی پاسخ reasoner لازم است — خطا در گام ۴ شناسایی شد",
        }

    if no_error:
        if verbose:
            print("⚠️ TOON(critic) [روش ۳ — keyword]: NO ERROR بدون TOON")
        return {
            "needs_revision": False,
            "issue":  "خطای صریحی یافت نشد",
            "action": "پاسخ قابل قبول است و نیازی به تغییر ندارد",
        }

    if verbose:
        print("❌ TOON(critic): هیچ روشی موفق نشد")
    return None


def extract_toon_verifier(content: str, verbose: bool = False) -> Optional[Dict[str, Any]]:
    """
    Parse option verifier output with two TOON tables:

    1) Multi-row scores:
    results{option,support_level,reasoning}:
    1,SUPPORTED,دلیل فارسی
    2,NOT_SUPPORTED,دلیل فارسی
    3,UNCLEAR,دلیل فارسی
    4,NOT_SUPPORTED,دلیل فارسی

    2) Single-row recommendation:
    results{recommended_answer}:
    2

    Returns:
    {
        "scores": [
            {"option_number": int, "support_level": str, "reasoning": str},
            ...
        ],
        "recommended_answer": int | None,
    }
    """
    result: Dict[str, Any] = {"scores": [], "recommended_answer": None}

    # ── TOON اول: multi-row scores ────────────────────────────────────────
    schema_scores = ["option", "support_level", "reasoning"]
    parsed_scores = extract_toon_multi_row(
        content, schema=schema_scores, max_rows=4, verbose=verbose
    )
    if parsed_scores:
        for row in parsed_scores.rows:
            opt_num = _to_int(row["option"])
            if opt_num is None:
                continue
            result["scores"].append({
                "option_number": opt_num,
                "support_level": row["support_level"].strip(),
                "reasoning":     row["reasoning"].strip(),
            })

    # ── TOON دوم: recommended_answer ─────────────────────────────────────
    schema_rec_new = ["recommended_answer"]
    parsed_rec = extract_toon_single_row(content, schema=schema_rec_new, verbose=False)
    if parsed_rec:
        result["recommended_answer"] = _to_int(parsed_rec.row["recommended_answer"])
    else:
        # fallback سازگار با فرمت قدیمی (با confidence)
        schema_rec_old = ["recommended_answer", "confidence"]
        parsed_rec_old = extract_toon_single_row(content, schema=schema_rec_old, verbose=verbose)
        if parsed_rec_old:
            result["recommended_answer"] = _to_int(parsed_rec_old.row["recommended_answer"])
            if verbose:
                print("⚠️ TOON(verifier): used legacy schema with confidence (fallback)")

    # ── اعتبارسنجی ────────────────────────────────────────────────────────
    if not result["scores"]:
        if verbose:
            print("⚠️ TOON(verifier): no valid scores found")
        return None

    return result