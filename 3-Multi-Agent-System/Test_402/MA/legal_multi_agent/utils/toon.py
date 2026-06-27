from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any


# ----------------------------
# Helpers
# ----------------------------
_CODE_FENCE_RE = re.compile(r"^\s*```")
_WS_RE = re.compile(r"\s+")


def _strip_code_fences(text: str) -> str:
    """Remove markdown ``` fences if present."""
    t = (text or "").strip()
    if not t.startswith("```"):
        return t

    lines = t.splitlines()

    # remove first fence
    if lines and _CODE_FENCE_RE.match(lines):
        lines = lines[1:]

    # remove last fence
    if lines and _CODE_FENCE_RE.match(lines[-1]):
        lines = lines[:-1]

    return "\n".join(lines).strip()


def _normalize_lines(text: str) -> List[str]:
    """Split into non-empty lines, trimmed."""
    t = _strip_code_fences(text)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return lines


def _schema_to_regex(schema: List[str]) -> re.Pattern:
    """
    Build a robust regex for header line:
      results{a,b,c}:
    (case-insensitive, spaces tolerant)
    """
    # escape each field name but keep it simple (fields are ascii identifiers)
    fields_pat = r"\s*,\s*".join([re.escape(f) for f in schema])
    pat = rf"^results\s*\{{\s*{fields_pat}\s*\}}\s*:\s*$"
    return re.compile(pat, re.IGNORECASE)


def _split_csv_n(text: str, n: int) -> Optional[List[str]]:
    """
    Split a 'CSV-like' line into exactly n fields.
    - We only split on first (n-1) commas: split(",", n-1)
    - This matches your approach: explanation may contain commas.
    """
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


# ----------------------------
# Generic single-row TOON parser
# ----------------------------
@dataclass
class ToonParseResult:
    schema: List[str]
    row: Dict[str, str]
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


# ----------------------------
# Multi-row TOON parser (for option verifier)
# ----------------------------
@dataclass
class ToonMultiRowResult:
    schema: List[str]
    rows: List[Dict[str, str]]
    header_line_index: int


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
      ...

    Stops reading rows when:
    - Next line starts with "results{" (new TOON table)
    - Reached max_rows
    - Line doesn't parse correctly

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

        # Stop if next TOON header encountered
        if line.lower().startswith("results{"):
            break

        parts = _split_csv_n(line, n=len(schema))
        if parts is None:
            # Invalid row, stop parsing
            break

        row = {schema[i]: parts[i].strip().strip('"').strip("'") for i in range(len(schema))}
        rows.append(row)

    if not rows:
        if verbose:
            print("⚠️ TOON: no valid data rows found")
        return None

    return ToonMultiRowResult(schema=schema, rows=rows, header_line_index=header_idx)


# ----------------------------
# Specific parsers
# ----------------------------
def extract_toon_answer(content: str, verbose: bool = False) -> Optional[Dict]:
    """
    Parse:
    results{explanation,answer}:
    <توضیح>,<1-4>

    با fallback برای فرمت نادرست.

    Returns:
    {
        "explanation": str,
        "answer": str (1-4),
    }
    """
    # ✅ schema بدون confidence
    schema = ["explanation", "answer"]
    parsed = extract_toon_single_row(content, schema=schema, verbose=verbose)

    if parsed:
        explanation = parsed.row["explanation"].strip()
        answer_raw  = parsed.row["answer"]

        m = re.search(r"([1-4])$", answer_raw.strip())
        answer = m.group(1) if m else None

        if answer is None:
            if verbose:
                print(f"⚠️ TOON(answer): invalid answer: {answer_raw!r}")
            return None

        return {"explanation": explanation, "answer": answer}

    # ── Fallback ──────────────────────────────────────────
    if verbose:
        print("⚠️ TOON(answer): standard format failed, trying fallback...")

    fallback_pattern = r'results\s*\{([^}]+)\}'
    match = re.search(fallback_pattern, content, re.IGNORECASE | re.DOTALL)

    if not match:
        if verbose:
            print("⚠️ TOON(answer): fallback also failed")
        return None

    inner_content = match.group(1).strip()

    # ✅ تلاش برای split به 2 قسمت (بدون confidence)
    parts = _split_csv_n(inner_content, n=2)

    if not parts:
        # fallback سازگار با نسخه قدیمی (3 قسمت) — اگر LLM هنوز confidence می‌فرستد
        parts3 = _split_csv_n(inner_content, n=3)
        if parts3:
            parts = [parts3[0], parts3[1]]  # explanation و answer را نگه‌دار
        else:
            if verbose:
                print(f"⚠️ TOON(answer): fallback couldn't split: {inner_content[:100]}")
            return None

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

def extract_toon_critic(content: str, verbose: bool = False) -> Optional[Dict]:
    """
    Parse:
      results{needs_revision,issue,action}:
      <true/false>,<issue>,<action>

    Returns:
      {
        "needs_revision": bool,
        "issue": str,
        "action": str
      }
    """
    schema = ["needs_revision", "issue", "action"]
    parsed = extract_toon_single_row(content, schema=schema, verbose=verbose)
    if not parsed:
        return None

    needs_raw = parsed.row["needs_revision"]
    needs_revision = _to_bool(needs_raw)

    if needs_revision is None:
        if verbose:
            print(f"⚠️ TOON(critic): invalid needs_revision: {needs_raw!r}")
        return None

    return {
        "needs_revision": needs_revision,
        "issue": parsed.row["issue"].strip(),
        "action": parsed.row["action"].strip(),
    }


def extract_toon_verifier(content: str, verbose: bool = False) -> Optional[Dict[str, Any]]:
    """
    Parse option verifier output with two TOON tables:

    1) Multi-row scores:
    results{option,support_level,reasoning}:
    1,SUPPORTED,دلیل کامل فارسی
    2,NOT_SUPPORTED,دلیل کامل فارسی
    3,UNCLEAR,دلیل کامل فارسی
    4,NOT_SUPPORTED,دلیل کامل فارسی

    2) Single-row recommendation (✅ بدون confidence):
    results{recommended_answer}:
    2

    Returns:
    {
        "scores": [
            {"option_number": 1, "support_level": "SUPPORTED", "reasoning": "..."},
            ...
        ],
        "recommended_answer": int | None,
    }
    """
    result: Dict[str, Any] = {"scores": [], "recommended_answer": None}

    # ── TOON اول: multi-row scores ────────────────────────
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

    # ── TOON دوم: recommended_answer بدون confidence ─────
    # ✅ ابتدا فرمت جدید (بدون confidence) را امتحان کن
    schema_rec_new = ["recommended_answer"]
    parsed_rec = extract_toon_single_row(content, schema=schema_rec_new, verbose=False)

    if parsed_rec:
        result["recommended_answer"] = _to_int(parsed_rec.row["recommended_answer"])
    else:
        # ✅ fallback سازگار: اگر LLM هنوز فرمت قدیمی (با confidence) فرستاد
        schema_rec_old = ["recommended_answer", "confidence"]
        parsed_rec_old = extract_toon_single_row(content, schema=schema_rec_old, verbose=verbose)
        if parsed_rec_old:
            result["recommended_answer"] = _to_int(parsed_rec_old.row["recommended_answer"])
            if verbose:
                print("⚠️ TOON(verifier): used legacy schema with confidence (fallback)")

    # ── اعتبارسنجی ────────────────────────────────────────
    if not result["scores"]:
        if verbose:
            print("⚠️ TOON(verifier): no valid scores found")
        return None

    return result