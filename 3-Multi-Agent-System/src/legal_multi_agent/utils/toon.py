from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


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
    if lines and _CODE_FENCE_RE.match(lines[0]):
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
# Specific parsers
# ----------------------------
def extract_toon_answer(content: str, verbose: bool = False) -> Optional[Dict]:
    """
    Parse:
      results{explanation,answer,confidence}:
      <explanation>,<1-4>,<1-5>

    Returns:
      {
        "explanation": str,
        "answer": str (1-4),
        "confidence": int | None
      }
    """
    schema = ["explanation", "answer", "confidence"]
    parsed = extract_toon_single_row(content, schema=schema, verbose=verbose)
    if not parsed:
        return None

    explanation = parsed.row["explanation"].strip()
    answer_raw = parsed.row["answer"]
    conf_raw = parsed.row["confidence"]

    # normalize answer to last digit 1-4
    m = re.search(r"([1-4])$", answer_raw.strip())
    answer = m.group(1) if m else None

    confidence = _to_int(conf_raw)

    if answer is None:
        if verbose:
            print(f"⚠️ TOON(answer): invalid answer: {answer_raw!r}")
        return None

    return {"explanation": explanation, "answer": answer, "confidence": confidence}


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