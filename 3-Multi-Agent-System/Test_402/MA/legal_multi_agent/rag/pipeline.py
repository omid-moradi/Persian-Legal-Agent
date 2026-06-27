from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from qdrant_client import QdrantClient

from .embedder import OpenRouterEmbedder
from .metadata import load_qavanin_list, parse_question_metadata_simple
from .qdrant_store import QdrantCollections, retrieve_semantic_only, retrieve_with_metadata
from .rerank import CohereReranker
from legal_multi_agent.utils.logger import log_debug, log_info


# ═══════════════════════════════════════════════════════════
# Qavanin path helpers
# ═══════════════════════════════════════════════════════════

def _resolve_qavanin_path(qavanin_path: Optional[str]) -> Path:
    """
    Robustly resolve qavanin_karbordi.txt across:
    - running from project root
    - running from notebooks/ (different CWD)
    - running as package

    Priority:
    1) explicit function arg qavanin_path
    2) env var QAVANIN_KARBORDI_PATH
    3) infer project root from this file location and use data/config/...
    4) fallback to CWD/data/config/...
    """
    # 1) explicit arg
    if qavanin_path:
        p = Path(qavanin_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"qavanin_path not found: {p}")

    # 2) env var
    env_path = os.environ.get("QAVANIN_KARBORDI_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"QAVANIN_KARBORDI_PATH not found: {p}")

    # 3) infer project root based on this file path:
    # .../3-Multi-Agent-System/src/legal_multi_agent/rag/pipeline.py
    this_file = Path(__file__).resolve()

    # parents[0]=rag, [1]=legal_multi_agent, [2]=src, [3]=project_root
    candidate_roots = []
    if len(this_file.parents) >= 4:
        candidate_roots.append(this_file.parents[3])
    if len(this_file.parents) >= 5:
        candidate_roots.append(this_file.parents[4])

    for root in candidate_roots:
        candidate = root / "data" / "config" / "qavanin_karbordi.txt"
        if candidate.exists():
            return candidate

    # 4) fallback: relative to current working directory (notebook case)
    cwd_candidate = Path.cwd() / "data" / "config" / "qavanin_karbordi.txt"
    if cwd_candidate.exists():
        return cwd_candidate

    raise FileNotFoundError(
        "qavanin_karbordi.txt not found. "
        "Provide qavanin_path argument or set QAVANIN_KARBORDI_PATH env var."
    )


def _load_qavanin_list(qavanin_path: Optional[str]) -> List[str]:
    path = _resolve_qavanin_path(qavanin_path)
    text = path.read_text(encoding="utf-8")
    return load_qavanin_list(text)


# ═══════════════════════════════════════════════════════════
# Formatting for LLM
# ═══════════════════════════════════════════════════════════

def format_results_for_llm(results: List[Dict], include_metadata: bool = True) -> str:
    formatted   = []
    fields_order = [
        "article_number", "principle_number",
        "volume", "book", "bab", "chapter", "section",
        "paragraph", "text_section", "title",
        "issuer", "text", "vote_text",
    ]

    for i, result in enumerate(results, 1):
        source_type = result.get("source_type", "نامشخص")
        meta        = result.get("metadata", {}) if include_metadata else {}
        law_name    = meta.get("law_name")

        header_lines = [f"[منبع {i}] ({source_type})"]
        if law_name:
            header_lines[0] += f" - {law_name}"

        principle = meta.get("principle_number")
        article   = meta.get("article_number")
        if principle is not None:
            header_lines.append(f"شماره_اصل: {principle}")
        if article is not None:
            header_lines.append(f"شماره_ماده: {article}")

        body_lines = []
        if include_metadata:
            for field in fields_order:
                if field in meta:
                    value = meta[field]
                    if field in ("text", "vote_text") and isinstance(value, str):
                        snippet = value.strip()
                        if len(snippet) > 1500:
                            snippet = snippet[:1500] + " ..."
                        body_lines.append(f"{field}: {snippet}")
                    else:
                        body_lines.append(f"{field}: {value}")

        # اگر text در meta نبود، از result["text"] استفاده کن
        if not include_metadata or "text" not in meta:
            main_text = result.get("text", "")
            if main_text:
                snippet = main_text.strip()
                if len(snippet) > 1500:
                    snippet = snippet[:1500] + " ..."
                body_lines.append(f"content: {snippet}")

        block = "\n".join(header_lines)
        if body_lines:
            block += "\n" + "\n".join(body_lines)

        formatted.append(block)

    return "\n---\n\n".join(formatted)


# ═══════════════════════════════════════════════════════════
# Main API
# ═══════════════════════════════════════════════════════════

def legal_rag_retrieve(
    query:                   str,
    method:                  str            = "auto",
    top_k:                   int            = 10,
    use_rerank:              bool           = True,
    verbose:                 bool           = False,
    qdrant_url:              Optional[str]  = None,
    collections:             Optional[QdrantCollections] = None,
    qavanin_path:            Optional[str]  = None,
    embed_model:             str            = "baai/bge-m3",
    # پارامترهای override برای فیلتر مستقیم metadata
    override_law_name:       Optional[str]  = None,
    override_article_number: Optional[str]  = None,
    override_article_type:   Optional[str]  = None,
) -> List[Dict]:
    """
    Main API used by LangGraph nodes.
    - method: auto|metadata|semantic
    - override_law_name:       نام قانون برای فیلتر مستقیم (اختیاری)
    - override_article_number: شماره ماده/اصل برای فیلتر مستقیم (اختیاری)
    - override_article_type:   نوع article ("ماده" یا "اصل") - اختیاری

    اگر override_* مشخص شود، به جای parsing از query استفاده می‌شود.
    """
    if not query or not query.strip():
        return []

    qdrant_url  = qdrant_url  or os.environ.get("QDRANT_URL", "http://localhost:6333")
    collections = collections or QdrantCollections()

    # بارگذاری لیست قوانین
    qavanin_list = _load_qavanin_list(qavanin_path)

    # clients
    qdrant   = QdrantClient(url=qdrant_url)
    embedder = OpenRouterEmbedder(model=embed_model)

    # ✅ اگر COHERE_API_KEY نبود، graceful fallback به جای crash
    cohere_key = os.environ.get("COHERE_API_KEY")
    reranker   = None
    if use_rerank:
        if not cohere_key:
            log_info("⚠️ pipeline: COHERE_API_KEY نیست — rerank غیرفعال شد")
            use_rerank = False
        else:
            reranker = CohereReranker(api_key=cohere_key)

    log_debug(f"📝 Query: {query[:80]}...")

    # ── ۱) parse metadata از query ────────────────────────
    parsed = parse_question_metadata_simple(query, qavanin_list)

    # استفاده از override اگر مشخص شده، وگرنه از parsed
    law_name     = override_law_name       if override_law_name       is not None else parsed.get("law_name")
    article_num  = override_article_number if override_article_number is not None else parsed.get("article_number")
    article_type = override_article_type   if override_article_type   is not None else parsed.get("article_type")

    # اگر article_number داریم اما article_type نداریم، پیش‌فرض "ماده"
    if article_num and not article_type:
        article_type = "ماده"

    # ── ۲) انتخاب روش بازیابی ─────────────────────────────
    if method == "auto":
        if article_num or law_name:
            method = "metadata"
            log_debug("🎯 روش انتخاب شده: Metadata-aware")
        else:
            method = "semantic"
            log_debug("🔍 روش انتخاب شده: Semantic search")

    # ── ۳) embed ───────────────────────────────────────────
    query_vec = embedder.embed_query(query).tolist()

    # ── ۴) retrieve (overfetch برای rerank بهتر) ──────────
    # ✅ حداقل 20 سند برای rerank — max قبلی اضافه بود
    top_k_total = max(top_k * 4, 20)

    if method == "metadata":
        results = retrieve_with_metadata(
            qdrant=qdrant,
            collections=collections,
            query_vector=query_vec,
            law_name=law_name,
            article_num=article_num,
            article_type=article_type,
            top_k_total=top_k_total,
            verbose=verbose,
        )
    else:
        results = retrieve_semantic_only(
            qdrant=qdrant,
            collections=collections,
            query_vector=query_vec,
            top_k_total=top_k_total,
        )
        log_debug(f"🔍 Semantic only: {len(results)} نتیجه")

    log_info(f"  ✅ Retrieved: {len(results)} documents (method={method})")

    # ── ۵) rerank ──────────────────────────────────────────
    if use_rerank and reranker is not None and results:
        results = reranker.rerank_with_cohere_smart(
            query=query,
            results=results,
            top_k=top_k,
            verbose=verbose,
        )
        log_info(f"  ✅ Reranked: top {len(results)} documents")

    return results[:top_k]