from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range


@dataclass
class QdrantCollections:
    laws: str = "legal_laws"
    unity: str = "legal_votes_unity"
    dadnameh: str = "legal_votes_dadnameh"


def _extract_main_text(payload: Dict[str, Any], source_type: str) -> str:
    """
    Normalize text field across different collections.
    Your upload saved `page_content` always, but some have `text`/`vote_text`. (kept)
    """
    if source_type == "قانون":
        return payload.get("text") or payload.get("page_content", "") or ""
    # votes
    return payload.get("vote_text") or payload.get("text") or payload.get("page_content", "") or ""


def retrieve_semantic_only(
    qdrant: QdrantClient,
    collections: QdrantCollections,
    query_vector: List[float],
    top_k_total: int = 20,
) -> List[Dict]:
    all_results: List[Dict] = []

    laws_limit = max(8, top_k_total // 2)
    unity_limit = max(4, top_k_total // 4)
    dadnameh_limit = max(4, top_k_total // 4)

    plan: List[Tuple[str, str, int]] = [
        (collections.laws, "قانون", laws_limit),
        (collections.unity, "وحدت رویه", unity_limit),
        (collections.dadnameh, "دادنامه", dadnameh_limit),
    ]

    for collection_name, source_type, limit in plan:
        search_result = qdrant.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        for hit in search_result.points:
            meta = dict(hit.payload or {})
            main_text = _extract_main_text(meta, source_type)
            all_results.append(
                {
                    "text": main_text[:3500],
                    "source_type": source_type,
                    "collection": collection_name,
                    "retrieval_score": float(hit.score),
                    "metadata": meta,
                    "matched_via": "semantic_only",
                }
            )

    all_results.sort(key=lambda x: x["retrieval_score"], reverse=True)
    return all_results[:top_k_total]


def retrieve_with_metadata(
    qdrant: QdrantClient,
    collections: QdrantCollections,
    query_vector: List[float],
    law_name: Optional[str],
    article_num: Optional[str],
    article_type: Optional[str],
    top_k_total: int = 20,
    verbose: bool = False,
) -> List[Dict]:
    all_results: List[Dict] = []
    priority = {"law+article": 0, "law_only": 1, "semantic": 2}

    if verbose:
        print(f"🎯 Metadata: law='{law_name}' | {article_type} {article_num}")

    # 1) law + article (Range)
    if law_name and article_num:
        if verbose:
            print("🔍 فیلتر قانون + شماره (با Range)...")

        if article_type == "اصل":
            field_name = "principle_number"
        elif article_type == "ماده":
            field_name = "article_number"
        else:
            field_name = None

        article_num_float: Optional[float] = None
        try:
            article_num_float = float(article_num)
        except Exception:
            article_num_float = None

        if field_name:
            filters = [
                FieldCondition(key="law_name", match=MatchValue(value=law_name)),
            ]
            if article_num_float is not None:
                filters.append(
                    FieldCondition(
                        key=field_name,
                        range=Range(gte=article_num_float - 0.1, lte=article_num_float + 0.1),
                    )
                )

            hits = qdrant.query_points(
                collection_name=collections.laws,
                query=query_vector,
                query_filter=Filter(must=filters),
                limit=10,
                with_payload=True,
                with_vectors=False,
            ).points

            if verbose:
                print(f"   ✓ {len(hits)} سند (law+{field_name})")

            for hit in hits:
                meta = dict(hit.payload or {})
                main_text = _extract_main_text(meta, "قانون")
                all_results.append(
                    {
                        "text": main_text[:3500],
                        "source_type": "قانون",
                        "collection": collections.laws,
                        "retrieval_score": float(hit.score),
                        "metadata": meta,
                        "matched_via": "law+article",
                    }
                )

    # 2) law only
    if law_name and len(all_results) < top_k_total // 2:
        if verbose:
            print("🔍 فیلتر فقط قانون...")

        hits = qdrant.query_points(
            collection_name=collections.laws,
            query=query_vector,
            query_filter=Filter(
                must=[FieldCondition(key="law_name", match=MatchValue(value=law_name))]
            ),
            limit=10,
            with_payload=True,
            with_vectors=False,
        ).points

        if verbose:
            print(f"   ✓ {len(hits)} سند (law_only)")

        for hit in hits:
            meta = dict(hit.payload or {})
            main_text = _extract_main_text(meta, "قانون")
            all_results.append(
                {
                    "text": main_text[:3500],
                    "source_type": "قانون",
                    "collection": collections.laws,
                    "retrieval_score": float(hit.score),
                    "metadata": meta,
                    "matched_via": "law_only",
                }
            )

    # 3) semantic fallback
    if len(all_results) < top_k_total:
        if verbose:
            print("🔍 Semantic fallback...")

        plan: List[Tuple[str, str, int]] = [
            (collections.laws, "قانون", 8),
            (collections.unity, "وحدت رویه", 6),
            (collections.dadnameh, "دادنامه", 6),
        ]
        for collection_name, source_type, limit in plan:
            hits = qdrant.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            ).points

            for hit in hits:
                meta = dict(hit.payload or {})
                main_text = _extract_main_text(meta, source_type)
                all_results.append(
                    {
                        "text": main_text[:3500],
                        "source_type": source_type,
                        "collection": collection_name,
                        "retrieval_score": float(hit.score),
                        "metadata": meta,
                        "matched_via": "semantic",
                    }
                )

    # de-dup
    seen = set()
    unique_results = []
    for r in all_results:
        h = (r.get("text") or "")[:200]
        if h not in seen:
            seen.add(h)
            unique_results.append(r)

    unique_results.sort(
        key=lambda x: (
            priority.get(x.get("matched_via", ""), 99),
            -x.get("retrieval_score", 0.0),
        )
    )
    final_results = unique_results[:top_k_total]

    if verbose:
        print(f"✅ نهایی: {len(final_results)} نتیجه")

    return final_results
