# src/legal_multi_agent/tools/retriever_tool.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langsmith import traceable

from legal_multi_agent.rag.pipeline import legal_rag_retrieve, format_results_for_llm
from legal_multi_agent.utils.logger import log_debug, log_info


class RetrieverInput(BaseModel):
    """ورودی ابزار جستجوی حقوقی."""

    query: str = Field(
        description="متن سوال یا کوئری جستجو به زبان فارسی. باید دقیق و کامل باشد."
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=15,
        description="تعداد اسناد برتر برای بازیابی (پیش‌فرض 5، حداکثر 15).",
    )
    use_rerank: bool = Field(
        default=True,
        description="آیا از reranking با Cohere استفاده شود؟ (پیشنهاد True)",
    )
    article_number: Optional[str] = Field(
        default=None,
        description=(
            "شماره ماده برای فیلتر کردن نتایج (مثال: '826' یا '416'). "
            "اگر مشخص شود، فقط نتایج مربوط به این ماده برگردانده می‌شود."
        ),
    )
    law_name: Optional[str] = Field(
        default=None,
        description=(
            "نام قانون برای فیلتر کردن نتایج (مثال: 'قانون مدنی' یا 'قانون تجارت'). "
            "اگر مشخص شود، فقط نتایج از این قانون برگردانده می‌شود."
        ),
    )


def _extract_metadata_summary(results: List[Dict]) -> List[str]:
    """استخراج خلاصه metadata از نتایج برای نمایش به ایجنت."""
    summary = []
    for i, result in enumerate(results[:5], 1):
        meta        = result.get("metadata", {})
        source_type = result.get("source_type", "نامشخص")

        parts = [f"[{i}] {source_type}"]

        law_name = meta.get("law_name")
        if law_name:
            parts.append(f"قانون: {law_name}")

        article = meta.get("article_number")
        if article:
            parts.append(f"ماده {article}")

        principle = meta.get("principle_number")
        if principle:
            parts.append(f"اصل {principle}")

        title = meta.get("title")
        if title:
            title_short = title[:40] + "..." if len(title) > 40 else title
            parts.append(f"عنوان: {title_short}")

        summary.append(" | ".join(parts))

    if len(results) > 5:
        summary.append(f"... و {len(results) - 5} سند دیگر")

    return summary


@tool(args_schema=RetrieverInput)
def retriever_tool(
    query:          str,
    top_k:          int = 5,
    use_rerank:     bool = True,
    article_number: Optional[str] = None,
    law_name:       Optional[str] = None,
) -> str:
    """
    جستجوی متون قانونی و آرای قضایی با RAG.

    این ابزار به ایجنت اجازه می‌دهد:
    - اسناد مرتبط با سوال را از پایگاه قوانین بیابد.
    - نتایج را بر اساس شماره ماده یا نام قانون فیلتر کند.
    - تعداد نتایج و استفاده از rerank را کنترل کند.

    خروجی: متن فرمت‌شده اسناد بازیابی شده (context) با metadata.
    """
    try:
        results = legal_rag_retrieve(
            query=query,
            method="auto",
            top_k=top_k,
            use_rerank=use_rerank,
            verbose=False,
            override_law_name=law_name,
            override_article_number=article_number,
        )

        if not results:
            filters_desc = []
            if article_number:
                filters_desc.append(f"ماده {article_number}")
            if law_name:
                filters_desc.append(f"قانون '{law_name}'")

            if filters_desc:
                return (
                    f"هیچ سندی با فیلتر {' و '.join(filters_desc)} یافت نشد.\n"
                    "پیشنهاد: فیلتر را حذف کنید یا کوئری را تغییر دهید."
                )
            else:
                return "هیچ سند مرتبطی یافت نشد. سوال را بازنویسی کنید یا top_k را افزایش دهید."

        context      = format_results_for_llm(results, include_metadata=True)
        meta_summary = _extract_metadata_summary(results)

        summary_lines = [f"✓ تعداد اسناد بازیابی شده: {len(results)}"]

        if article_number:
            summary_lines.append(f"✓ فیلتر ماده: {article_number}")
        if law_name:
            summary_lines.append(f"✓ فیلتر قانون: {law_name}")

        summary_lines.append(f"✓ Rerank فعال: {use_rerank}")
        summary_lines.append("")
        summary_lines.append("📄 خلاصه اسناد:")
        summary_lines.extend(meta_summary)

        summary = "\n".join(summary_lines)
        return f"{summary}\n\n{'='*60}\nمتن کامل اسناد:\n{'='*60}\n\n{context}"

    except Exception as e:
        return f"خطا در جستجو: {str(e)}"


# ═══════════════════════════════════════════════════════════
# retrieve_documents — برای استفاده مستقیم در nodes
# ═══════════════════════════════════════════════════════════
@traceable(name="retrieve_documents")
def retrieve_documents(
    query:      str,
    top_k:      int  = 5,
    use_rerank: bool = True,
) -> Dict[str, Any]:
    """
    نسخه ساده برای صدا زدن مستقیم در nodes (بدون tool calling).
    خروجی: dict شامل results, context, docs_meta

    در صورت خطا: dict با کلید error برمی‌گرداند (crash نمی‌کند).
    """
    try:
        log_debug(f"  📚 retrieve_documents: query='{query[:60]}...' top_k={top_k}")

        results = legal_rag_retrieve(
            query=query,
            method="auto",
            top_k=top_k,
            use_rerank=use_rerank,
            verbose=False,   # ✅ verbose=False — لاگ خودمان را داریم
        )

        if not results:
            log_info("  ⚠️ retrieve_documents: no results found")
            return {
                "rag_results":      [],
                "context":          "",
                "context_preview":  "",
                "docs_meta":        [],
                "error":            "هیچ سند مرتبطی یافت نشد",
            }

        context = format_results_for_llm(results, include_metadata=True)
        preview = context[:2500]

        docs_meta = []
        for i, r in enumerate(results[:10], start=1):
            m = r.get("metadata", {}) if isinstance(r, dict) else {}
            docs_meta.append({
                "i":                i,
                "law":              m.get("law_name"),
                "article_number":   m.get("article_number"),
                "principle_number": m.get("principle_number"),
                "source_type":      r.get("source_type"),
                "title":            m.get("title"),
            })

        log_info(f"  ✅ retrieve_documents: {len(results)} docs retrieved")

        return {
            "rag_results":     results,
            "context":         context,
            "context_preview": preview,
            "docs_meta":       docs_meta,
        }

    except Exception as e:
        log_debug(f"  ❌ retrieve_documents error: {e}")
        return {
            "rag_results":     [],
            "context":         "",
            "context_preview": "",
            "docs_meta":       [],
            "error":           str(e),
        }