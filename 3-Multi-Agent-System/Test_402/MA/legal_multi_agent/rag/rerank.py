from __future__ import annotations

from typing import Dict, List, Optional

import cohere

from legal_multi_agent.utils.logger import log_debug, log_info


class CohereReranker:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model:   str            = "rerank-v3.5",
    ) -> None:
        self.api_key = api_key
        self.model   = model
        if not self.api_key:
            raise ValueError("COHERE_API_KEY is not set.")
        self.client = cohere.Client(api_key=self.api_key)

    def rerank_with_cohere_smart(
        self,
        query:   str,
        results: List[Dict],
        top_k:   int  = 5,
        verbose: bool = False,
    ) -> List[Dict]:
        if not results:
            return []

        # ✅ law+article AND law_only_supplement هر دو در اولویت بالا
        HIGH_PRIORITY = {"law+article", "law_only_supplement"}
        law_article = [r for r in results if r.get("matched_via") in HIGH_PRIORITY]
        others      = [r for r in results if r.get("matched_via") not in HIGH_PRIORITY]

        # ── مسیر ۱: بدون نتیجه high-priority — rerank همه ─────────
        if not law_article:
            documents = [(r.get("text") or "")[:4000] for r in results]
            log_debug(f"🔄 Reranking {len(documents)} سند (بدون law+article)...")

            try:
                response = self.client.rerank(
                    query=query,
                    documents=documents,
                    model=self.model,
                    top_n=min(top_k, len(documents)),
                )
                reranked = []
                for item in response.results:
                    r = results[item.index].copy()
                    r["rerank_score"] = float(item.relevance_score)
                    reranked.append(r)
                return reranked

            except Exception as e:
                # ✅ لاگ خطا به جای نادیده گرفتن
                log_info(f"⚠️ rerank failed (no law+article): {e} — fallback to retrieval_score")
                out = []
                for r in results[:top_k]:
                    rr = r.copy()
                    rr["rerank_score"] = float(rr.get("retrieval_score", 0.0))
                    out.append(rr)
                return out

        # ── مسیر ۲: با نتیجه high-priority — آن‌ها را بالا نگه دار ─
        log_debug(f"🎯 {len(law_article)} نتیجه high-priority یافت شد")

        for dm in law_article:
            dm["rerank_score"] = 1.0

        available_slots = max(top_k - len(law_article), 0)

        reranked_others: List[Dict] = []
        if others and available_slots > 0:
            documents = [(r.get("text") or "")[:4000] for r in others]
            log_debug(f"🔄 Reranking {len(documents)} سند دیگر...")

            try:
                response = self.client.rerank(
                    query=query,
                    documents=documents,
                    model=self.model,
                    top_n=min(available_slots, len(documents)),
                )
                for item in response.results:
                    r = others[item.index].copy()
                    r["rerank_score"] = float(item.relevance_score)
                    reranked_others.append(r)

            except Exception as e:
                # ✅ لاگ خطا به جای نادیده گرفتن
                log_info(f"⚠️ rerank failed (others): {e} — fallback to retrieval_score")
                reranked_others = sorted(
                    others,
                    key=lambda x: x.get("retrieval_score", 0.0),
                    reverse=True,
                )[:available_slots]

        final_results = law_article + reranked_others

        # تکمیل اگر هنوز به top_k نرسیدیم
        if len(final_results) < top_k:
            seen        = {id(r) for r in reranked_others}
            remaining   = [r for r in others if id(r) not in seen]
            rem_sorted  = sorted(
                remaining,
                key=lambda x: x.get("retrieval_score", 0.0),
                reverse=True,
            )
            for r in rem_sorted[: top_k - len(final_results)]:
                rr = r.copy()
                rr["rerank_score"] = float(rr.get("retrieval_score", 0.0))
                final_results.append(rr)

        log_debug(f"✅ rerank نهایی: {len(final_results[:top_k])} نتیجه")
        return final_results[:top_k]