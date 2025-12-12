from __future__ import annotations

from typing import Dict, List, Optional

import cohere


class CohereReranker:
    def __init__(self, api_key: Optional[str] = None, model: str = "rerank-multilingual-v3.0") -> None:
        self.api_key = api_key
        self.model = model
        if not self.api_key:
            raise ValueError("COHERE_API_KEY is not set.")
        self.client = cohere.Client(api_key=self.api_key)

    def rerank_with_cohere_smart(self, query: str, results: List[Dict], top_k: int = 5, verbose: bool = False) -> List[Dict]:
        if not results:
            return []

        law_article = [r for r in results if r.get("matched_via") == "law+article"]
        others = [r for r in results if r.get("matched_via") != "law+article"]

        # no law+article -> rerank all
        if not law_article:
            documents = [(r.get("text") or "")[:4000] for r in results]
            if verbose:
                print(f"🔄 Reranking {len(documents)} سند (بدون law+article)...")

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
            except Exception:
                # fallback
                out = []
                for r in results[:top_k]:
                    rr = r.copy()
                    rr["rerank_score"] = float(rr.get("retrieval_score", 0.0))
                    out.append(rr)
                return out

        # keep law+article at top
        if verbose:
            print(f"🎯 {len(law_article)} نتیجه law+article یافت شد")

        for dm in law_article:
            dm["rerank_score"] = 1.0

        available_slots = max(top_k - len(law_article), 0)

        reranked_others = []
        if others and available_slots > 0:
            documents = [(r.get("text") or "")[:4000] for r in others]
            if verbose:
                print(f"🔄 Reranking {len(documents)} سند دیگر...")

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
            except Exception:
                reranked_others = sorted(
                    others, key=lambda x: x.get("retrieval_score", 0.0), reverse=True
                )[:available_slots]

        final_results = law_article + reranked_others

        if len(final_results) < top_k:
            remaining = [r for r in others if r not in reranked_others]
            remaining_sorted = sorted(remaining, key=lambda x: x.get("retrieval_score", 0.0), reverse=True)
            for r in remaining_sorted[: top_k - len(final_results)]:
                rr = r.copy()
                rr["rerank_score"] = float(rr.get("retrieval_score", 0.0))
                final_results.append(rr)

        return final_results[:top_k]
