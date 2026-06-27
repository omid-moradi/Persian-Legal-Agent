from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from openai import OpenAI

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterEmbedder:
    """
    OpenRouter embedder wrapper using OpenAI-compatible SDK.

    Notes:
    - OpenRouter supports the OpenAI SDK with a custom base_url.
    - از مدل baai/bge-m3 برای متون فارسی استفاده می‌شود.
    """

    def __init__(
        self,
        api_key:  Optional[str]   = None,
        base_url: str              = DEFAULT_OPENROUTER_BASE_URL,
        model:    str              = "baai/bge-m3",
        timeout:  Optional[float]  = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
            "OPENROUTER_EMBEDDINGS_API_KEY"
        )
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not set. "
                "Set OPENROUTER_API_KEY (recommended) or OPENROUTER_EMBEDDINGS_API_KEY."
            )

        self.base_url = base_url
        self.model    = model
        self.client   = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=timeout)

    def embed_query(self, text: str) -> np.ndarray:
        """
        یک متن را embed می‌کند.

        Returns:
            np.ndarray با dtype float32

        Raises:
            RuntimeError: اگر API خطا بدهد یا پاسخ خالی باشد
        """
        if not text or not text.strip():
            raise ValueError("embed_query: متن ورودی نمی‌تواند خالی باشد.")

        try:
            resp = self.client.embeddings.create(model=self.model, input=[text])
            if not resp.data:
                raise RuntimeError("embed_query: پاسخ API خالی بود.")
            return np.array(resp.data[0].embedding, dtype="float32")
        except Exception as e:
            raise RuntimeError(f"embed_query failed: {e}") from e

    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """
        ✅ batch embed — چند متن را با یک API call embed می‌کند.

        Args:
            texts: لیست متون برای embed

        Returns:
            لیستی از np.ndarray با dtype float32 (به ترتیب ورودی)

        Raises:
            ValueError: اگر لیست خالی باشد
            RuntimeError: اگر API خطا بدهد
        """
        if not texts:
            raise ValueError("embed_documents: لیست متون نمی‌تواند خالی باشد.")

        # فیلتر کردن متون خالی با نگه‌داشتن index اصلی
        clean_texts = [t.strip() if t else "" for t in texts]

        try:
            resp = self.client.embeddings.create(model=self.model, input=clean_texts)
            if not resp.data:
                raise RuntimeError("embed_documents: پاسخ API خالی بود.")
            # ترتیب پاسخ را بر اساس index اصلی مرتب می‌کنیم
            sorted_data = sorted(resp.data, key=lambda x: x.index)
            return [np.array(item.embedding, dtype="float32") for item in sorted_data]
        except Exception as e:
            raise RuntimeError(f"embed_documents failed: {e}") from e