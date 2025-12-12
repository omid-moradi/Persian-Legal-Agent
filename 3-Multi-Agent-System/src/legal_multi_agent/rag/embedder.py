from __future__ import annotations

import os
from typing import Optional

import numpy as np
from openai import OpenAI


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterEmbedder:
    """
    OpenRouter embedder wrapper using OpenAI-compatible SDK.

    Notes:
    - OpenRouter supports the OpenAI SDK with a custom base_url. [web:114]
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        model: str = "baai/bge-m3",
        timeout: Optional[float] = None,
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
        self.model = model
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=timeout)

    def embed_query(self, text: str) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=[text])
        emb = np.array(resp.data[0].embedding, dtype="float32")
        return emb
