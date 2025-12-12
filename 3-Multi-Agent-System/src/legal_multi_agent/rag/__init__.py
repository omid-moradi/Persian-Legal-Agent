"""
RAG package for Iranian legal QA project.

Entry points:
- legal_rag_retrieve
- format_results_for_llm
"""

from .pipeline import legal_rag_retrieve, format_results_for_llm  # noqa: F401
