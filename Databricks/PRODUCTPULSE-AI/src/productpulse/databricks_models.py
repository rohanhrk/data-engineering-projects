"""Databricks Foundation Model API helpers.

This file is the only place where the project directly calls Databricks-hosted
AI models. It uses the Databricks SDK serving endpoint client instead of an
OpenAI-compatible helper so notebook jobs, Databricks Apps, and local runs all
follow the same Databricks authentication chain.

Models used by default:
- Embeddings: `databricks-qwen3-embedding-0-6b`
- Chat response: `databricks-meta-llama-3-3-70b-instruct`

Both values come from environment variables through `load_settings()`, so an
instructor can switch endpoints without changing source code.
"""

from __future__ import annotations

import time
from typing import Any, Iterable

from productpulse.config import load_settings


def response_as_dict(response: Any) -> dict[str, Any]:
    """Convert Databricks SDK response objects into plain dictionaries.

    The SDK returns typed response objects in Databricks, while tests or future
    helper code may return dictionaries directly. Normalizing once here makes
    the embedding and chat parsers small and readable.
    """

    if isinstance(response, dict):
        return response
    if hasattr(response, "as_dict"):
        return response.as_dict()
    result: dict[str, Any] = {}
    for key in ["data", "predictions", "choices"]:
        value = getattr(response, key, None)
        if value is not None:
            result[key] = value
    return result


def extract_chat_text(response: Any) -> str:
    """Pull the assistant text out of a serving endpoint chat response."""
    payload = response_as_dict(response)
    choices = payload.get("choices") or []
    if not choices:
        return str(payload)
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        first_choice = first_choice.as_dict() if hasattr(first_choice, "as_dict") else {}
    
    message = first_choice.get("message") or {}
    if not isinstance(message, dict):
        message = message.as_dict() if hasattr(message, "as_dict") else {}
    return message.get("content") or first_choice.get("text") or ""
    

def chat_completion( 
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 900
) -> str:
    """Generate one chat response from a Databricks-hosted LLM.

    Where this is used:
    - `retrieval.rewrite_query()` asks the model to convert a vague business
      question into a better search query.
    - `agent.answer_question()` asks the model to synthesize the final answer
      from retrieved evidence, metrics, and memory.

    The function accepts separate system and user prompts because production
    RAG systems usually keep durable behavior instructions in the system prompt
    and request-specific context in the user prompt.
    """
    settings = load_settings()
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
    except ImportError as exc:
        raise RuntimeError(
            "databricks-sdk is required for LLM responses. "
            "Install requirements.txt before running the notebook or app."
        ) from exc
    
    # Keep temperature low for an enterprise copilot. The goal is grounded,
    # repeatable support guidance, not creative writing.
    workspace = WorkspaceClient()
    try:
        response = workspace.serving_endpoint.query(
            name = settings.llm_endpoint,
            messages = [
                ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt),
                ChatMessage(Role=ChatMessageRole.USER, content=user_prompt)
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
    except Exception as exc:
        raise RuntimeError(
            "Chat endpoint call failed. The configured endpoint is "
            f"'{settings.llm_endpoint}'. Confirm this endpoint exists in your "
            "Databricks workspace/region, or set PRODUCTPULSE_LLM_ENDPOINT to "
            f"an available chat endpoint. Last Databricks error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return extract_chat_text(response)