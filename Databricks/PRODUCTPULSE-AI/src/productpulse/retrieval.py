def retrieve_context(
    conn,
    question: str,
    product_id: str | None = None,
) -> dict[str, object]:
    """Run the full RAG retrieval sequence.

    Steps performed:
    1. Rewrite the user's question for better search.
    2. Embed the rewritten query with the Databricks embedding endpoint.
    3. Search Lakebase `rag_chunks` for nearest text chunks.
    4. Rerank candidates with semantic distance plus keyword overlap.
    5. Format top evidence into numbered citation blocks.

    The final answer model sees numbered blocks such as `[1]`, `[2]`, so it can
    cite where each claim came from.
    """