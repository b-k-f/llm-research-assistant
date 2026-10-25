"""Retrieval-augmented generation: retrieve chunks, then ask the LLM to answer with citations."""
from openai import OpenAI

from .config import Settings
from .vectorstore import VectorStore

SYSTEM_PROMPT = """You are a research assistant that answers questions about scientific papers.
Rules:
- Use ONLY the numbered sources provided. Do not use outside knowledge.
- Cite every claim inline with the source number, e.g. [1] or [2][3].
- If the sources do not contain the answer, say "I couldn't find this in the indexed papers."
- Be concise and precise."""


def format_context(hits: list[dict]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        m = hit["metadata"]
        blocks.append(f"[{i}] {m['title']} (arXiv:{m['arxiv_id']}, page {m['page']})\n{hit['text']}")
    return "\n\n".join(blocks)


def answer(question: str, store: VectorStore, settings: Settings) -> dict:
    hits = store.search(question, settings.top_k)
    if not hits:
        return {"answer": "No papers indexed yet. Run the `fetch` command first.", "sources": []}

    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key or "not-needed")
    response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sources:\n\n{format_context(hits)}\n\nQuestion: {question}"},
        ],
    )
    return {"answer": response.choices[0].message.content, "sources": hits}
