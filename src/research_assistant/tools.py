"""Tools the research agent can call, with the JSON schemas the LLM sees."""
import json
from pathlib import Path

from .arxiv_client import PaperInfo, download_paper, search_papers
from .config import Settings
from .ingest import chunk_pdf
from .vectorstore import VectorStore

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_arxiv",
            "description": (
                "Search arXiv for papers. Returns ids, titles and abstracts; nothing is downloaded. "
                'Supports arXiv query syntax, e.g. ti:"self-rag" or abs:retrieval AND abs:reranking.'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "arXiv search query"},
                    "max_results": {"type": "integer", "description": "1 to 8, default 5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_papers",
            "description": (
                "Download and index up to 3 papers so their full text can be searched. "
                "Only use ids returned by search_arxiv."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "arXiv ids, e.g. ['2310.11511v1']",
                    }
                },
                "required": ["arxiv_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_index",
            "description": (
                "Semantic search over the full text of indexed papers. Returns passages with "
                "their source (arXiv id and page) to cite."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "a focused search query"},
                    "k": {"type": "integer", "description": "number of passages, 1 to 8"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_indexed_papers",
            "description": "List the papers already indexed and the number of chunks.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def index_paper(paper, pdf_path: Path, store: VectorStore, settings: Settings) -> int:
    """Chunk a PDF and add it to the vector store. Returns the number of chunks."""
    chunks = chunk_pdf(pdf_path, settings.chunk_size, settings.chunk_overlap)
    if chunks:
        store.add(
            ids=[f"{paper.arxiv_id}-{c.chunk_index}" for c in chunks],
            texts=[c.text for c in chunks],
            metadatas=[
                {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": ", ".join(paper.authors),
                    "published": paper.published,
                    "url": paper.url,
                    "page": c.page,
                }
                for c in chunks
            ],
        )
    return len(chunks)


class ResearchToolbox:
    schemas = TOOL_SCHEMAS

    def __init__(self, store: VectorStore, settings: Settings):
        self.store = store
        self.settings = settings
        self.seen: dict[str, PaperInfo] = {}  # papers returned by search_arxiv

    def call(self, name: str, args: dict) -> str:
        """Run a tool and return its result as JSON. Errors are returned to the LLM, not raised."""
        handlers = {
            "search_arxiv": self.search_arxiv,
            "add_papers": self.add_papers,
            "search_index": self.search_index,
            "list_indexed_papers": self.list_indexed_papers,
        }
        if name not in handlers:
            result = {"error": f"unknown tool '{name}'"}
        else:
            try:
                result = handlers[name](**args)
            except (TypeError, ValueError, KeyError, OSError) as error:
                result = {"error": f"{type(error).__name__}: {error}"}
        return json.dumps(result, ensure_ascii=False)

    def search_arxiv(self, query: str, max_results: int = 5) -> dict:
        papers = search_papers(query, max(1, min(int(max_results), 8)))
        for paper in papers:
            self.seen[paper.arxiv_id] = paper
        return {
            "results": [
                {
                    "arxiv_id": p.arxiv_id,
                    "title": p.title,
                    "published": p.published,
                    "abstract": p.summary[:400],
                }
                for p in papers
            ]
        }

    def add_papers(self, arxiv_ids: list[str]) -> dict:
        if isinstance(arxiv_ids, str):
            arxiv_ids = [arxiv_ids]
        added, errors = [], []
        for arxiv_id in arxiv_ids[:3]:
            paper = self.seen.get(arxiv_id)
            if paper is None:
                errors.append(f"{arxiv_id}: unknown id, find it with search_arxiv first")
                continue
            try:
                path = download_paper(paper, self.settings.pdf_dir)
            except OSError as error:
                errors.append(f"{arxiv_id}: download failed ({error})")
                continue
            chunks = index_paper(paper, path, self.store, self.settings)
            added.append({"arxiv_id": arxiv_id, "title": paper.title, "chunks": chunks})
        return {"added": added, "errors": errors}

    def search_index(self, query: str, k: int = 5) -> dict:
        hits = self.store.search(query, max(1, min(int(k), 8)))
        if not hits:
            return {"results": [], "note": "the index is empty, add papers first"}
        return {
            "results": [
                {
                    "source": f"arXiv:{h['metadata']['arxiv_id']}, p.{h['metadata']['page']}",
                    "title": h["metadata"]["title"],
                    "score": round(h["score"], 3),
                    "text": h["text"][:900],
                }
                for h in hits
            ]
        }

    def list_indexed_papers(self) -> dict:
        data = self.store.collection.get(include=["metadatas"])
        papers = {m["arxiv_id"]: m["title"] for m in data["metadatas"]}
        return {
            "papers": [{"arxiv_id": i, "title": t} for i, t in papers.items()],
            "chunks": self.store.count(),
        }
