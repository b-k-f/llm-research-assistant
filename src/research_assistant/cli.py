"""Command-line interface.

    research-assistant fetch "retrieval augmented generation" --max 5
    research-assistant ask "What chunking strategies do these papers compare?"
    research-assistant stats
"""
import argparse

from .arxiv_client import search_and_download
from .config import settings
from .ingest import chunk_pdf
from .rag import answer
from .vectorstore import VectorStore


def cmd_fetch(args: argparse.Namespace) -> None:
    print(f"Searching arXiv for: {args.query!r}")
    papers = search_and_download(args.query, args.max, settings.pdf_dir)
    store = VectorStore(settings.chroma_dir, settings.embedding_model)

    for paper in papers:
        chunks = chunk_pdf(paper.pdf_path, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            print(f"  ! no text extracted from {paper.arxiv_id}, skipped")
            continue
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
        print(f"  + {paper.arxiv_id}  {paper.title[:70]}  ({len(chunks)} chunks)")
    print(f"Done. {store.count()} chunks in the index.")


def cmd_ask(args: argparse.Namespace) -> None:
    store = VectorStore(settings.chroma_dir, settings.embedding_model)
    result = answer(args.question, store, settings)
    print("\n" + result["answer"] + "\n")
    if result["sources"]:
        print("Sources:")
        for i, hit in enumerate(result["sources"], start=1):
            m = hit["metadata"]
            print(f"  [{i}] {m['title'][:70]} - p.{m['page']} - {m['url']} (score {hit['score']:.2f})")


def cmd_stats(_: argparse.Namespace) -> None:
    store = VectorStore(settings.chroma_dir, settings.embedding_model)
    print(f"{store.count()} chunks indexed in {settings.chroma_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="research-assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="search arXiv, download and index papers")
    p_fetch.add_argument("query")
    p_fetch.add_argument("--max", type=int, default=5)
    p_fetch.set_defaults(func=cmd_fetch)

    p_ask = sub.add_parser("ask", help="ask a question about the indexed papers")
    p_ask.add_argument("question")
    p_ask.set_defaults(func=cmd_ask)

    p_stats = sub.add_parser("stats", help="show index size")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
