"""Command-line interface.

    research-assistant fetch "retrieval augmented generation" --max 5
    research-assistant ask "What chunking strategies do these papers compare?"
    research-assistant agent "How does Self-RAG decide when to retrieve?"
    research-assistant evaluate --chunk-sizes 500,1000,1500 --judge
    research-assistant stats
"""
import argparse
from pathlib import Path

from .arxiv_client import search_and_download
from .config import settings
from .rag import answer
from .tools import index_paper
from .vectorstore import VectorStore


def cmd_fetch(args: argparse.Namespace) -> None:
    print(f"Searching arXiv for: {args.query!r}")
    papers = search_and_download(args.query, args.max, settings.pdf_dir)
    store = VectorStore(settings.chroma_dir, settings.embedding_model)
    for paper in papers:
        chunks = index_paper(paper, paper.pdf_path, store, settings)
        if chunks:
            print(f"  + {paper.arxiv_id}  {paper.title[:70]}  ({chunks} chunks)")
        else:
            print(f"  ! no text extracted from {paper.arxiv_id}, skipped")
    print(f"Done. {store.count()} chunks in the index.")


def cmd_ask(args: argparse.Namespace) -> None:
    store = VectorStore(settings.chroma_dir, settings.embedding_model)
    result = answer(args.question, store, settings)
    print("\n" + result["answer"] + "\n")
    if result["sources"]:
        print("Sources:")
        for i, hit in enumerate(result["sources"], start=1):
            m = hit["metadata"]
            print(f"  [{i}] {m['title'][:70]} - p.{m['page']} - {m['url']} "
                  f"(score {hit['score']:.2f})")


def cmd_agent(args: argparse.Namespace) -> None:
    from openai import OpenAI, RateLimitError

    from .agent import format_trace, run_agent
    from .tools import ResearchToolbox

    store = VectorStore(settings.chroma_dir, settings.embedding_model)
    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key or "not-needed")
    result = run_agent(
        args.question,
        ResearchToolbox(store, settings),
        client,
        settings.llm_model,
        max_steps=args.max_steps,
        retry_on=(RateLimitError,),
    )
    trace = format_trace(args.question, result, settings.llm_model)
    trace_file = Path(args.trace_file)
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.write_text(trace)
    print("\n" + trace)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from .evaluation import run_evaluation  # heavy imports only when evaluating

    report = run_evaluation(
        eval_file=Path(args.eval_file),
        chunk_sizes=[int(size) for size in args.chunk_sizes.split(",")],
        methods=[m.strip() for m in args.methods.split(",")],
        judge=args.judge,
        out_dir=Path(args.out_dir),
        settings=settings,
    )
    print(report)


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

    p_agent = sub.add_parser("agent", help="let the agent find papers and answer")
    p_agent.add_argument("question")
    p_agent.add_argument("--max-steps", type=int, default=8)
    p_agent.add_argument("--trace-file", default="results/agent_trace.md")
    p_agent.set_defaults(func=cmd_agent)

    p_eval = sub.add_parser("evaluate", help="evaluate retrieval (and answers) on the test set")
    p_eval.add_argument("--eval-file", default="eval/questions.json")
    p_eval.add_argument("--chunk-sizes", default="500,1000,1500")
    p_eval.add_argument("--methods", default="vector,hybrid,hybrid+rerank")
    p_eval.add_argument("--judge", action="store_true", help="also grade answers with an LLM judge")
    p_eval.add_argument("--out-dir", default="results")
    p_eval.set_defaults(func=cmd_evaluate)

    p_stats = sub.add_parser("stats", help="show index size")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
