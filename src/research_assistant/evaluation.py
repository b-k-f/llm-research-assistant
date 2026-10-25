"""Stage 2: evaluate retrieval (and optionally answers) on a fixed corpus and question set."""
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI, RateLimitError
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from .arxiv_client import download_pdf
from .config import Settings
from .ingest import chunk_pdf
from .metrics import hit_at_k, is_relevant, parse_verdict, reciprocal_rank, reciprocal_rank_fusion
from .rag import SYSTEM_PROMPT, format_context

METHODS = ("vector", "hybrid", "hybrid+rerank")
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 20  # size of the candidate pool for fusion and reranking

JUDGE_PROMPT = """You grade answers to questions about scientific papers.
Compare the ANSWER with the REFERENCE answer.
Score 2 if the answer contains the key facts of the reference,
1 if it is partially correct or incomplete,
0 if it is wrong, missing, or says the information could not be found.
Reply with only a JSON object: {"score": 0, "reason": "one short sentence"}"""

TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class IndexedChunk:
    text: str
    arxiv_id: str
    page: int


class EvalIndex:
    """In-memory index holding both dense embeddings and a BM25 keyword index."""

    def __init__(self, chunks: list[IndexedChunk], embedder: SentenceTransformer):
        self.chunks = chunks
        self.embedder = embedder
        self.embeddings = embedder.encode(
            [c.text for c in chunks], normalize_embeddings=True, batch_size=64
        )
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def vector_ranking(self, query: str, n: int) -> list[int]:
        q = self.embedder.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q  # cosine similarity (vectors are normalised)
        return [int(i) for i in np.argsort(-scores)[:n]]

    def bm25_ranking(self, query: str, n: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(-scores)[:n]]


def retrieve(index: EvalIndex, query: str, method: str, k: int, reranker=None) -> list[int]:
    if method == "vector":
        return index.vector_ranking(query, k)

    fused = reciprocal_rank_fusion(
        [index.vector_ranking(query, CANDIDATES), index.bm25_ranking(query, CANDIDATES)]
    )
    if method == "hybrid":
        return fused[:k]

    pool = fused[:CANDIDATES]  # hybrid+rerank: a cross-encoder re-scores the candidates
    scores = reranker.predict([(query, index.chunks[i].text) for i in pool])
    return [pool[int(i)] for i in np.argsort(-scores)][:k]


def ensure_corpus(papers: list[dict], pdf_dir: Path) -> dict[str, Path]:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for paper in papers:
        path = pdf_dir / f"{paper['arxiv_id']}.pdf"
        if not path.exists():
            print(f"Downloading {paper['arxiv_id']} - {paper['title']}")
            download_pdf(f"https://arxiv.org/pdf/{paper['arxiv_id']}", path)
            time.sleep(1)
        paths[paper["arxiv_id"]] = path
    return paths


def build_chunks(pdf_paths: dict[str, Path], chunk_size: int, overlap: int) -> list[IndexedChunk]:
    return [
        IndexedChunk(text=c.text, arxiv_id=arxiv_id, page=c.page)
        for arxiv_id, path in pdf_paths.items()
        for c in chunk_pdf(path, chunk_size, overlap)
    ]


def evaluate_retrieval(
    index: EvalIndex, questions: list[dict], method: str, reranker
) -> tuple[dict, list[str]]:
    totals = {"hit@1": 0.0, "hit@5": 0.0, "mrr@10": 0.0, "paper_hit@5": 0.0}
    latencies, missed = [], []
    for q in questions:
        start = time.perf_counter()
        ranking = retrieve(index, q["question"], method, k=10, reranker=reranker)
        latencies.append((time.perf_counter() - start) * 1000)

        chunks = [index.chunks[i] for i in ranking]
        rel = [is_relevant(c.text, c.arxiv_id, q["arxiv_id"], q["evidence"]) for c in chunks]
        totals["hit@1"] += hit_at_k(rel, 1)
        totals["hit@5"] += hit_at_k(rel, 5)
        totals["mrr@10"] += reciprocal_rank(rel, 10)
        totals["paper_hit@5"] += hit_at_k([c.arxiv_id == q["arxiv_id"] for c in chunks], 5)
        if not any(rel[:5]):
            missed.append(q["id"])

    n = len(questions)
    metrics = {name: value / n for name, value in totals.items()}
    metrics["latency_ms"] = sum(latencies) / n
    return metrics, missed


def chat(client: OpenAI, model: str, messages: list[dict], retries: int = 4) -> str:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model, temperature=0, messages=messages
            )
            return response.choices[0].message.content or ""
        except RateLimitError:
            wait = 15 * (attempt + 1)
            print(f"  rate limited, waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError("LLM API still rate limited after several retries")


def judge_answers(
    questions: list[dict], index: EvalIndex, method: str, reranker,
    titles: dict[str, str], settings: Settings,
) -> list[dict]:
    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key or "not-needed")
    rows = []
    for q in questions:
        ranking = retrieve(index, q["question"], method, k=settings.top_k, reranker=reranker)
        hits = [
            {
                "text": index.chunks[i].text,
                "metadata": {
                    "title": titles[index.chunks[i].arxiv_id],
                    "arxiv_id": index.chunks[i].arxiv_id,
                    "page": index.chunks[i].page,
                },
            }
            for i in ranking
        ]
        answer = chat(client, settings.llm_model, [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sources:\n\n{format_context(hits)}\n\nQuestion: {q['question']}"},
        ])
        verdict = chat(client, settings.llm_model, [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": (
                f"QUESTION: {q['question']}\nREFERENCE: {q['reference']}\nANSWER: {answer}"
            )},
        ])
        score, reason = parse_verdict(verdict)
        rows.append({"id": q["id"], "question": q["question"], "answer": answer,
                     "score": score, "reason": reason})
        print(f"  judged {q['id']}: {score}")
        time.sleep(1)
    return rows


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "/")


def format_report(rows: list[dict], best: dict, misses: dict, n_questions: int,
                  n_papers: int, judge_rows: list[dict], model: str) -> str:
    lines = [
        "## Retrieval evaluation",
        "",
        f"{n_questions} questions over {n_papers} papers. A chunk is relevant when it comes from "
        "the expected paper and contains an evidence phrase.",
        "",
        "| Chunk size | Method | Chunks | Hit@1 | Hit@5 | MRR@10 | Paper hit@5 | Latency (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['chunk_size']} | {r['method']} | {r['chunks']} | {r['hit@1']:.2f} | "
            f"{r['hit@5']:.2f} | {r['mrr@10']:.2f} | {r['paper_hit@5']:.2f} | {r['latency_ms']:.0f} |"
        )
    key = f"{best['chunk_size']}/{best['method']}"
    lines += [
        "",
        f"**Best configuration:** chunk size {best['chunk_size']}, {best['method']} "
        f"(MRR@10 {best['mrr@10']:.2f}, Hit@5 {best['hit@5']:.2f})",
        f"Questions missed in the top 5 by the best configuration: {', '.join(misses[key]) or 'none'}",
    ]

    if judge_rows:
        scored = [r["score"] for r in judge_rows if r["score"] is not None]
        average = sum(scored) / len(scored) if scored else 0.0
        correct = sum(1 for s in scored if s == 2)
        lines += [
            "",
            "## Answer evaluation (LLM-as-judge)",
            "",
            f"Configuration: {key}. Generator and judge: `{model}`. "
            f"Average score: **{average:.2f} / 2**, fully correct: **{correct}/{len(judge_rows)}**.",
            "",
            "> Note: the same model generates and judges, which can bias scores upwards.",
            "",
            "| ID | Question | Score | Judge's reason |",
            "|---|---|---|---|",
        ]
        for r in judge_rows:
            score = "-" if r["score"] is None else r["score"]
            lines.append(f"| {r['id']} | {_cell(r['question'])} | {score} | {_cell(r['reason'])} |")
    return "\n".join(lines) + "\n"


def run_evaluation(eval_file: Path, chunk_sizes: list[int], methods: list[str], judge: bool,
                   out_dir: Path, settings: Settings) -> str:
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown methods {sorted(unknown)}, choose from {METHODS}")

    eval_set = json.loads(eval_file.read_text())
    papers, questions = eval_set["papers"], eval_set["questions"]
    titles = {p["arxiv_id"]: p["title"] for p in papers}

    pdf_paths = ensure_corpus(papers, settings.pdf_dir)
    embedder = SentenceTransformer(settings.embedding_model)
    reranker = CrossEncoder(RERANKER_MODEL) if "hybrid+rerank" in methods else None

    rows, misses, indexes = [], {}, {}
    for size in chunk_sizes:
        chunks = build_chunks(pdf_paths, size, overlap=size // 5)
        indexes[size] = EvalIndex(chunks, embedder)
        for method in methods:
            print(f"Evaluating chunk size {size}, {method}")
            metrics, missed = evaluate_retrieval(indexes[size], questions, method, reranker)
            rows.append({"chunk_size": size, "method": method, "chunks": len(chunks), **metrics})
            misses[f"{size}/{method}"] = missed

    best = max(rows, key=lambda r: (r["mrr@10"], r["hit@5"]))
    judge_rows = []
    if judge:
        print(f"Judging answers with chunk size {best['chunk_size']}, {best['method']}")
        judge_rows = judge_answers(questions, indexes[best["chunk_size"]], best["method"],
                                   reranker, titles, settings)

    report = format_report(rows, best, misses, len(questions), len(papers), judge_rows,
                           settings.llm_model)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(report)
    (out_dir / "results.json").write_text(json.dumps(
        {"retrieval": rows, "missed": misses, "best": best, "answers": judge_rows}, indent=2
    ))
    return report
