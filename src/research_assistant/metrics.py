"""Pure evaluation helpers (no heavy dependencies, so they are cheap to unit-test)."""
import json
import re


def is_relevant(chunk_text: str, chunk_paper: str, expected_paper: str, evidence: list[str]) -> bool:
    """A chunk is relevant if it comes from the expected paper and contains an evidence phrase."""
    if chunk_paper != expected_paper:
        return False
    text = chunk_text.lower()
    return any(phrase.lower() in text for phrase in evidence)


def hit_at_k(relevances: list[bool], k: int) -> float:
    """1.0 if at least one of the top-k results is relevant, else 0.0."""
    return 1.0 if any(relevances[:k]) else 0.0


def reciprocal_rank(relevances: list[bool], k: int) -> float:
    """1 / rank of the first relevant result within the top k (0.0 if none)."""
    for rank, relevant in enumerate(relevances[:k], start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Merge several rankings of document ids: score(d) = sum of 1 / (k + rank of d)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)


def parse_verdict(text: str | None) -> tuple[int | None, str]:
    """Extract {"score": 0-2, "reason": "..."} from an LLM judge reply."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None, "unparseable judge output"
    try:
        data = json.loads(match.group(0))
        score = int(data.get("score"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, "unparseable judge output"
    if score not in (0, 1, 2):
        return None, f"invalid score {score}"
    return score, str(data.get("reason", ""))
