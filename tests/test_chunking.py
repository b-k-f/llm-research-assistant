import itertools

import pytest

from research_assistant.ingest import split_text


def test_short_text_gives_one_chunk():
    assert split_text("hello world", chunk_size=100, overlap=10) == ["hello world"]


def test_chunks_respect_size():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_all_words_are_covered():
    words = [f"w{i}" for i in range(300)]
    chunks = split_text(" ".join(words), chunk_size=120, overlap=30)
    covered = set(" ".join(chunks).split())
    assert covered == set(words)


def test_consecutive_chunks_overlap():
    text = " ".join(f"token{i}" for i in range(200))
    chunks = split_text(text, chunk_size=150, overlap=40)
    for a, b in itertools.pairwise(chunks):
        assert set(a.split()) & set(b.split())


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=10, overlap=10)
