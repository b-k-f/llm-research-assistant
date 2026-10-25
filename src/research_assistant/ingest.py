"""Turn PDFs into text chunks that keep track of their page number."""
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Chunk:
    text: str
    page: int
    chunk_index: int


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return (page_number, text) for every page that contains text."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())  # normalise whitespace
        if text:
            pages.append((page_number, text))
    return pages


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks of ~chunk_size characters, cutting on spaces, with overlap.

    Overlap keeps sentences that fall on a boundary visible in both neighbouring chunks.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start:
                end = space  # avoid cutting a word in half
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = end - overlap
        if next_start > 0 and text[next_start - 1] != " ":
            # snap forward to the start of the next word so chunks never begin mid-word
            space = text.find(" ", next_start, end)
            next_start = space + 1 if space != -1 else end
        start = max(next_start, start + 1)  # always move forward
    return chunks


def chunk_pdf(pdf_path: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    chunks = []
    for page_number, text in extract_pages(pdf_path):
        for piece in split_text(text, chunk_size, overlap):
            chunks.append(Chunk(text=piece, page=page_number, chunk_index=len(chunks)))
    return chunks
