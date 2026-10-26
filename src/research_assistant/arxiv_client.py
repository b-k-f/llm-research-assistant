"""Search arXiv and download paper PDFs."""
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import arxiv

USER_AGENT = "llm-research-assistant/0.1 (https://github.com/b-k-f/llm-research-assistant)"


@dataclass
class PaperInfo:
    """Metadata from an arXiv search result (nothing downloaded yet)."""

    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    url: str
    pdf_url: str
    summary: str


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    url: str
    pdf_path: Path


def download_pdf(url: str, dest: Path) -> None:
    """Download a PDF with a polite User-Agent (arXiv asks clients to identify themselves)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        dest.write_bytes(response.read())


def search_papers(query: str, max_results: int) -> list[PaperInfo]:
    """Search arXiv by relevance and return metadata only."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )
    papers = []
    for result in client.results(search):
        papers.append(
            PaperInfo(
                arxiv_id=result.get_short_id(),
                title=" ".join(result.title.split()),
                authors=[a.name for a in result.authors],
                published=result.published.date().isoformat(),
                url=result.entry_id,
                pdf_url=result.pdf_url or result.entry_id.replace("/abs/", "/pdf/"),
                summary=" ".join((getattr(result, "summary", "") or "").split()),
            )
        )
    return papers


def download_paper(info: PaperInfo, pdf_dir: Path) -> Path:
    """Download one paper's PDF (skipped if already downloaded) and return its path."""
    pdf_dir.mkdir(parents=True, exist_ok=True)
    path = pdf_dir / f"{info.arxiv_id.replace('/', '_')}.pdf"
    if not path.exists():
        download_pdf(info.pdf_url, path)
        time.sleep(1)  # be gentle with arXiv's servers
    return path


def search_and_download(query: str, max_results: int, pdf_dir: Path) -> list[Paper]:
    """Search arXiv and download every result (used by the `fetch` command)."""
    papers = []
    for info in search_papers(query, max_results):
        try:
            path = download_paper(info, pdf_dir)
        except OSError as error:  # network, HTTP or file errors: skip this paper
            print(f"  ! could not download {info.arxiv_id}: {error}")
            continue
        papers.append(
            Paper(
                arxiv_id=info.arxiv_id,
                title=info.title,
                authors=info.authors,
                published=info.published,
                url=info.url,
                pdf_path=path,
            )
        )
    return papers
