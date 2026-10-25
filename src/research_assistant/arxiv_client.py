"""Search arXiv and download paper PDFs."""
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import arxiv

USER_AGENT = "llm-research-assistant/0.1 (https://github.com/b-k-f/llm-research-assistant)"


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


def search_and_download(query: str, max_results: int, pdf_dir: Path) -> list[Paper]:
    """Search arXiv by relevance and download each PDF (skips files already downloaded)."""
    pdf_dir.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []
    for result in client.results(search):
        arxiv_id = result.get_short_id()
        pdf_path = pdf_dir / f"{arxiv_id.replace('/', '_')}.pdf"

        if not pdf_path.exists():
            pdf_url = result.pdf_url or result.entry_id.replace("/abs/", "/pdf/")
            try:
                download_pdf(pdf_url, pdf_path)
            except Exception as error:  # one failed download should not stop the run
                print(f"  ! could not download {arxiv_id}: {error}")
                continue
            time.sleep(1)  # be gentle with arXiv's servers

        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=result.title,
                authors=[a.name for a in result.authors],
                published=result.published.date().isoformat(),
                url=result.entry_id,
                pdf_path=pdf_path,
            )
        )
    return papers
