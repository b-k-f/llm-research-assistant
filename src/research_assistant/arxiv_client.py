"""Search arXiv and download paper PDFs."""
from dataclasses import dataclass
from pathlib import Path

import arxiv


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    url: str
    pdf_path: Path


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
        filename = f"{arxiv_id.replace('/', '_')}.pdf"
        pdf_path = pdf_dir / filename
        if not pdf_path.exists():
            result.download_pdf(dirpath=str(pdf_dir), filename=filename)

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
