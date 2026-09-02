"""Assemble the evaluation corpus of public-domain PDFs.

Run once: python eval/build_corpus.py
Downloads three PDFs and renders one Project Gutenberg text to PDF, so the
retrieval benchmark spans four document types:

    transformer.pdf   research paper   (arXiv 1706.03762)
    rag_paper.pdf      research paper   (arXiv 2005.11401)
    nist_ai_rmf.pdf    government report(NIST AI 100-1)
    art_of_war.pdf     classic prose   (Gutenberg #132)

The existing data/EffectiveProjectManagement_Wysocki.pdf (technical book) is the
fifth document and is not downloaded here.
"""

import re
import textwrap
import urllib.request
from pathlib import Path

from fpdf import FPDF
from fpdf.errors import FPDFException

OUT = Path(__file__).resolve().parent.parent / "data" / "eval"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

PDFS = {
    "transformer.pdf": "https://arxiv.org/pdf/1706.03762",
    "rag_paper.pdf": "https://arxiv.org/pdf/2005.11401",
    "nist_ai_rmf.pdf": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
}
GUTENBERG_TXT = "https://www.gutenberg.org/cache/epub/132/pg132.txt"


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def download_pdfs():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, url in PDFS.items():
        dest = OUT / name
        if dest.exists():
            print(f"  {name} exists, skipping")
            continue
        print(f"  downloading {name} ...")
        dest.write_bytes(_get(url))


def build_art_of_war():
    dest = OUT / "art_of_war.pdf"
    if dest.exists():
        print("  art_of_war.pdf exists, skipping")
        return
    print("  fetching + rendering art_of_war.pdf ...")
    text = _get(GUTENBERG_TXT).decode("utf-8", "replace")

    # keep only the book body, dropping the licence boilerplate
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if start != -1 and end != -1:
        text = text[text.find("\n", start) + 1:end]

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    wrapper = textwrap.TextWrapper(
        width=80, break_long_words=True, break_on_hyphens=True
    )
    for raw in text.splitlines():
        safe = raw.encode("latin-1", "replace").decode("latin-1").rstrip()
        # collapse runs of divider punctuation that have no break points
        safe = re.sub(r"([_=*.\-])\1{6,}", r"\1\1\1", safe)
        if not safe:
            pdf.ln(3)
            continue
        for piece in wrapper.wrap(safe) or [safe]:
            try:
                pdf.multi_cell(0, 5, piece[:80])
            except FPDFException:
                pass
    pdf.output(str(dest))


if __name__ == "__main__":
    print("Building evaluation corpus in", OUT)
    download_pdfs()
    build_art_of_war()
    print("\nCorpus:")
    for p in sorted(OUT.glob("*.pdf")):
        print(f"  {p.name:20s} {p.stat().st_size // 1024:>6d} KB")
