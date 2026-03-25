"""
core/document.py - Layer 1: Data Model
=======================================

The Document class is the "filing cabinet" for the RLM system.
It stores the full text OUTSIDE the LLM's attention mechanism.
The LLM only ever receives a short preview and word count — never
the full text directly. It reads chunks by calling split() and
sub_call() through the REPL.

Supports loading from:
  - PDF files  (via PyMuPDF / fitz)
  - Plain .txt files (utf-8)

Public API (also available as free functions in the REPL namespace):
  - doc.word_count       -> int   (approximates token count)
  - len(doc)             -> int   (same as word_count)
  - doc.fits_in_window(K)-> bool
  - doc.peek(start, end) -> str   (words[start:end] joined)
  - doc.slice(start, end)-> Document
  - doc.split(k)         -> list[Document]  (k equal chunks)
  - doc.preview(n)       -> str   (first n words + '...')
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Document dataclass
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A chunk of text kept outside the LLM's context window."""

    name: str
    content: str
    parent_name: Optional[str] = None
    depth: int = 0

    # Cached word list — computed once on first access
    _words: List[str] = field(default_factory=list, init=False, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_words(self) -> List[str]:
        """Return (and cache) the list of whitespace-split words."""
        if not self._words:
            self._words = self.content.split()
        return self._words

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def word_count(self) -> int:
        """Number of whitespace-delimited words — used as token proxy."""
        return len(self._get_words())

    def __len__(self) -> int:
        return self.word_count

    def fits_in_window(self, window_size: int) -> bool:
        """Return True when this document can be sent to the LLM in one shot."""
        return self.word_count <= window_size

    # ------------------------------------------------------------------
    # Reading operations (used by REPL namespace)
    # ------------------------------------------------------------------

    def peek(self, start: int, end: int) -> str:
        """
        Return words[start:end] as a single space-joined string.
        Safe: clamps indices to valid range.
        """
        words = self._get_words()
        start = max(0, start)
        end = min(len(words), end)
        return " ".join(words[start:end])

    def preview(self, n: int = 150) -> str:
        """First *n* words of content, followed by '...' if truncated."""
        words = self._get_words()
        snippet = " ".join(words[:n])
        return snippet + ("..." if len(words) > n else "")

    # ------------------------------------------------------------------
    # Slicing / splitting (the core RLM operation)
    # ------------------------------------------------------------------

    def slice(self, start: int, end: int) -> "Document":
        """
        Create a child Document from word indices [start, end).
        The child's name encodes its position for traceability.
        """
        words = self._get_words()
        start = max(0, start)
        end = min(len(words), end)
        child_content = " ".join(words[start:end])
        child_name = "{} [words {}-{}]".format(self.name, start, end)
        return Document(
            name=child_name,
            content=child_content,
            parent_name=self.name,
            depth=self.depth + 1,
        )

    def split(self, k: int) -> List["Document"]:
        """
        Split this document into *k* roughly-equal chunks.

        Raises:
            ValueError: if k < 1
        """
        if k < 1:
            raise ValueError(
                "split(k) requires k >= 1, got k={}".format(k)
            )
        words = self._get_words()
        total = len(words)
        if total == 0:
            return [Document(name=self.name + " [empty]", content="", parent_name=self.name)]

        # Ensure we don't request more chunks than words
        k = min(k, total)

        chunk_size = (total + k - 1) // k  # ceiling division
        chunks = []
        for i in range(k):
            s = i * chunk_size
            e = min(s + chunk_size, total)
            if s >= total:
                break
            chunks.append(self.slice(s, e))
        return chunks

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return "Document(name={!r}, words={}, depth={})".format(
            self.name, self.word_count, self.depth
        )


# ---------------------------------------------------------------------------
# Loader — reads PDF or TXT from a folder
# ---------------------------------------------------------------------------

def load_document_from_folder(folder_path: str) -> Document:
    """
    Load the first PDF or .txt file found in *folder_path*.

    PDF extraction uses PyMuPDF (fitz). Raises FileNotFoundError if
    no supported file is found, ImportError if fitz is missing for PDF.

    Args:
        folder_path: Path to the folder containing the document.

    Returns:
        A Document instance with the full extracted text.

    Raises:
        FileNotFoundError: No PDF or .txt file found.
        ImportError: PyMuPDF not installed (PDF only).
        RuntimeError: PDF extracted no text (scanned / image PDF).
    """
    folder_path = os.path.abspath(folder_path)

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            "Folder not found: '{}'. Create it and drop a PDF or .txt file inside.".format(
                folder_path
            )
        )

    # Collect candidates in priority order: PDF first, then TXT
    pdf_files = sorted(glob.glob(os.path.join(folder_path, "*.pdf")))
    txt_files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))
    candidates = pdf_files + txt_files

    if not candidates:
        raise FileNotFoundError(
            "No .pdf or .txt files found in '{}'. Drop a document file there.".format(
                folder_path
            )
        )

    file_path = candidates[0]
    ext = os.path.splitext(file_path)[1].lower()
    base_name = os.path.basename(file_path)

    if ext == ".pdf":
        content = _extract_pdf_text(file_path)
    else:
        content = _extract_txt_text(file_path)

    if not content.strip():
        raise RuntimeError(
            "Extracted no text from '{}'. "
            "The file may be empty, scanned, or image-only.".format(base_name)
        )

    return Document(name=base_name, content=content)


def _extract_pdf_text(file_path: str) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required to read PDF files.\n"
            "Install it with:  pip install pymupdf"
        )

    pages = []
    with fitz.open(file_path) as doc:
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text)

    return "\n".join(pages)


def _extract_txt_text(file_path: str) -> str:
    """Read a plain text file (UTF-8 with BOM fallback)."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise RuntimeError(
        "Could not decode '{}' with utf-8 or latin-1.".format(
            os.path.basename(file_path)
        )
    )
