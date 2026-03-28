"""
Document loading and slicing utilities for RLM.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

def extract_text_from_pdf(pdf_path: str)-> str:
    """Extract all text from a PDF using PyMuPDF (fitz)."""
    try:
        import fitz
        doc=fitz.open(pdf_path)
        pages=[]
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n".join(pages)
    except ImportError:
        raise ImportError("Run: pip install pymupdf")
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF '{pdf_path}:{e}")
    
def load_document_from_folder(folder: str ="document_pdf")-> "Document":
    """
    Scan the document_pdf/folder and load the first supported file found.
    """

    folder_path=Path(folder)
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
        raise FileNotFoundError(
            f"folder '{folder}' was empty or missing."
        )
    supported=[".pdf",".txt",".md"]
    found_files=[
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in supported
    ]

    if not found_files:
        raise FileNotFoundError(
            f"No supported files in '{folder}/'."
        )
    
    found_files.sort()
    chosen = found_files[0]
    print(f"[Loader] Found: {chosen.name}")
 
    if chosen.suffix.lower() == ".pdf":
        content = extract_text_from_pdf(str(chosen))
    else:
        content = chosen.read_text(encoding="utf-8", errors="replace")
 
    content = " ".join(content.split())
 
    print(f"[Loader] Loaded {len(content.split()):,} words from '{chosen.name}'")
    return Document(content=content, name=chosen.name)   
    
@dataclass
class Document:
    """
    Stores the full document text externally - NOT in the LLM's context window
    """

    content:str
    name: str="document"
    parent_name: Optional[str]=None
    depth: int=0

    @property
    def word_count(self)->int:
        """
        Approximate token count
        Use tiktoken.encode() for exact counts.
        """

        return len(self.content.split())
    
    def __len__(self)->int:
        return self.word_count
    
    def fits_in_window(self, context_window: int) -> bool:
        """
        True if the chunk fits within the configured context window.
        """
        return self.word_count <= context_window

    def fits_int_windows(self, context_window: int) -> bool:
        """
        Backward-compatible alias for the old misspelled method name.
        """
        return self.fits_in_window(context_window)
    
    def peek(self,start: int=0, end: int=100)->str:
        """
        Read a slice of text by word index.
        Used by LLm to inspect a chunk before deciding to process it.
        
        In RLM: the llm generates 'peek(p,0,200) in its code
        """

        words=self.content.split()
        return " ".join(words[start:end])
    
    def split(self,k:int)->list["Document"]:
        """
        Split into k equal chunks. Always deterministic same k -> same chunks
        In RLM: the LLM decided k
        """

        if k<1:
            raise ValueError(f"split(k): k must be >=1. got {k}")
        if k==1:
            return[self]
        
        words = self.content.split()
        chunk_size = max(1, len(words) // k)
        chunks = []
        for i in range(k):
            start = i * chunk_size
            end = start + chunk_size if i < k - 1 else len(words)
            chunks.append(Document(
                content=" ".join(words[start:end]),
                name=f"{self.name}[chunk {i+1}/{k}]",
                parent_name=self.name,
                depth=self.depth + 1,
            ))
        return chunks

    def slice(self, start: int, end: int) -> "Document":
        """
        Return a sub-document using word offsets.
        """
        words = self.content.split()
        start = max(0, start)
        end = min(len(words), end)
        return Document(
            content=" ".join(words[start:end]),
            name=f"{self.name}[words {start}:{end}]",
            parent_name=self.name,
            depth=self.depth + 1,
        )
 
    def preview(self, n: int = 60) -> str:
        p = self.peek(0, n)
        return p + ("..." if self.word_count > n else "")
 
    def __repr__(self):
        return f"Document('{self.name}', {self.word_count} words)"
