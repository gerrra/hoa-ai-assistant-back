from __future__ import annotations
from typing import Iterable, Dict, Optional, Tuple
import io, re, math

from pypdf import PdfReader

def _split_text_windows(txt: str, max_chars: int = 1200, overlap: int = 200):
    txt = txt.replace("\r", "")
    i = 0
    n = len(txt)
    while i < n:
        j = min(i + max_chars, n)
        # try cut on sentence boundary
        cut = txt.rfind("\n", i + max(0, j - i - 200), j)
        if cut == -1:
            cut = txt.rfind(". ", i + max(0, j - i - 200), j)
        if cut != -1 and cut > i + 200:
            j = cut + 1
        chunk = txt[i:j].strip()
        if chunk:
            yield chunk, (i, j)
        if j >= n: break
        i = max(0, j - overlap)

def iter_pdf_chunks(data: bytes, filename: str, max_chars: int = 1200, overlap: int = 200):
    """
    Yields dicts: {index, page, text, start, end, filename}
    """
    reader = PdfReader(io.BytesIO(data))
    idx = 0
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if not txt.strip():
            continue
        for text, (start, end) in _split_text_windows(txt, max_chars=max_chars, overlap=overlap):
            yield {
                "index": idx,
                "page": page_num,
                "text": text,
                "start": start,
                "end": end,
                "filename": filename,
            }
            idx += 1
