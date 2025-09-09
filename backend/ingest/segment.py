from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional
import re, math

from syntok import segmenter
import tiktoken

# --- token counter ---
_enc = tiktoken.get_encoding("cl100k_base")
def count_tokens(text: str) -> int:
    return len(_enc.encode(text or ""))

# --- heading detection (HOA style + RU/EN) ---
_HEADING_RE = re.compile(
    r"^\s*(?:ARTICLE|SECTION|CHAPTER|APPENDIX|РАЗДЕЛ|СТАТЬЯ|ПРИЛОЖЕНИЕ|§)\b[^\n]{0,80}$|^[A-Z0-9 .:-]{6,120}$"
)

def is_heading(line: str) -> bool:
    line = (line or "").strip()
    if not line: return False
    if len(line) > 120: return False
    return bool(_HEADING_RE.match(line))

@dataclass
class SentFrag:
    page: int
    text: str

def sentence_fragments(pages: List[Tuple[int, str]]) -> List[SentFrag]:
    """pages: list of (page_number, page_text). Returns sentence fragments with page tags."""
    out: List[SentFrag] = []
    for page, txt in pages:
        if not txt or not txt.strip():
            continue
            
        # Clean text: normalize whitespace and fix common PDF issues
        cleaned_txt = txt.replace('\n', ' ').replace('\r', ' ')
        # Fix common PDF spacing issues
        cleaned_txt = re.sub(r'\s+', ' ', cleaned_txt).strip()
        
        if not cleaned_txt:
            continue
            
        # Check if text contains Cyrillic characters (Russian text)
        has_cyrillic = bool(re.search(r'[а-яё]', cleaned_txt, re.IGNORECASE))
        
        if has_cyrillic:
            # Use simple sentence splitting for Russian text
            sentences = re.split(r'[.!?]+', cleaned_txt)
            for sent in sentences:
                sent = sent.strip()
                if sent and len(sent) > 10:
                    out.append(SentFrag(page=page, text=sent))
        else:
            # Use syntok for English text
            try:
                for para in segmenter.process(cleaned_txt):
                    para_txt = "".join(tok.value for sent in para for tok in sent).strip()
                    if not para_txt:
                        continue
                    # split to sentences
                    for sent in para:
                        s = "".join(tok.value for tok in sent).strip()
                        if s and len(s) > 10:  # Filter out very short fragments
                            out.append(SentFrag(page=page, text=s))
            except Exception as e:
                print(f"Error processing page {page} with syntok: {e}")
                # Fallback: simple sentence splitting
                sentences = re.split(r'[.!?]+', cleaned_txt)
                for sent in sentences:
                    sent = sent.strip()
                    if sent and len(sent) > 10:
                        out.append(SentFrag(page=page, text=sent))
    return out

def pack_sentences(fragments: List[SentFrag], max_tokens=500, overlap_sents=2):
    """Yield chunks preserving sentence boundaries and page info."""
    if not fragments: return
    buf: List[SentFrag] = []
    cur_tok = 0
    for frag in fragments:
        t = count_tokens(frag.text)
        if cur_tok and cur_tok + t > max_tokens:
            # flush
            text = " ".join(f.text for f in buf).strip()
            pages = sorted({f.page for f in buf})
            yield {"text": text, "pages": pages, "token_count": cur_tok}
            # overlap by sentences
            buf = buf[-overlap_sents:] if overlap_sents > 0 else []
            cur_tok = sum(count_tokens(f.text) for f in buf)
        buf.append(frag)
        cur_tok += t
    if buf:
        text = " ".join(f.text for f in buf).strip()
        pages = sorted({f.page for f in buf})
        yield {"text": text, "pages": pages, "token_count": cur_tok}

# --- topic segmentation (adjacent similarity drop) ---
def cosine(a, b):
    import math
    s = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(x*x for x in b))
    return s / (na*nb + 1e-8)

def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    from openai import OpenAI
    import os
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # batching for long docs
    out = []
    B = 100
    for i in range(0, len(texts), B):
        resp = client.embeddings.create(model=model, input=texts[i:i+B])
        out.extend([d.embedding for d in resp.data])
    return out

def paragraphs_from_pages(pages: List[Tuple[int,str]]) -> List[Tuple[int,str]]:
    """Rough paragraphs per page using blank lines / bullets as separators."""
    out: List[Tuple[int,str]] = []
    for page, txt in pages:
        if not txt: continue
        parts = re.split(r"\n\s*\n+", txt)
        for p in parts:
            p = p.strip()
            if len(p) < 20:  # skip noise
                continue
            out.append((page, p))
    return out

def topic_segments(pages: List[Tuple[int,str]], min_sim_drop=0.20, min_para=2, max_tokens=2000):
    """
    Segment by detecting drops in similarity between consecutive paragraph embeddings.
    min_sim_drop: boundary if sim[i] < (rolling_mean - drop)
    """
    paras = paragraphs_from_pages(pages)
    if not paras:
        return []
    
    # For very large documents, limit paragraphs to avoid timeout
    if len(paras) > 100:
        print(f"[info] Large document detected ({len(paras)} paragraphs), using sampling for topic segmentation")
        # Sample every 3rd paragraph to reduce processing time
        paras = paras[::3]
        print(f"[info] Sampled to {len(paras)} paragraphs")
    
    texts = [p for _,p in paras]
    
    # Batch embeddings to avoid timeout
    print(f"[info] Computing embeddings for {len(texts)} paragraphs...")
    vecs = embed_texts(texts)
    print(f"[info] Embeddings computed, analyzing similarities...")
    
    # local similarity between neighbors
    sims = [cosine(vecs[i], vecs[i+1]) for i in range(len(vecs)-1)]
    # rolling mean
    import statistics
    mean = statistics.fmean(sims) if sims else 0.0
    thr = mean - min_sim_drop
    # cut indices
    cuts = [i+1 for i,s in enumerate(sims) if s < thr]
    # always include 0 and len
    idxs = [0] + cuts + [len(paras)]
    # build segments
    segs = []
    for k in range(len(idxs)-1):
        lo, hi = idxs[k], idxs[k+1]
        if hi - lo < min_para and k+1 < len(idxs)-1:
            # merge too small segment forward
            idxs[k+1] = hi = min(len(paras), idxs[k+1] + (min_para - (hi-lo)))
        chunk_texts = []
        pages_set = set()
        tok = 0
        for i in range(lo, hi):
            pg, txt = paras[i]
            pages_set.add(pg)
            tok_next = tok + count_tokens(txt)
            if tok_next > max_tokens and chunk_texts:
                # flush current subchunk
                segs.append({
                    "topic_index": len(segs),
                    "text": "\n\n".join(chunk_texts).strip(),
                    "pages": sorted(pages_set),
                })
                chunk_texts = [txt]; pages_set = {pg}; tok = count_tokens(txt)
            else:
                chunk_texts.append(txt); tok = tok_next
        if chunk_texts:
            segs.append({
                "topic_index": len(segs),
                "text": "\n\n".join(chunk_texts).strip(),
                "pages": sorted(pages_set),
            })
    print(f"[info] Created {len(segs)} topic segments")
    return segs
