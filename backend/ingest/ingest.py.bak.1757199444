#!/usr/bin/env python3
"""
Document ingestion module for HOA AI Assistant
"""

import re
import os
from pathlib import Path
from typing import List, Dict
import tiktoken
from pypdf import PdfReader

def extract_text(path: str) -> str:
    """
    Extract text from .txt or .pdf files
    
    Args:
        path: Path to the file
        
    Returns:
        Extracted and cleaned text
    """
    file_path = Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if file_path.suffix.lower() == '.txt':
        # Read text file as UTF-8
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    
    elif file_path.suffix.lower() == '.pdf':
        # Read PDF file
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    # Remove null bytes and garbage characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    return text

def clean_text(s: str) -> str:
    """
    Clean and normalize text
    
    Args:
        s: Input text
        
    Returns:
        Cleaned text
    """
    # Replace multiple spaces with single space
    s = re.sub(r' +', ' ', s)
    
    # Normalize line breaks (max 2 consecutive)
    s = re.sub(r'\n{3,}', '\n\n', s)
    
    # Trim whitespace
    s = s.strip()
    
    return s

def estimate_tokens(s: str) -> int:
    """
    Estimate number of tokens using tiktoken
    
    Args:
        s: Input text
        
    Returns:
        Number of tokens
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(s))

def split_text_by_tokens(text: str, max_tokens: int = None, overlap: int = None, encoding_name: str = "cl100k_base") -> list[str]:
    """
    Режет текст по токенам с overlap. Возвращает список строк-чанков.
    """
    # Get default values from environment variables
    if max_tokens is None:
        max_tokens = int(os.getenv("CHUNK_MAX_TOKENS", "800"))
    if overlap is None:
        overlap = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))
    
    enc = tiktoken.get_encoding(encoding_name)
    toks = enc.encode(text)
    chunks = []
    if max_tokens <= 0:
        return [text] if text else []
    i = 0
    while i < len(toks):
        window = toks[i:i+max_tokens]
        chunks.append(enc.decode(window))
        i += max_tokens - overlap if max_tokens > overlap else max_tokens
    return chunks

def chunk_text(s: str, max_tokens: int = None, overlap: int = None) -> List[Dict]:
    """
    Split text into chunks with overlap
    
    Args:
        s: Input text
        max_tokens: Maximum tokens per chunk (defaults to CHUNK_MAX_TOKENS from env)
        overlap: Number of overlapping tokens between chunks (defaults to CHUNK_OVERLAP_TOKENS from env)
        
    Returns:
        List of chunks with text, section reference, and token count
    """
    # Clean text first
    s = clean_text(s)
    
    # Use split_text_by_tokens for token-based splitting
    text_chunks = split_text_by_tokens(s, max_tokens, overlap)
    
    chunks = []
    for idx, chunk_text in enumerate(text_chunks):
        # Get token count for this chunk
        token_count = estimate_tokens(chunk_text)
        
        chunks.append({
            "text": chunk_text.strip(),
            "token_count": token_count,
            "section_ref": f"p{idx+1}"
        })
    
    return chunks

if __name__ == "__main__":
    # Mini-test: read a file from data/ and show token count and chunks
    data_dir = Path(__file__).parent.parent.parent / "data"
    
    if data_dir.exists():
        # Find first available file
        for file_path in data_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.pdf']:
                try:
                    print(f"Testing file: {file_path.name}")
                    
                    # Extract text
                    text = extract_text(str(file_path))
                    print(f"Original text length: {len(text)} characters")
                    
                    # Clean text
                    cleaned_text = clean_text(text)
                    print(f"Cleaned text length: {len(cleaned_text)} characters")
                    
                    # Count tokens
                    token_count = estimate_tokens(cleaned_text)
                    print(f"Token count: {token_count}")
                    
                    # Create chunks
                    chunks = chunk_text(cleaned_text, max_tokens=600, overlap=80)
                    print(f"Number of chunks: {len(chunks)}")
                    
                    # Show first chunk as example
                    if chunks:
                        print(f"\nFirst chunk ({chunks[0]['section_ref']}):")
                        print(f"Tokens: {estimate_tokens(chunks[0]['text'])}")
                        print(f"Text: {chunks[0]['text'][:200]}...")
                    
                    break
                    
                except Exception as e:
                    print(f"Error processing {file_path.name}: {e}")
                    continue
        else:
            print("No .txt or .pdf files found in data/ directory")
    else:
        print("data/ directory not found")
