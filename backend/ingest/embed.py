#!/usr/bin/env python3
"""
Text embedding module for HOA AI Assistant
"""

import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import tiktoken

def get_embedding(text: str) -> List[float]:
    """
    Get text embedding using OpenAI API
    
    Args:
        text: Input text to embed
        
    Returns:
        List of floats (vector of length 1536)
    """
    # Load environment variables
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Check if text is empty
    if not text or not text.strip():
        return []
    
    # Truncate text to model limit before calling API
    enc = tiktoken.get_encoding("cl100k_base")
    MAX = 8191  # чуть меньше лимита модели
    toks = enc.encode(text)
    if len(toks) > MAX:
        toks = toks[:MAX]
        text = enc.decode(toks)
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Get embedding
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        
        # Return the embedding vector
        return response.data[0].embedding
        
    except Exception as e:
        print(f"Error getting embedding: {e}")
        raise
