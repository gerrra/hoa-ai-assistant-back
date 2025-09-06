#!/usr/bin/env python3
"""
LLM answer generation module for HOA AI Assistant
"""

import os
from typing import Dict, List
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Import local modules
from ..search.retrieve import search_chunks, format_context

def get_openai_client():
    """Get OpenAI client with API key from environment"""
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    return OpenAI(api_key=api_key)

def answer_question(community_id: int, role: str, question: str, k: int = 6) -> Dict:
    """
    Generate answer to user question using LLM and document search
    
    Args:
        community_id: ID of the community
        role: User role (resident, board, staff)
        question: User question
        k: Number of chunks to retrieve for context
        
    Returns:
        Dictionary with answer, sources, and confidence
    """
    try:
        # Step 1: Import search functions
        from ..search.retrieve import search_chunks, format_context
        
        # Step 2: Search for relevant chunks
        chunks = search_chunks(community_id, question, k=k)
        
        # Step 3: Check if chunks are empty
        if not chunks:
            return {
                "answer": "Недостаточно данных в документах, обратитесь к менеджеру сообщества.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Step 4: Build system prompt
        system_prompt = """Ты — ассистент HOA. Отвечай строго на основе ПРИЛОЖЕННЫХ ФРАГМЕНТОВ. Если информации недостаточно — скажи обратиться к менеджеру. Формат: (1) краткий ответ; (2) детали; (3) источники с точными разделами (например "CC&R §5.2; Bylaws Art. III, Sec. 7")."""
        
        # Step 5: Build user prompt
        context = format_context(chunks)
        user_prompt = f"{question}\n\nCONTEXT:\n{context}"
        
        # Step 6: Call OpenAI Chat Completions
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for consistent answers
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Step 7: Build sources list
        sources = []
        for chunk in chunks:
            sources.append({
                "title": chunk["document_title"],
                "section": chunk["section_ref"]
            })
        
        # Step 8: Calculate average confidence
        if chunks:
            avg_similarity = sum(chunk["similarity"] for chunk in chunks) / len(chunks)
        else:
            avg_similarity = 0.0
        
        # Step 9: Build meta information with effective dates
        effective_dates = set()
        for chunk in chunks:
            if chunk.get("effective_from"):
                effective_dates.add(str(chunk["effective_from"]))
        
        meta = {"effective_dates": sorted(list(effective_dates))}
        
        # Step 10: Prepend effective dates to answer if available
        if effective_dates:
            dates_str = ", ".join(sorted(effective_dates))
            answer = f"Актуально по документам от: {dates_str}\n\n{answer}"
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": avg_similarity,
            "meta": meta
        }
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        # Return fallback response on error
        return {
            "answer": f"Произошла ошибка при обработке вопроса: {str(e)}. Обратитесь к менеджеру сообщества.",
            "sources": [],
            "confidence": 0.0
        }

if __name__ == "__main__":
    # Test function
    try:
        result = answer_question(
            community_id=1,
            role="resident",
            question="Можно ли оставить лодку на улице на выходные?",
            k=3
        )
        
        print("Question answered successfully!")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print(f"Confidence: {result['confidence']:.3f}")
        
    except Exception as e:
        print(f"Test failed: {e}")
