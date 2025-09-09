#!/usr/bin/env python3
"""
LLM answer generation module for HOA AI Assistant
"""

import os
import re
from typing import Dict, List
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Import local modules
from search.retrieve import search_topics, format_context_topics, search_chunks, format_context

def _strip_inline_citations(text: str) -> str:
    """Remove inline citations and source references from answer text"""
    # remove trailing "источники: ..." block
    t = re.sub(r'(?is)\n*источники\s*:\s*.*$', '', text).strip()
    # remove lone trailing [1], [2] ... if present at very end
    t = re.sub(r'\s*\[\d+(?:,\s*\d+)*\]$', '', t).strip()
    return t

def get_openai_client():
    """Get OpenAI client with API key from environment"""
    env_path = Path('/app/.env')
    load_dotenv(env_path)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    return OpenAI(api_key=api_key)

def answer_question(community_id: int, role: str, question: str, k: int = 6, use_topics: bool = True) -> Dict:
    """
    Generate answer to user question using LLM and document search
    
    Args:
        community_id: ID of the community
        role: User role (resident, board, staff)
        question: User question
        k: Number of topics/chunks to retrieve for context
        use_topics: Whether to use topic-based search (new system) or chunk-based (legacy)
        
    Returns:
        Dictionary with answer, sources, and confidence
    """
    try:
        # Step 1: Search for relevant content
        print(f"[DEBUG] answer_question: use_topics={use_topics}, community_id={community_id}, question='{question}'")
        if use_topics:
            # Try topic-based search first
            topics = search_topics(community_id, question, k=k)
            print(f"[DEBUG] Found {len(topics)} topics")
            
            if topics:
                # Use topics for context
                context = format_context_topics(topics)
                sources = []
                for topic in topics:
                    # Parse page numbers
                    try:
                        import json
                        page_numbers = json.loads(topic['page_numbers']) if topic['page_numbers'] else []
                        pages_str = f" (страницы {', '.join(map(str, page_numbers))})" if page_numbers else ""
                    except:
                        pages_str = ""
                    
                    source = {
                        "text": f"{topic['topic_title']} - {topic['document_title']}{pages_str}",
                        "document": {
                            "id": topic['document_id'],
                            "title": topic['document_title'],
                            "doc_type": topic['doc_type'],
                            "rel_path": topic['file_path']
                        }
                    }
                    print(f"[DEBUG] Created source: {source}")
                    sources.append(source)
                
                # Calculate average confidence
                avg_similarity = sum(topic["similarity"] for topic in topics) / len(topics)
                
                # Build meta information with effective dates
                effective_dates = set()
                for topic in topics:
                    if topic.get("effective_from"):
                        effective_dates.add(str(topic["effective_from"]))
                
                meta = {"effective_dates": sorted(list(effective_dates))}
                
            else:
                # Fallback to chunk-based search
                print("[DEBUG] No topics found, falling back to chunk search")
                chunks = search_chunks(community_id, question, k=k)
                print(f"[DEBUG] Found {len(chunks)} chunks")
                
                if not chunks:
                    return {
                        "answer": "Недостаточно данных в документах, обратитесь к менеджеру сообщества.",
                        "sources": [],
                        "confidence": 0.0
                    }
                
                context = format_context(chunks)
                sources = []
                for chunk in chunks:
                    sources.append({
                        "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                        "document": {
                            "id": chunk["document_id"],
                            "title": chunk["document_title"],
                            "doc_type": chunk["doc_type"],
                            "rel_path": chunk["file_path"]
                        }
                    })
                
                avg_similarity = sum(chunk["similarity"] for chunk in chunks) / len(chunks)
                
                # Build meta information with effective dates
                effective_dates = set()
                for chunk in chunks:
                    if chunk.get("effective_from"):
                        effective_dates.add(str(chunk["effective_from"]))
                
                meta = {"effective_dates": sorted(list(effective_dates))}
        else:
            # Use legacy chunk-based search
            chunks = search_chunks(community_id, question, k=k)
            
            if not chunks:
                return {
                    "answer": "Недостаточно данных в документах, обратитесь к менеджеру сообщества.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            context = format_context(chunks)
            sources = []
            for chunk in chunks:
                sources.append({
                    "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    "document": {
                        "id": chunk["document_id"],
                        "title": chunk["document_title"],
                        "doc_type": chunk["doc_type"],
                        "rel_path": chunk["file_path"]
                    }
                })
            
            avg_similarity = sum(chunk["similarity"] for chunk in chunks) / len(chunks)
            
            # Build meta information with effective dates
            effective_dates = set()
            for chunk in chunks:
                if chunk.get("effective_from"):
                    effective_dates.add(str(chunk["effective_from"]))
            
            meta = {"effective_dates": sorted(list(effective_dates))}
        
        # Step 2: Build system prompt
        system_prompt = """Ты — ассистент HOA. Отвечай строго на основе ПРИЛОЖЕННЫХ ФРАГМЕНТОВ. Если информации недостаточно — скажи обратиться к менеджеру. 

Пиши живыми связными абзацами. Не добавляй цитаты-метки вида [1], (ref) и раздел «Источники» в тексте ответа. Источники возвращает сервер отдельно."""
        
        # Step 3: Build user prompt
        user_prompt = f"{question}\n\nCONTEXT:\n{context}"
        
        # Step 4: Call OpenAI Chat Completions
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Slightly higher for more natural responses
            max_tokens=1000
        )
        
        raw_answer = response.choices[0].message.content.strip()
        answer = _strip_inline_citations(raw_answer)
        
        # Step 5: Prepend effective dates to answer if available
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
