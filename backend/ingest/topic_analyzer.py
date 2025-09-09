#!/usr/bin/env python3
"""
Topic analysis module for HOA AI Assistant
Анализирует документы и выделяет тематические разделы (топики) с помощью LLM
"""

import os
import json
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

def get_openai_client() -> OpenAI:
    """Get OpenAI client with API key from environment"""
    env_path = Path('/app/.env')
    load_dotenv(env_path)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    return OpenAI(api_key=api_key)

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def extract_text_from_pdf(file_path: str) -> List[Tuple[int, str]]:
    """
    Extract text from PDF file, returning list of (page_number, page_text)
    """
    from pypdf import PdfReader
    import io
    
    with open(file_path, 'rb') as f:
        data = f.read()
    
    reader = PdfReader(io.BytesIO(data))
    pages = []
    
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
            # Clean text
            text = re.sub(r'\s+', ' ', text).strip()
            if text and len(text) > 50:  # Skip empty or very short pages
                pages.append((i, text))
        except Exception as e:
            print(f"Error extracting page {i}: {e}")
            continue
    
    return pages

def extract_text_from_txt(file_path: str) -> List[Tuple[int, str]]:
    """
    Extract text from TXT file, returning list of (page_number, page_text)
    For TXT files, we'll treat the whole file as one "page"
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
    
    if text and len(text) > 50:
        return [(1, text)]
    return []

def analyze_document_topics(file_path: str, doc_type: str = "CC&R") -> List[Dict]:
    """
    Analyze document and extract topics using LLM
    
    Args:
        file_path: Path to the document file
        doc_type: Type of document (CC&R, Bylaws, Rules, etc.)
        
    Returns:
        List of topic dictionaries with title, description, content, and page_numbers
    """
    print(f"Analyzing document: {file_path}")
    
    # Extract text from file
    if file_path.lower().endswith('.pdf'):
        pages = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.txt'):
        pages = extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    if not pages:
        print("No text extracted from document")
        return []
    
    print(f"Extracted {len(pages)} pages of text")
    
    # Combine all text for analysis
    full_text = "\n\n".join([f"Страница {page_num}:\n{text}" for page_num, text in pages])
    
    # Check if text is too long for single analysis
    max_tokens = 100000  # Leave room for response
    if count_tokens(full_text) > max_tokens:
        print(f"Document too long ({count_tokens(full_text)} tokens), using chunked analysis")
        return analyze_document_topics_chunked(pages, doc_type)
    
    return analyze_document_topics_single(full_text, pages, doc_type)

def analyze_document_topics_single(full_text: str, pages: List[Tuple[int, str]], doc_type: str) -> List[Dict]:
    """Analyze document in single pass"""
    client = get_openai_client()
    
    system_prompt = f"""Ты эксперт по анализу документов ТСЖ/ЖКХ. Твоя задача - проанализировать документ типа "{doc_type}" и выделить основные тематические разделы (топики).

Для каждого топика определи:
1. Название топика (краткое, 2-4 слова)
2. Описание топика (1-2 предложения о чем этот раздел)
3. Весь релевантный контент из документа по этой теме
4. Номера страниц, где упоминается эта тема

Примеры топиков для документов ТСЖ:
- "Парковка" - правила парковки, штрафы, места
- "Платежи" - размеры взносов, сроки оплаты, пени
- "Общие собрания" - порядок проведения, кворум, голосование
- "Управление" - полномочия правления, выборы
- "Содержание" - ремонт, уборка, коммунальные услуги
- "Правила проживания" - шум, домашние животные, перепланировка

Верни результат в формате JSON:
{{
  "topics": [
    {{
      "title": "Название топика",
      "description": "Описание топика",
      "content": "Весь релевантный контент по теме",
      "page_numbers": [1, 2, 3]
    }}
  ]
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Проанализируй этот документ и выдели топики:\n\n{full_text}"}
            ],
            temperature=0.3,
            max_tokens=8000
        )
        
        content = response.choices[0].message.content.strip()
        print(f"LLM response length: {len(content)} characters")
        
        # Parse JSON response
        try:
            # Extract JSON from response (in case there's extra text)
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                result = json.loads(json_str)
                
                topics = result.get('topics', [])
                print(f"Extracted {len(topics)} topics from document")
                
                # Validate and clean topics
                validated_topics = []
                for topic in topics:
                    if validate_topic(topic):
                        validated_topics.append(topic)
                    else:
                        print(f"Skipping invalid topic: {topic}")
                
                return validated_topics
            else:
                print("No valid JSON found in response")
                return []
                
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response content: {content[:500]}...")
            return []
            
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return []

def analyze_document_topics_chunked(pages: List[Tuple[int, str]], doc_type: str) -> List[Dict]:
    """Analyze document in chunks for very long documents"""
    client = get_openai_client()
    
    # Split pages into chunks of reasonable size
    chunk_size = 20  # pages per chunk
    chunks = [pages[i:i+chunk_size] for i in range(0, len(pages), chunk_size)]
    
    all_topics = []
    
    for chunk_idx, chunk_pages in enumerate(chunks):
        print(f"Processing chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk_pages)} pages)")
        
        chunk_text = "\n\n".join([f"Страница {page_num}:\n{text}" for page_num, text in chunk_pages])
        
        system_prompt = f"""Ты эксперт по анализу документов ТСЖ/ЖКХ. Проанализируй эту часть документа типа "{doc_type}" и выдели основные тематические разделы.

Верни результат в формате JSON:
{{
  "topics": [
    {{
      "title": "Название топика",
      "description": "Описание топика", 
      "content": "Релевантный контент из этой части документа",
      "page_numbers": [номера страниц из этой части]
    }}
  ]
}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Проанализируй эту часть документа:\n\n{chunk_text}"}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            
            content = response.choices[0].message.content.strip()
            
            try:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    result = json.loads(json_str)
                    
                    topics = result.get('topics', [])
                    for topic in topics:
                        if validate_topic(topic):
                            all_topics.append(topic)
                            
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON for chunk {chunk_idx + 1}: {e}")
                continue
                
        except Exception as e:
            print(f"Error processing chunk {chunk_idx + 1}: {e}")
            continue
    
    # Merge similar topics from different chunks
    merged_topics = merge_similar_topics(all_topics)
    print(f"Merged {len(all_topics)} topics into {len(merged_topics)} final topics")
    
    return merged_topics

def validate_topic(topic: Dict) -> bool:
    """Validate topic structure and content"""
    required_fields = ['title', 'description', 'content', 'page_numbers']
    
    for field in required_fields:
        if field not in topic:
            return False
    
    if not topic['title'] or not topic['content']:
        return False
    
    if not isinstance(topic['page_numbers'], list):
        return False
    
    if len(topic['content'].strip()) < 50:  # Too short content
        return False
    
    return True

def merge_similar_topics(topics: List[Dict]) -> List[Dict]:
    """Merge topics with similar titles"""
    if not topics:
        return []
    
    # Group by title similarity (simple approach)
    merged = {}
    
    for topic in topics:
        title = topic['title'].lower().strip()
        
        # Check if similar topic already exists
        found_similar = False
        for existing_title in merged:
            if are_titles_similar(title, existing_title):
                # Merge content
                merged[existing_title]['content'] += f"\n\n{topic['content']}"
                merged[existing_title]['page_numbers'].extend(topic['page_numbers'])
                merged[existing_title]['page_numbers'] = sorted(list(set(merged[existing_title]['page_numbers'])))
                found_similar = True
                break
        
        if not found_similar:
            merged[title] = topic.copy()
    
    return list(merged.values())

def are_titles_similar(title1: str, title2: str) -> bool:
    """Check if two topic titles are similar"""
    # Simple similarity check - can be improved
    words1 = set(title1.split())
    words2 = set(title2.split())
    
    if not words1 or not words2:
        return False
    
    # If more than 50% of words match, consider similar
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    similarity = len(intersection) / len(union)
    return similarity > 0.5

if __name__ == "__main__":
    # Test function
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python topic_analyzer.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        topics = analyze_document_topics(file_path)
        print(f"\nFound {len(topics)} topics:")
        
        for i, topic in enumerate(topics, 1):
            print(f"\n{i}. {topic['title']}")
            print(f"   Description: {topic['description']}")
            print(f"   Pages: {topic['page_numbers']}")
            print(f"   Content length: {len(topic['content'])} characters")
            print(f"   Content preview: {topic['content'][:200]}...")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
