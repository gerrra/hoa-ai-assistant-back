#!/usr/bin/env python3
"""
Document search and retrieval module for HOA AI Assistant
"""

import os
import json
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv
import psycopg
from openai import OpenAI

def get_query_embedding(text: str) -> List[float]:
    """
    Get embedding for search query using OpenAI API
    
    Args:
        text: Search query text
        
    Returns:
        List of floats (vector of length 1536)
    """
    # Load environment variables
    env_path = Path('/app/.env')
    load_dotenv(env_path)
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Check if text is empty
    if not text or not text.strip():
        return []
    
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
        print(f"Error getting query embedding: {e}")
        raise

def get_db_connection():
    """Get database connection using environment variables"""
    env_path = Path('/app/.env')
    load_dotenv(env_path)
    
    # First try to get DATABASE_URL
    db_url = os.getenv('DATABASE_URL')
    if db_url:
        return psycopg.connect(db_url)
    
    # If not found, build from individual components
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_pass = os.getenv('DB_PASS')
    
    # Check if all required components are present
    if not all([db_host, db_port, db_name, db_user, db_pass]):
        missing = []
        if not db_host: missing.append('DB_HOST')
        if not db_port: missing.append('DB_PORT')
        if not db_name: missing.append('DB_NAME')
        if not db_user: missing.append('DB_USER')
        if not db_pass: missing.append('DB_PASS')
        
        raise ValueError(f"Missing required database environment variables: {', '.join(missing)}")
    
    # Build connection string
    db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    return psycopg.connect(db_url)

def search_topics(community_id: int, query: str, k: int = 6, min_relevance: float = 0.0) -> List[Dict]:
    """
    Search for relevant document topics using vector similarity
    
    Args:
        community_id: ID of the community to search in
        query: Search query text
        k: Maximum number of results to return
        min_relevance: Minimum similarity score (0.0 to 1.0)
        
    Returns:
        List of dictionaries with topic information and similarity scores
    """
    try:
        # Step 1: Get query embedding
        query_embedding = get_query_embedding(query)
        if not query_embedding:
            print("Warning: Empty query embedding, returning empty results")
            return []
        
        # Step 2: Connect to database
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Step 3: Execute similarity search on topics
                cur.execute("""
                    SELECT
                      t.id AS topic_id,
                      t.title AS topic_title,
                      t.description AS topic_description,
                      t.content AS topic_content,
                      t.page_numbers,
                      d.id AS document_id,
                      d.title AS document_title,
                      d.doc_type,
                      d.effective_from,
                      d.file_path,
                      1 - (t.embedding <=> %s::vector) AS similarity
                    FROM topics t
                    JOIN documents d ON d.id = t.document_id
                    WHERE d.community_id = %s
                    ORDER BY t.embedding <=> %s::vector
                    LIMIT %s;
                """, (query_embedding, community_id, query_embedding, k))
                
                # Fetch results
                rows = cur.fetchall()
                
                # Convert to list of dictionaries
                results = []
                for row in rows:
                    topic_data = {
                        "topic_id": row[0],
                        "topic_title": row[1],
                        "topic_description": row[2],
                        "topic_content": row[3],
                        "page_numbers": row[4],  # JSON string
                        "document_id": row[5],
                        "document_title": row[6],
                        "doc_type": row[7],
                        "effective_from": row[8],
                        "file_path": row[9],
                        "similarity": float(row[10])
                    }
                    
                    # Step 4: Filter by minimum relevance if specified
                    if topic_data["similarity"] >= min_relevance:
                        results.append(topic_data)
                
                return results
                
    except Exception as e:
        print(f"Error searching topics: {e}")
        raise

def search_chunks(community_id: int, query: str, k: int = 6, min_relevance: float = 0.0) -> List[Dict]:
    """
    Search for relevant document chunks using vector similarity (legacy function)
    
    Args:
        community_id: ID of the community to search in
        query: Search query text
        k: Maximum number of results to return
        min_relevance: Minimum similarity score (0.0 to 1.0)
        
    Returns:
        List of dictionaries with chunk information and similarity scores
    """
    print(f"Searching chunks for community_id: {community_id}")
    print(f"Query: {query}")
    
    try:
        # Step 1: Get query embedding
        query_embedding = get_query_embedding(query)
        if not query_embedding:
            print("Warning: Empty query embedding, returning empty results")
            return []
        
        # Step 2: Connect to database
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Step 3: Execute similarity search
                cur.execute("""
                    SELECT
                      c.id AS chunk_id,
                      c.section_ref,
                      c.text,
                      d.id AS document_id,
                      d.title AS document_title,
                      d.doc_type,
                      d.effective_from,
                      d.file_path,
                      1 - (c.embedding <=> %s::vector) AS similarity
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE d.community_id = %s
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s;
                """, (query_embedding, community_id, query_embedding, k))
                
                # Fetch results
                rows = cur.fetchall()
                print(f"Found {len(rows)} chunks from database")
                
                # Convert to list of dictionaries
                results = []
                for row in rows:
                    chunk_data = {
                        "chunk_id": row[0],
                        "section_ref": row[1],
                        "text": row[2],
                        "document_id": row[3],
                        "document_title": row[4],
                        "doc_type": row[5],
                        "effective_from": row[6],
                        "file_path": row[7],
                        "similarity": float(row[8])
                    }
                    
                    print(f"  - Chunk: title='{chunk_data['document_title']}', file_path='{chunk_data['file_path']}', similarity={chunk_data['similarity']:.3f}")
                    
                    # Step 4: Filter by minimum relevance if specified
                    if chunk_data["similarity"] >= min_relevance:
                        results.append(chunk_data)
                
                print(f"Returning {len(results)} chunks after relevance filtering")
                return results
                
    except Exception as e:
        print(f"Error searching chunks: {e}")
        raise

def format_context_topics(topics: List[Dict]) -> str:
    """
    Format topic search results into readable context string
    
    Args:
        topics: List of topic dictionaries from search_topics
        
    Returns:
        Formatted context string
    """
    if not topics:
        return "No relevant topics found."
    
    context_parts = []
    for topic in topics:
        # Parse page numbers
        try:
            page_numbers = json.loads(topic['page_numbers']) if topic['page_numbers'] else []
            pages_str = f" (страницы {', '.join(map(str, page_numbers))})" if page_numbers else ""
        except:
            pages_str = ""
        
        # Create header: [ТОПИК: название | ДОКУМЕНТ: название]
        header = f"[ТОПИК: {topic['topic_title']} | ДОКУМЕНТ: {topic['document_title']}{pages_str}]"
        
        # Add topic description if available
        if topic.get('topic_description'):
            header += f"\nОписание: {topic['topic_description']}"
        
        # Add topic content
        topic_content = topic['topic_content'].strip()
        
        # Combine header and content
        context_parts.append(f"{header}\n{topic_content}")
    
    # Join all topics with double newlines
    return "\n\n".join(context_parts)

def format_context(chunks: List[Dict]) -> str:
    """
    Format search results into readable context string (legacy function)
    
    Args:
        chunks: List of chunk dictionaries from search_chunks
        
    Returns:
        Formatted context string
    """
    if not chunks:
        return "No relevant documents found."
    
    context_parts = []
    for chunk in chunks:
        # Create header: [DOC: title | §section_ref]
        header = f"[DOC: {chunk['document_title']} | {chunk['section_ref']}]"
        
        # Add chunk text
        chunk_text = chunk['text'].strip()
        
        # Combine header and text
        context_parts.append(f"{header}\n{chunk_text}")
    
    # Join all chunks with double newlines
    return "\n\n".join(context_parts)

if __name__ == "__main__":
    # Test function
    try:
        # Test topic search
        print("Testing topic search...")
        results = search_topics(
            community_id=1,
            query="парковка лодок",
            k=3,
            min_relevance=0.5
        )
        
        print(f"Found {len(results)} relevant topics:")
        for i, topic in enumerate(results, 1):
            print(f"\n{i}. {topic['topic_title']} - {topic['document_title']}")
            print(f"   Similarity: {topic['similarity']:.3f}")
            print(f"   Description: {topic['topic_description']}")
            print(f"   Content: {topic['topic_content'][:100]}...")
        
        # Test context formatting
        context = format_context_topics(results)
        print(f"\n\nFormatted context:\n{context}")
        
    except Exception as e:
        print(f"Test failed: {e}")
