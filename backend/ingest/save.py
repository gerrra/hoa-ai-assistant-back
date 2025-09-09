#!/usr/bin/env python3
"""
Document saving module for HOA AI Assistant
"""

import os
import sys
import json
from typing import Dict, List
from pathlib import Path
from dotenv import load_dotenv
import psycopg
from pgvector.psycopg import register_vector

# Import local modules
from ingest.ingest import extract_text, clean_text, chunk_text
from ingest.embed import get_embedding
from ingest.topic_analyzer import analyze_document_topics

# Note: _exec is not imported here to avoid circular imports
# Topics are saved in the upload endpoint, not here

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

def save_document(community_id: int, title: str, doc_type: str, file_path: str, visibility: str = "resident", 
                 use_topic_analysis: bool = True) -> Dict:
    """
    Save document with topic analysis to database
    
    Args:
        community_id: ID of the community
        title: Document title
        doc_type: Type of document (CC&R, Bylaws, Rules, etc.)
        file_path: Path to the document file
        visibility: Document visibility (resident, board, staff)
        use_topic_analysis: Whether to use LLM-based topic analysis
        
    Returns:
        Dictionary with document_id, topics_inserted, chunks_inserted, and title
    """
    try:
        print(f"Processing document: {title}")
        print(f"File: {file_path}")
        print(f"Using topic analysis: {use_topic_analysis}")
        
        # Step 1: Connect to database
        print("Connecting to database...")
        with get_db_connection() as conn:
            # Register vector type for pgvector
            register_vector(conn)
            
            with conn.cursor() as cur:
                # Step 2: Insert document record
                print("Inserting document record...")
                cur.execute("""
                    INSERT INTO documents (community_id, title, doc_type, file_path, visibility)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (community_id, title, doc_type, file_path, visibility))
                
                document_id = cur.fetchone()[0]
                print(f"Document inserted with ID: {document_id}")
                
                topics_inserted = 0
                chunks_inserted = 0
                
                if use_topic_analysis:
                    # Step 3: Analyze document and extract topics using LLM
                    print("Analyzing document topics with LLM...")
                    topics = analyze_document_topics(file_path, doc_type)
                    print(f"Found {len(topics)} topics")
                    
                    # Step 4: Insert topics
                    for i, topic in enumerate(topics):
                        print(f"Processing topic {i+1}/{len(topics)}: {topic['title']}")
                        
                        # Get embedding for topic
                        topic_embedding = get_embedding(topic['content'])
                        if not topic_embedding:
                            print(f"Warning: Empty embedding for topic '{topic['title']}', skipping...")
                            continue
                        
                        # Insert topic
                        cur.execute("""
                            INSERT INTO topics (document_id, title, description, content, embedding, 
                                             token_count, page_numbers, visibility)
                            VALUES (%s, %s, %s, %s, %s::vector, %s, %s, %s)
                            RETURNING id
                        """, (
                            document_id,
                            topic['title'],
                            topic['description'],
                            topic['content'],
                            topic_embedding,
                            count_tokens(topic['content']),
                            topic['page_numbers'],  # Pass array directly, not JSON string
                            visibility
                        ))
                        
                        topic_id = cur.fetchone()[0]
                        topics_inserted += 1
                        print(f"Topic '{topic['title']}' inserted with ID: {topic_id}")
                        
                        # Step 5: Create chunks from topic content (for backward compatibility)
                        # Split topic content into smaller chunks if needed
                        topic_chunks = create_chunks_from_topic(topic['content'], topic_id, topic['page_numbers'])
                        
                        for chunk in topic_chunks:
                            chunk_embedding = get_embedding(chunk['text'])
                            if not chunk_embedding:
                                continue
                            
                            cur.execute("""
                                INSERT INTO chunks (document_id, topic_id, section_ref, text, embedding, 
                                                 token_count, visibility)
                                VALUES (%s, %s, %s, %s, %s::vector, %s, %s)
                            """, (
                                document_id,
                                topic_id,
                                chunk['section_ref'],
                                chunk['text'],
                                chunk_embedding,
                                chunk['token_count'],
                                visibility
                            ))
                            
                            chunks_inserted += 1
                
                else:
                    # Fallback: Use old chunking system
                    print("Using legacy chunking system...")
                    raw_text = extract_text(file_path)
                    cleaned_text = clean_text(raw_text)
                    
                    chunks = chunk_text(cleaned_text, max_tokens=600, overlap=80)
                    
                    for chunk in chunks:
                        chunk_embedding = get_embedding(chunk['text'])
                        if not chunk_embedding:
                            continue
                        
                        cur.execute("""
                            INSERT INTO chunks (document_id, section_ref, text, embedding, token_count, visibility)
                            VALUES (%s, %s, %s, %s::vector, %s, %s)
                        """, (
                            document_id,
                            chunk['section_ref'],
                            chunk['text'],
                            chunk_embedding,
                            chunk['token_count'],
                            visibility
                        ))
                        
                        chunks_inserted += 1
                
                # Commit all changes
                conn.commit()
                print(f"All changes committed successfully")
                
                return {
                    "document_id": document_id,
                    "topics_inserted": topics_inserted,
                    "chunks_inserted": chunks_inserted,
                    "title": title
                }
                
    except Exception as e:
        print(f"Error saving document: {e}")
        raise

def create_chunks_from_topic(topic_content: str, topic_id: int, page_numbers: List[int]) -> List[Dict]:
    """
    Create smaller chunks from topic content for backward compatibility
    
    Args:
        topic_content: Full content of the topic
        topic_id: ID of the topic
        page_numbers: List of page numbers where this topic appears
        
    Returns:
        List of chunk dictionaries
    """
    import tiktoken
    
    # Use smaller chunk size for topics
    max_tokens = 800
    overlap = 100
    
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(topic_content)
    
    chunks = []
    i = 0
    chunk_idx = 0
    
    while i < len(tokens):
        # Get chunk tokens
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        
        # Create section reference
        min_page = min(page_numbers) if page_numbers else 1
        section_ref = f"topic_{topic_id}_chunk_{chunk_idx + 1}_p{min_page}"
        
        chunks.append({
            'text': chunk_text.strip(),
            'section_ref': section_ref,
            'token_count': len(chunk_tokens)
        })
        
        chunk_idx += 1
        i += max_tokens - overlap if max_tokens > overlap else max_tokens
    
    return chunks

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

if __name__ == "__main__":
    # Test function
    try:
        result = save_document(
            community_id=1,
            title="Test Document",
            doc_type="Rules",
            file_path="data/test.txt",
            visibility="resident"
        )
        print(f"Success: {result}")
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
