#!/usr/bin/env python3
"""
Document saving module for HOA AI Assistant
"""

import os
import sys
from typing import Dict, List
from pathlib import Path
from dotenv import load_dotenv
import psycopg
from pgvector.psycopg import register_vector

# Import local modules
from .ingest import extract_text, clean_text, chunk_text
from .embed import get_embedding

def get_db_connection():
    """Get database connection using environment variables"""
    env_path = Path(__file__).parent.parent.parent / '.env'
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

def save_document(community_id: int, title: str, doc_type: str, file_path: str, visibility: str = "resident") -> Dict:
    """
    Save document and its chunks to database
    
    Args:
        community_id: ID of the community
        title: Document title
        doc_type: Type of document (CC&R, Bylaws, Rules, etc.)
        file_path: Path to the document file
        visibility: Document visibility (resident, board, staff)
        
    Returns:
        Dictionary with document_id, chunks_inserted, and title
    """
    try:
        print(f"Processing document: {title}")
        print(f"File: {file_path}")
        
        # Step 1: Extract and clean text
        print("Extracting text from file...")
        raw_text = extract_text(file_path)
        cleaned_text = clean_text(raw_text)
        print(f"Text extracted: {len(cleaned_text)} characters")
        
        # Step 2: Split into chunks
        print("Splitting text into chunks...")
        chunks = chunk_text(cleaned_text, max_tokens=600, overlap=80)
        print(f"Created {len(chunks)} chunks")
        
        # Step 3: Connect to database
        print("Connecting to database...")
        with get_db_connection() as conn:
            # Register vector type for pgvector
            register_vector(conn)
            
            with conn.cursor() as cur:
                # Step 4: Insert document record
                print("Inserting document record...")
                cur.execute("""
                    INSERT INTO documents (community_id, title, doc_type, file_path, visibility)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (community_id, title, doc_type, file_path, visibility))
                
                document_id = cur.fetchone()[0]
                print(f"Document inserted with ID: {document_id}")
                
                # Step 5: Insert chunks with embeddings
                chunks_inserted = 0
                for i, chunk in enumerate(chunks):
                    print(f"Processing chunk {i+1}/{len(chunks)}: {chunk['section_ref']}")
                    
                    # Get embedding
                    embedding = get_embedding(chunk['text'])
                    if not embedding:
                        print(f"Warning: Empty embedding for chunk {chunk['section_ref']}, skipping...")
                        continue
                    
                    # Count tokens
                    token_count = len(chunk['text'].split())  # Simple token estimation
                    
                    # Insert chunk
                    cur.execute("""
                        INSERT INTO chunks (document_id, section_ref, text, embedding, token_count, visibility)
                        VALUES (%s, %s, %s, %s::vector, %s, %s)
                    """, (
                        document_id,
                        chunk['section_ref'],
                        chunk['text'],
                        embedding,
                        token_count,
                        visibility
                    ))
                    
                    chunks_inserted += 1
                    print(f"Chunk {chunk['section_ref']} inserted successfully")
                
                # Commit all changes
                conn.commit()
                print(f"All changes committed successfully")
                
                return {
                    "document_id": document_id,
                    "chunks_inserted": chunks_inserted,
                    "title": title
                }
                
    except Exception as e:
        print(f"Error saving document: {e}")
        raise

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
