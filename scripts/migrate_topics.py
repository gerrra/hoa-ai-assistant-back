#!/usr/bin/env python3
"""
Migration script to add topics table to existing database
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import psycopg
from pgvector.psycopg import register_vector

def get_db_connection():
    """Get database connection using environment variables"""
    env_path = Path(__file__).parent.parent / '.env'
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

def migrate_database():
    """Add topics table and update chunks table"""
    try:
        print("Starting database migration...")
        
        with get_db_connection() as conn:
            # Register vector type for pgvector
            register_vector(conn)
            
            with conn.cursor() as cur:
                # Check if topics table already exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'topics'
                    );
                """)
                
                topics_exists = cur.fetchone()[0]
                
                if topics_exists:
                    print("Topics table already exists, skipping creation")
                else:
                    print("Creating topics table...")
                    # Create topics table
                    cur.execute("""
                        CREATE TABLE topics (
                          id BIGSERIAL PRIMARY KEY,
                          document_id INT REFERENCES documents(id) ON DELETE CASCADE,
                          title TEXT NOT NULL,
                          description TEXT,
                          content TEXT NOT NULL,
                          embedding VECTOR(1536),
                          token_count INT,
                          page_numbers TEXT,
                          visibility TEXT DEFAULT 'resident',
                          created_at TIMESTAMPTZ DEFAULT now()
                        );
                    """)
                    print("Topics table created successfully")
                
                # Check if topic_id column exists in chunks table
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_name = 'chunks' AND column_name = 'topic_id'
                    );
                """)
                
                topic_id_exists = cur.fetchone()[0]
                
                if topic_id_exists:
                    print("topic_id column already exists in chunks table")
                else:
                    print("Adding topic_id column to chunks table...")
                    # Add topic_id column to chunks table
                    cur.execute("""
                        ALTER TABLE chunks 
                        ADD COLUMN topic_id INT REFERENCES topics(id) ON DELETE SET NULL;
                    """)
                    print("topic_id column added to chunks table")
                
                # Create indexes
                print("Creating indexes...")
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_topics_document ON topics(document_id);
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_topic ON chunks(topic_id);
                """)
                print("Indexes created successfully")
                
                # Commit all changes
                conn.commit()
                print("Migration completed successfully!")
                
    except Exception as e:
        print(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    try:
        migrate_database()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
