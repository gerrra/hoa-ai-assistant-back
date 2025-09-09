#!/usr/bin/env python3
"""
Validation middleware for HOA AI Assistant
"""

import os
import psycopg
from typing import Dict, Any, Tuple
from pathlib import Path
from dotenv import load_dotenv

def get_db_connection():
    """Get database connection using environment variables"""
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

class RequestValidator:
    """Validator for API requests"""
    
    @staticmethod
    def validate_community_id(community_id: int, request_id: str) -> Tuple[bool, str]:
        """Validate that community exists"""
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM communities WHERE id = %s", (community_id,))
                    result = cur.fetchone()
                    
            if result:
                return True, f"Community {community_id} exists"
            else:
                return False, f"Community {community_id} not found"
        except Exception as e:
            return False, f"Database error: {str(e)}"
    
    @staticmethod
    def validate_question(question: str, request_id: str) -> Tuple[bool, str]:
        """Validate question format"""
        if not question or not question.strip():
            return False, "Question cannot be empty"
        
        if len(question.strip()) < 3:
            return False, "Question too short (minimum 3 characters)"
        
        if len(question) > 1000:
            return False, "Question too long (maximum 1000 characters)"
        
        return True, "Question format is valid"
    
    @staticmethod
    def validate_role(role: str, request_id: str) -> Tuple[bool, str]:
        """Validate user role"""
        valid_roles = ['resident', 'board', 'staff']
        if role not in valid_roles:
            return False, f"Invalid role '{role}'. Must be one of: {', '.join(valid_roles)}"
        
        return True, f"Role '{role}' is valid"
    
    @staticmethod
    def check_documents_availability(community_id: int, request_id: str) -> Tuple[bool, str, int]:
        """Check if community has documents"""
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Check documents table
                    cur.execute("SELECT COUNT(*) FROM documents WHERE community_id = %s", (community_id,))
                    doc_count = cur.fetchone()[0]
                    
                    # Check chunks table
                    cur.execute("""
                        SELECT COUNT(*) FROM chunks c
                        JOIN documents d ON c.document_id = d.id
                        WHERE d.community_id = %s
                    """, (community_id,))
                    chunk_count = cur.fetchone()[0]
                    
                    # Check topics table
                    cur.execute("""
                        SELECT COUNT(*) FROM topics t
                        JOIN documents d ON t.document_id = d.id
                        WHERE d.community_id = %s
                    """, (community_id,))
                    topic_count = cur.fetchone()[0]
            
            total_sources = doc_count + chunk_count + topic_count
            
            if total_sources == 0:
                return False, f"No documents, chunks, or topics found for community {community_id}", 0
            
            return True, f"Found {doc_count} documents, {chunk_count} chunks, {topic_count} topics", total_sources
            
        except Exception as e:
            return False, f"Database error checking documents: {str(e)}", 0
    
    @staticmethod
    def validate_openai_config(request_id: str) -> Tuple[bool, str]:
        """Validate OpenAI configuration"""
        try:
            from openai import OpenAI
            from pathlib import Path
            from dotenv import load_dotenv
            
            # Load environment variables
            env_path = Path('/app/.env')
            load_dotenv(env_path)
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return False, "OPENAI_API_KEY not found in environment variables"
            
            if len(api_key) < 10:
                return False, "OPENAI_API_KEY appears to be invalid (too short)"
            
            # Test API key by creating client
            client = OpenAI(api_key=api_key)
            
            return True, "OpenAI configuration is valid"
            
        except Exception as e:
            return False, f"OpenAI configuration error: {str(e)}"
