#!/usr/bin/env python3
"""
Database initialization script for HOA AI Assistant
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import psycopg

def load_env():
    """Load environment variables from .env file"""
    env_path = Path('/app/.env')
    load_dotenv(env_path)
    
    # First try to get DATABASE_URL
    db_url = os.getenv('DATABASE_URL')
    if db_url:
        return db_url
    
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
        
        print(f"Error: Missing required database environment variables: {', '.join(missing)}")
        print("Please set either DATABASE_URL or all individual DB_* variables")
        sys.exit(1)
    
    # Build connection string
    db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    return db_url

def read_schema():
    """Read schema.sql file"""
    schema_path = Path(__file__).parent / 'schema.sql'
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Schema file not found at {schema_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading schema file: {e}")
        sys.exit(1)

def init_database():
    """Initialize database with schema"""
    try:
        # Load environment and read schema
        db_url = load_env()
        schema_sql = read_schema()
        
        print("Connecting to database...")
        
        # Connect to database and execute schema
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                print("Executing schema...")
                cur.execute(schema_sql)
                conn.commit()
                print("DB schema applied")
                
    except psycopg.Error as e:
        print(f"Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    init_database()
