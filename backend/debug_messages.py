#!/usr/bin/env python3
"""
Debug script to check what's happening with message saving
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import uuid

def debug_messages():
    """Debug message saving"""
    
    # Load environment
    env_path = Path('..') / '.env'
    load_dotenv(env_path)
    
    try:
        from chat import _exec
        
        print("✅ Successfully imported _exec from chat module")
        
        # Create a test conversation
        conversation_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        
        print(f"Creating conversation: {conversation_id}")
        
        # Create session
        _exec("""
            INSERT INTO sessions(id, user_agent, ip) 
            VALUES(%s, %s, %s)
        """, (
            uuid.UUID(session_id),
            "test",
            "127.0.0.1"
        ))
        
        # Create conversation
        _exec("""
            INSERT INTO conversations(id, session_id, title) 
            VALUES(%s, %s, %s)
        """, (
            uuid.UUID(conversation_id),
            uuid.UUID(session_id),
            "Debug conversation"
        ))
        
        print("✅ Created session and conversation")
        
        # Test 1: Insert user message
        print("\n=== TEST 1: Insert user message ===")
        try:
            user_msg_id = _exec("""
                INSERT INTO messages(conversation_id, role, content, meta) 
                VALUES(%s, %s, %s, %s)
                RETURNING id
            """, (
                uuid.UUID(conversation_id),
                "user",
                "Test user message",
                '{"test": true, "community_id": 2}'
            ), fetch=True)
            
            if user_msg_id:
                print(f"✅ User message inserted with ID: {user_msg_id[0][0]}")
            else:
                print("❌ Failed to insert user message")
        except Exception as e:
            print(f"❌ Error inserting user message: {e}")
        
        # Test 2: Insert assistant message
        print("\n=== TEST 2: Insert assistant message ===")
        try:
            assistant_msg_id = _exec("""
                INSERT INTO messages(conversation_id, role, content, meta) 
                VALUES(%s, %s, %s, %s)
                RETURNING id
            """, (
                uuid.UUID(conversation_id),
                "assistant",
                "Test assistant message",
                '{"confidence": 0.0, "sources": []}'
            ), fetch=True)
            
            if assistant_msg_id:
                print(f"✅ Assistant message inserted with ID: {assistant_msg_id[0][0]}")
            else:
                print("❌ Failed to insert assistant message")
        except Exception as e:
            print(f"❌ Error inserting assistant message: {e}")
        
        # Test 3: Check messages
        print("\n=== TEST 3: Check messages ===")
        try:
            messages = _exec("""
                SELECT id, role, content, meta
                FROM messages 
                WHERE conversation_id = %s
                ORDER BY created_at ASC
            """, (uuid.UUID(conversation_id),), fetch=True)
            
            print(f"Found {len(messages)} messages:")
            for msg in messages:
                print(f"  ID: {msg[0]}, Role: {msg[1]}, Content: {msg[2][:50]}...")
                if msg[3]:
                    print(f"    Meta: {msg[3]}")
        except Exception as e:
            print(f"❌ Error checking messages: {e}")
        
        # Clean up
        print("\n=== CLEANUP ===")
        try:
            _exec("DELETE FROM messages WHERE conversation_id = %s", (uuid.UUID(conversation_id),))
            _exec("DELETE FROM conversations WHERE id = %s", (uuid.UUID(conversation_id),))
            _exec("DELETE FROM sessions WHERE id = %s", (uuid.UUID(session_id),))
            print("✅ Cleaned up test data")
        except Exception as e:
            print(f"❌ Error cleaning up: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_messages()
