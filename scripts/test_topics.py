#!/usr/bin/env python3
"""
Test script for the new topic-based system
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add backend to path
backend_path = Path(__file__).parent.parent / 'backend'
sys.path.insert(0, str(backend_path))

def test_topic_analyzer():
    """Test the topic analyzer module"""
    print("Testing topic analyzer...")
    
    try:
        from ingest.topic_analyzer import analyze_document_topics
        
        # Test with a sample file
        test_file = backend_path / 'data' / 'test.txt'
        if not test_file.exists():
            print(f"Test file not found: {test_file}")
            return False
        
        print(f"Analyzing file: {test_file}")
        topics = analyze_document_topics(str(test_file), "CC&R")
        
        print(f"Found {len(topics)} topics:")
        for i, topic in enumerate(topics, 1):
            print(f"\n{i}. {topic['title']}")
            print(f"   Description: {topic['description']}")
            print(f"   Pages: {topic['page_numbers']}")
            print(f"   Content length: {len(topic['content'])} characters")
            print(f"   Content preview: {topic['content'][:200]}...")
        
        return True
        
    except Exception as e:
        print(f"Error testing topic analyzer: {e}")
        return False

def test_search_system():
    """Test the search system"""
    print("\nTesting search system...")
    
    try:
        from search.retrieve import search_topics, format_context_topics
        
        # Test topic search
        topics = search_topics(community_id=1, query="парковка", k=3)
        print(f"Found {len(topics)} topics for query 'парковка':")
        
        for i, topic in enumerate(topics, 1):
            print(f"\n{i}. {topic['topic_title']} - {topic['document_title']}")
            print(f"   Similarity: {topic['similarity']:.3f}")
            print(f"   Description: {topic['topic_description']}")
        
        # Test context formatting
        if topics:
            context = format_context_topics(topics)
            print(f"\nFormatted context:\n{context[:500]}...")
        
        return True
        
    except Exception as e:
        print(f"Error testing search system: {e}")
        return False

def test_answer_system():
    """Test the answer generation system"""
    print("\nTesting answer system...")
    
    try:
        from llm.answer import answer_question
        
        # Test with topic-based search
        result = answer_question(
            community_id=1,
            role="resident",
            question="Можно ли оставить лодку на улице?",
            k=3,
            use_topics=True
        )
        
        print("Answer generated successfully!")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print(f"Confidence: {result['confidence']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Error testing answer system: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing new topic-based system...")
    print("=" * 50)
    
    # Load environment variables
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
    
    tests = [
        ("Topic Analyzer", test_topic_analyzer),
        ("Search System", test_search_system),
        ("Answer System", test_answer_system),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"Test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed!")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
