#!/usr/bin/env python3
"""
Document ingestion script for HOA AI Assistant
"""

import argparse
import sys
import tiktoken
from pathlib import Path

# Add backend to Python path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from ingest.save import save_document

def split_text(text: str, max_tokens: int = 1000, overlap: int = 100) -> list[str]:
    """
    Split text into chunks by token count with overlap
    
    Args:
        text: Input text to split
        max_tokens: Maximum tokens per chunk
        overlap: Number of overlapping tokens between chunks
        
    Returns:
        List of text chunks
    """
    # Get tiktoken encoding
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # Encode text to tokens
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        # Get chunk tokens
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        
        # Decode chunk back to text
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        # Move start position with overlap
        start = end - overlap
        if start >= len(tokens):
            break
    
    return chunks

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Ingest a document into HOA AI Assistant database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest_file.py --community 1 --title "CC&R 2024" --doc_type "CC&R" --path "data/ccr.pdf"
  python ingest_file.py --community 1 --title "Pool Rules" --doc_type "Rules" --path "data/pool_rules.txt" --visibility "resident"
        """
    )
    
    parser.add_argument(
        "--community",
        type=int,
        required=True,
        help="Community ID (integer)"
    )
    
    parser.add_argument(
        "--title",
        type=str,
        required=True,
        help="Document title"
    )
    
    parser.add_argument(
        "--doc_type",
        type=str,
        required=True,
        choices=["CC&R", "Bylaws", "Rules", "Policy", "Guidelines"],
        help="Document type"
    )
    
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the document file"
    )
    
    parser.add_argument(
        "--visibility",
        type=str,
        default="resident",
        choices=["resident", "board", "staff"],
        help="Document visibility (default: resident)"
    )
    
    args = parser.parse_args()
    
    # Validate file path
    file_path = Path(args.path)
    if not file_path.exists():
        print(f"Error: File not found: {args.path}")
        sys.exit(1)
    
    if not file_path.is_file():
        print(f"Error: Path is not a file: {args.path}")
        sys.exit(1)
    
    # Check file extension
    if file_path.suffix.lower() not in ['.txt', '.pdf']:
        print(f"Error: Unsupported file type: {file_path.suffix}")
        print("Supported types: .txt, .pdf")
        sys.exit(1)
    
    try:
        # Ingest document
        result = save_document(
            community_id=args.community,
            title=args.title,
            doc_type=args.doc_type,
            file_path=str(file_path),
            visibility=args.visibility
        )
        
        # Print success report with chunk information
        print(f"OK: document_id={result['document_id']}, chunks={result['chunks_inserted']}, title=\"{result['title']}\"")
        print(f"Document processed successfully with {result['chunks_inserted']} chunks")
        
    except FileNotFoundError:
        print(f"Error: File not found or no access: {args.path}")
        sys.exit(1)
    except PermissionError:
        print(f"Error: No permission to read file: {args.path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error ingesting document: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Test split_text function
    test_text = "This is a test text that will be split into chunks. " * 100
    chunks = split_text(test_text, max_tokens=100, overlap=20)
    print(f"Test split_text: {len(chunks)} chunks created")
    
    # Run main function
    main()
