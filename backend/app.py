#!/usr/bin/env python3
"""
FastAPI application for HOA AI Assistant
"""

from fastapi import Cookie, Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from starlette.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

class CommunityCreateRequest(BaseModel):
    name: str

class CommunityUpdateRequest(BaseModel):
    name: str
from typing import List, Dict, Optional, Literal
import psycopg
import json
import secrets
import shutil
from pathlib import Path
import os
import re
from collections import defaultdict
from dotenv import load_dotenv
from datetime import datetime

# Import local modules
from llm.answer import answer_question, _strip_inline_citations
import chat as chat_mod
from middleware.logging import RequestLogger
from middleware.validation import RequestValidator

# Helper functions for database operations
def _dsn() -> str:
    """Get database connection string from environment"""
    url = os.getenv("DATABASE_URL")
    if url:
        return url.replace("postgresql+psycopg://", "postgresql://")
    
    # Use localhost for local development, host.docker.internal for Docker
    if os.path.exists('/.dockerenv'):
        # Running in Docker container
        host = os.getenv("DB_HOST", "host.docker.internal")
    else:
        # Running locally
        host = os.getenv("DB_HOST", "localhost")
    
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "hoa")
    user = os.getenv("DB_USER", "hoa")
    pwd = os.getenv("DB_PASS", "hoa")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{name}"

def _exec(sql: str, params: Optional[tuple] = None, fetch: bool = False, many: bool = False):
    """Execute SQL query with psycopg"""
    with psycopg.connect(_dsn(), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            if fetch:
                return cur.fetchall()
            if many:
                return cur.rowcount
            return None

def sse_stream(iterable):
    """Convert iterable to Server-Sent Events stream"""
    for item in iterable:
        yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

def _make_doc_url(request: Request, rel_path: str, page: int | None) -> str | None:
    """Create clickable URL for document with optional page anchor"""
    if not rel_path or not rel_path.strip():
        return None  # Return None for empty paths
    
    base = str(request.base_url).rstrip("/")
    # Use /data/ instead of /static/ for PDF files
    if rel_path.startswith('data/'):
        url = f"{base}/{rel_path.lstrip('/')}"
    else:
        url = f"{base}/data/{rel_path.lstrip('/')}"  # Changed to always use /data/
    
    if page:
        url += f"#page={int(page)}"
    
    return url

def group_sources(chunks: list[dict], request: Request) -> list[dict]:
    """Group chunks by document and create clickable sources"""
    print(f"Grouping sources from {len(chunks)} chunks")
    
    if not chunks:
        print("No chunks to group, returning empty sources")
        return []
    
    # expected chunk fields: file_path, document_title, section_ref
    acc = defaultdict(lambda: {"title": None, "rel_path": None, "hits": set()})
    
    for ch in chunks or []:
        # Extract page number from section_ref if it contains page info
        page = None
        section_ref = ch.get("section_ref", "")
        if "p" in section_ref.lower():
            # Try to extract page number from section_ref like "p62", "page 62", etc.
            page_match = re.search(r'p(?:age)?\s*(\d+)', section_ref.lower())
            if page_match:
                page = int(page_match.group(1))
        
        rel = ch.get("file_path") or ""
        title = ch.get("document_title") or os.path.basename(rel) or "document"
        key = rel or title
        
        print(f"  - Processing chunk: title='{title}', rel_path='{rel}', section_ref='{section_ref}', page={page}")
        
        acc[key]["title"] = title
        acc[key]["rel_path"] = rel
        if page:
            acc[key]["hits"].add(int(page))
    
    out = []
    for key, v in acc.items():
        hits_sorted = sorted(v["hits"]) if v["hits"] else []
        url = _make_doc_url(request, v["rel_path"], hits_sorted[0] if hits_sorted else None) if v["rel_path"] else None
        links = [_make_doc_url(request, v["rel_path"], p) for p in hits_sorted] if v["rel_path"] else []
        
        print(f"  - Grouped source: title='{v['title']}', url='{url}', pages={hits_sorted}")
        
        out.append({
            "title": v["title"],
            "url": url,
            "pages": hits_sorted,
            # optional: per-page deep links
            "links": links
        })
    
    # sort by title
    out.sort(key=lambda x: x["title"].lower())
    print(f"Returning {len(out)} grouped sources")
    return out

# Load environment variables
# Use different paths for local development vs Docker
if os.path.exists('/.dockerenv'):
    # Running in Docker container
    env_path = Path('/app/.env')
else:
    # Running locally
    env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Document registry SQL
DOCS_SQL = """
CREATE TABLE IF NOT EXISTS admin_documents (
  id UUID PRIMARY KEY,
  filename TEXT NOT NULL,
  title TEXT, -- название документа, указанное пользователем
  doc_type TEXT, -- тип документа (CC&R, Bylaws, Rules, Policy, Guidelines)
  visibility TEXT DEFAULT 'resident', -- видимость документа
  rel_path TEXT NOT NULL, -- relative path under /static
  pages INT NOT NULL DEFAULT 0,
  size_bytes BIGINT NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC')
);

-- Миграция: добавляем новые поля к существующей таблице
ALTER TABLE admin_documents ADD COLUMN IF NOT EXISTS title TEXT;
ALTER TABLE admin_documents ADD COLUMN IF NOT EXISTS doc_type TEXT;
ALTER TABLE admin_documents ADD COLUMN IF NOT EXISTS visibility TEXT DEFAULT 'resident';
CREATE TABLE IF NOT EXISTS admin_doc_chunks (
  id BIGSERIAL PRIMARY KEY,
  doc_id UUID NOT NULL REFERENCES admin_documents(id) ON DELETE CASCADE,
  page INT,
  start_pos INT,
  end_pos INT,
  text TEXT NOT NULL,
  section TEXT NULL,
  topic_id INT NULL,
  token_count INT NULL
);
CREATE INDEX IF NOT EXISTS idx_admin_doc_chunks_doc ON admin_doc_chunks(doc_id, page, id);

-- Topics table for topic segmentation
CREATE TABLE IF NOT EXISTS doc_topics (
  id BIGSERIAL PRIMARY KEY,
  doc_id UUID NOT NULL REFERENCES admin_documents(id) ON DELETE CASCADE,
  topic_index INT NOT NULL,
  title TEXT NULL,
  start_page INT NULL,
  end_page INT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC')
);
CREATE INDEX IF NOT EXISTS idx_doc_topics_doc ON doc_topics(doc_id, topic_index);
"""

# Additional schema updates (run after main tables exist)
SCHEMA_UPDATES_SQL = """
-- Create topics table if it doesn't exist
CREATE TABLE IF NOT EXISTS topics (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    page_numbers INTEGER[],
    created_at TIMESTAMPTZ DEFAULT now(),
    description TEXT,
    embedding vector(1536),
    token_count INTEGER,
    visibility TEXT DEFAULT 'resident',
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- Add missing columns to existing topics table
ALTER TABLE topics ADD COLUMN IF NOT EXISTS description TEXT;
ALTER TABLE topics ADD COLUMN IF NOT EXISTS embedding vector(1536);
ALTER TABLE topics ADD COLUMN IF NOT EXISTS token_count INTEGER;
ALTER TABLE topics ADD COLUMN IF NOT EXISTS visibility TEXT DEFAULT 'resident';

-- Add extra columns to existing doc_chunks table
ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS section TEXT NULL;
ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS topic_id INT NULL;
ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS token_count INT NULL;

-- Fix doc_topics table if it exists with wrong foreign key
DROP TABLE IF EXISTS doc_topics CASCADE;
CREATE TABLE doc_topics (
  id BIGSERIAL PRIMARY KEY,
  doc_id UUID NOT NULL REFERENCES admin_documents(id) ON DELETE CASCADE,
  topic_index INT NOT NULL,
  title TEXT NULL,
  start_page INT NULL,
  end_page INT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC')
);
CREATE INDEX IF NOT EXISTS idx_doc_topics_doc ON doc_topics(doc_id, topic_index);

-- Add missing columns to admin_doc_chunks
ALTER TABLE admin_doc_chunks ADD COLUMN IF NOT EXISTS section TEXT NULL;
ALTER TABLE admin_doc_chunks ADD COLUMN IF NOT EXISTS topic_id INT NULL;
ALTER TABLE admin_doc_chunks ADD COLUMN IF NOT EXISTS token_count INT NULL;
"""

# Admin password from environment
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "changeme")

# Cookie security settings
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"

def parse_origins(value: str) -> list[str]:
    """Parse CORS origins from environment variable (JSON array or comma-separated list)"""
    v = (value or "").strip()
    if v.startswith("["):
        try: 
            return json.loads(v)
        except Exception: 
            pass
    return [x.strip() for x in v.split(",") if x.strip()]

app = FastAPI(
    title="HOA AI Assistant",
    description="AI-powered assistant for Homeowners Association document queries",
    version="1.0.0"
)

# Disable redirect_slashes to prevent 307/308 redirects that break CORS preflight
app.router.redirect_slashes = False

# CORS configuration with robust origins parsing
origins = parse_origins(os.getenv("CORS_ORIGINS")) or [
    "https://admin.gerrra.com", "https://app.gerrra.com",
    "http://localhost:5173", "http://localhost:5174", "http://localhost:3000",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SID cookie middleware
from uuid import uuid4
from fastapi import Request

@app.middleware("http")
async def ensure_sid_cookie(request: Request, call_next):
    sid = request.cookies.get("sid")
    new_sid = None
    if not sid:
        new_sid = str(uuid4())
        request.state.sid = new_sid
    else:
        request.state.sid = sid
    resp = await call_next(request)
    if new_sid:
        resp.set_cookie(
            "sid", new_sid,
            httponly=True, samesite="lax",
            max_age=60*60*24*365,
        )
    return resp

# Mount static files and templates
from pathlib import Path
STATIC_DIR = (Path(__file__).parent / "static").resolve()
DATA_DIR = (Path(__file__).parent / "data").resolve()

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    print(f"[WARN] static dir not found: {STATIC_DIR}; skipping static mount")

if DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")
else:
    print(f"[WARN] data dir not found: {DATA_DIR}; skipping data mount")
templates = Jinja2Templates(directory="templates")

# Initialize chat module and include router
chat_mod.init_chat(app)
app.include_router(chat_mod.router, prefix="/chat", tags=["chat"])

# Initialize document registry
_exec(DOCS_SQL)

# Apply schema updates (run after main tables exist)
try:
    _exec(SCHEMA_UPDATES_SQL)
except Exception as e:
    print(f"[warn] Schema updates failed (may be expected): {e}")

# Pydantic models
class AskRequest(BaseModel):
    community_id: int = Field(..., gt=0, description="Community ID (must be positive)")
    role: Literal["resident", "board", "staff"] = Field(..., description="User role")
    question: str = Field(..., min_length=1, description="User question")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for chat history")

class DocumentInfo(BaseModel):
    id: int
    title: str
    doc_type: str
    rel_path: str

class Source(BaseModel):
    text: str
    document: Dict

class GroupedSource(BaseModel):
    title: str
    url: str | None
    pages: List[int]
    links: List[str]

class AskResponse(BaseModel):
    answer: str
    sources: List[Source]  # Keep for backward compatibility
    sources_grouped: List[GroupedSource]  # New grouped sources
    confidence: float

# Admin API models

class LogEntry(BaseModel):
    created_at: str
    user_role: str
    question: str
    confidence: float

class UploadResponse(BaseModel):
    document_id: int
    topics_inserted: int
    chunks_inserted: int

class LoginBody(BaseModel):
    password: str

class GenerateTopicTitleRequest(BaseModel):
    text: str

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

def is_admin(admin_auth: str | None = Cookie(default=None)):
    """Check if user is authenticated as admin"""
    if admin_auth == "1":
        return True
    raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest, http_request: Request):
    """
    Ask a question to HOA AI Assistant with detailed logging and error handling
    
    Args:
        request: AskRequest with community_id, role, question, and optional conversation_id
        http_request: HTTP request object for accessing session state
        
    Returns:
        AskResponse with answer, sources, confidence, and conversation_id
    """
    request_id = None
    
    try:
        # Start logging
        request_data = {
            "community_id": request.community_id,
            "role": request.role,
            "question": request.question[:100] + "..." if len(request.question) > 100 else request.question,
            "conversation_id": request.conversation_id
        }
        request_id = RequestLogger.log_request(request_data)
        
        # Step 1: Validate input data
        RequestLogger.log_validation(request_id, "question", request.question, True, "Basic validation passed")
        
        # Validate community exists
        community_valid, community_msg = RequestValidator.validate_community_id(request.community_id, request_id)
        RequestLogger.log_validation(request_id, "community_id", request.community_id, community_valid, community_msg)
        if not community_valid:
            raise HTTPException(status_code=404, detail=f"Community validation failed: {community_msg}")
        
        # Validate question format
        question_valid, question_msg = RequestValidator.validate_question(request.question, request_id)
        RequestLogger.log_validation(request_id, "question_format", request.question, question_valid, question_msg)
        if not question_valid:
            raise HTTPException(status_code=400, detail=f"Question validation failed: {question_msg}")
        
        # Validate role
        role_valid, role_msg = RequestValidator.validate_role(request.role, request_id)
        RequestLogger.log_validation(request_id, "role", request.role, role_valid, role_msg)
        if not role_valid:
            raise HTTPException(status_code=400, detail=f"Role validation failed: {role_msg}")
        
        # Step 2: Check database connection
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            RequestLogger.log_database_connection(request_id, True)
        except Exception as e:
            RequestLogger.log_database_connection(request_id, False, str(e))
            raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")
        
        # Step 3: Save user message to chat history if conversation_id provided
        if request.conversation_id:
            print(f"Saving user message for conversation: {request.conversation_id}")
            try:
                from chat import _exec
                import uuid
                print(f"Inserting user message: {request.question[:50]}...")
                _exec("""
                    INSERT INTO messages(conversation_id, role, content, meta) 
                    VALUES(%s, %s, %s, %s)
                """, (
                    uuid.UUID(request.conversation_id),
                    "user",
                    request.question,
                    json.dumps({"community_id": request.community_id, "role": request.role})
                ))
                print("✅ User message saved successfully")
                RequestLogger.log_validation(request_id, "chat_history", "user_message", True, "User message saved")
            except Exception as e:
                print(f"❌ Error saving user message: {e}")
                RequestLogger.log_error(request_id, "chat_history", str(e), {"stage": "save_user_message"})
        else:
            print("No conversation_id provided, skipping user message save")

        # Step 4: Check documents availability
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM documents WHERE community_id = %s", (request.community_id,))
                    doc_count = cur.fetchone()[0]
            
            docs_available = doc_count > 0
            docs_msg = f"Found {doc_count} documents" if docs_available else f"No documents found for community {request.community_id}"
            RequestLogger.log_validation(request_id, "documents", doc_count, docs_available, docs_msg)
            print(f"Documents availability check: available={docs_available}, msg='{docs_msg}', count={doc_count}")
            
        except Exception as e:
            RequestLogger.log_error(request_id, "documents_check", str(e), {"community_id": request.community_id})
            docs_available = False
            docs_msg = f"Error checking documents: {str(e)}"
            doc_count = 0
        
        if not docs_available:
            # Get community name
            try:
                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT name FROM communities WHERE id = %s", (request.community_id,))
                        result = cur.fetchone()
                        community_name = result[0] if result else f"ID {request.community_id}"
            except Exception as e:
                community_name = f"ID {request.community_id}"
            
            # Generate appropriate response for no documents
            answer = f"Для сообщества '{community_name}' пока нет загруженных документов. Обратитесь к менеджеру сообщества для загрузки документов."
            
            # Save assistant response to chat history
            if request.conversation_id:
                print(f"Saving assistant message for conversation: {request.conversation_id}")
                try:
                    from chat import _exec
                    import uuid
                    print(f"Inserting assistant message: {answer[:50]}...")
                    _exec("""
                        INSERT INTO messages(conversation_id, role, content, meta) 
                        VALUES(%s, %s, %s, %s)
                    """, (
                        uuid.UUID(request.conversation_id),
                        "assistant",
                        answer,
                        json.dumps({"confidence": 0.0, "sources": []})
                    ))
                    print("✅ Assistant message saved successfully")
                    # Update conversation timestamp
                    _exec("""
                        UPDATE conversations 
                        SET updated_at=NOW() AT TIME ZONE 'UTC' 
                        WHERE id=%s
                    """, (uuid.UUID(request.conversation_id),))
                    print("✅ Conversation timestamp updated")
                    RequestLogger.log_validation(request_id, "chat_history", "assistant_message", True, "Assistant message saved")
                except Exception as e:
                    print(f"❌ Error saving assistant message: {e}")
                    RequestLogger.log_error(request_id, "chat_history", str(e), {"stage": "save_assistant_message"})
            else:
                print("No conversation_id provided, skipping message save")
            
            return AskResponse(
                answer=answer,
                sources=[],
                sources_grouped=[],
                confidence=0.0
            )
        
        # Step 5: Validate OpenAI configuration
        openai_valid, openai_msg = RequestValidator.validate_openai_config(request_id)
        RequestLogger.log_validation(request_id, "openai_config", "API_KEY", openai_valid, openai_msg)
        if not openai_valid:
            raise HTTPException(status_code=500, detail=f"OpenAI configuration failed: {openai_msg}")
        
        # Step 6: Search for relevant documents
        try:
            print(f"Ask request: community_id={request.community_id}, question='{request.question}'")
            
            from search.retrieve import search_chunks
            chunks = search_chunks(request.community_id, request.question, k=6)
            RequestLogger.log_document_search(request_id, request.community_id, request.question, len(chunks), "chunks")
            
            if not chunks:
                RequestLogger.log_error(request_id, "document_search", "No chunks found", {"community_id": request.community_id})
                
                # Check if there are any documents for this community at all
                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT COUNT(*) FROM documents WHERE community_id = %s", (request.community_id,))
                        doc_count = cur.fetchone()[0]
                        
                        if doc_count == 0:
                            # No documents for this community - get community name
                            try:
                                cur.execute("SELECT name FROM communities WHERE id = %s", (request.community_id,))
                                result = cur.fetchone()
                                community_name = result[0] if result else f"ID {request.community_id}"
                            except Exception:
                                community_name = f"ID {request.community_id}"
                            
                            return AskResponse(
                                answer=f"Для сообщества '{community_name}' пока нет загруженных документов. Обратитесь к менеджеру сообщества для загрузки документов.",
                                sources=[],
                                sources_grouped=[],
                                confidence=0.0
                            )
                        else:
                            # Documents exist but no relevant chunks found
                            return AskResponse(
                                answer="Не удалось найти релевантную информацию в документах для вашего вопроса. Попробуйте переформулировать вопрос или обратитесь к менеджеру сообщества.",
                                sources=[],
                                sources_grouped=[],
                                confidence=0.0
                            )
        except Exception as e:
            RequestLogger.log_error(request_id, "document_search", str(e), {"community_id": request.community_id})
            raise HTTPException(status_code=500, detail=f"Document search failed: {str(e)}")
        
        # Step 7: Get LLM answer
        try:
            RequestLogger.log_openai_request(request_id, "gpt-4o-mini", len(request.question), 1000)
            
            result = answer_question(
                community_id=request.community_id,
                role=request.role,
                question=request.question,
                k=6
            )
            
            answer_length = len(result.get("answer", ""))
            sources_count = len(result.get("sources", []))
            confidence = result.get("confidence", 0.0)
            
            RequestLogger.log_openai_response(request_id, True, answer_length)
            RequestLogger.log_final_response(request_id, answer_length, sources_count, confidence)
            
        except Exception as e:
            RequestLogger.log_openai_response(request_id, False, 0, str(e))
            RequestLogger.log_error(request_id, "openai_request", str(e), {"question": request.question})
            raise HTTPException(status_code=500, detail=f"OpenAI request failed: {str(e)}")
        
        # Step 8: Save assistant message to chat history
        if request.conversation_id:
            try:
                from chat import _exec
                import uuid
                _exec("""
                    INSERT INTO messages(conversation_id, role, content, meta) 
                    VALUES(%s, %s, %s, %s)
                """, (
                    uuid.UUID(request.conversation_id),
                    "assistant",
                    result["answer"],
                    json.dumps({
                        "confidence": result["confidence"],
                        "sources": result["sources"]
                    })
                ))
                # Update conversation timestamp
                _exec("""
                    UPDATE conversations 
                    SET updated_at=NOW() AT TIME ZONE 'UTC' 
                    WHERE id=%s
                """, (uuid.UUID(request.conversation_id),))
                RequestLogger.log_validation(request_id, "chat_history", "assistant_message", True, "Assistant message saved")
            except Exception as e:
                RequestLogger.log_error(request_id, "chat_history", str(e), {"stage": "save_assistant_message"})
        
        # Step 9: Log Q&A interaction
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO qa_logs (community_id, user_role, question, answer, confidence, sources)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    """, (
                        request.community_id,
                        request.role,
                        request.question,
                        result.get("answer", ""),
                        float(result.get("confidence") or 0.0),
                        json.dumps(result.get("sources") or [])
                    ))
                    conn.commit()
            RequestLogger.log_validation(request_id, "qa_logs", "insert", True, "Q&A logged successfully")
        except Exception as e:
            RequestLogger.log_error(request_id, "qa_logs", str(e), {"stage": "log_interaction"})
        
        # Step 10: Create grouped sources
        try:
            sources_grouped = group_sources(chunks, http_request)
            RequestLogger.log_validation(request_id, "sources_grouped", len(sources_grouped), True, "Grouped sources created")
            
            print(f"Returning {len(sources_grouped)} sources")
            for source in sources_grouped:
                print(f"  - {source['title']} (url: {source['url']})")
        except Exception as e:
            RequestLogger.log_error(request_id, "sources_grouped", str(e), {"chunks_count": len(chunks)})
            sources_grouped = []
        
        # Step 11: Return response
        try:
            response = AskResponse(
                answer=result["answer"],
                sources=[Source(text=s["text"], document=s["document"]) for s in result["sources"]],
                sources_grouped=[GroupedSource(**sg) for sg in sources_grouped],
                confidence=result["confidence"]
            )
            RequestLogger.log_validation(request_id, "final_response", "success", True, "Response created successfully")
            return response
        except Exception as e:
            RequestLogger.log_error(request_id, "final_response", str(e), {"sources_count": len(result.get("sources", []))})
            raise HTTPException(status_code=500, detail=f"Response creation failed: {str(e)}")
        
    except HTTPException as e:
        if request_id:
            RequestLogger.log_error(request_id, "http_exception", str(e.detail), {"status_code": e.status_code})
        raise
    except Exception as e:
        if request_id:
            RequestLogger.log_error(request_id, "unexpected_error", str(e), {"error_type": type(e).__name__})
        else:
            RequestLogger.log_error("unknown", "unexpected_error", str(e), {"error_type": type(e).__name__})
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HOA AI Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "POST /ask": "Ask a question to the HOA assistant",
            "GET /communities": "Get list of all communities"
        }
    }

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    try:
        # Test database connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        
        return {"status": "healthy", "database": "connected", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"status": "unhealthy", "database": str(e), "timestamp": datetime.now().isoformat()}

@app.get("/health/openai")
async def health_check_openai():
    """OpenAI health check endpoint"""
    try:
        from openai import OpenAI
        from pathlib import Path
        from dotenv import load_dotenv
        
        # Load environment variables
        env_path = Path('/app/.env')
        load_dotenv(env_path)
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return {"status": "unhealthy", "openai": "API key not found", "timestamp": datetime.now().isoformat()}
        
        # Test API key by creating client
        client = OpenAI(api_key=api_key)
        
        # Test with a simple request
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        
        return {
            "status": "healthy", 
            "openai": "connected", 
            "model": "gpt-4o-mini",
            "response_length": len(response.choices[0].message.content),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "unhealthy", "openai": str(e), "timestamp": datetime.now().isoformat()}

@app.get("/health/documents")
async def health_check_documents():
    """Documents health check endpoint"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Check documents count
                cur.execute("SELECT COUNT(*) FROM documents")
                doc_count = cur.fetchone()[0]
                
                # Check chunks count
                cur.execute("SELECT COUNT(*) FROM chunks")
                chunk_count = cur.fetchone()[0]
                
                # Check topics count
                cur.execute("SELECT COUNT(*) FROM topics")
                topic_count = cur.fetchone()[0]
                
                # Check communities count
                cur.execute("SELECT COUNT(*) FROM communities")
                community_count = cur.fetchone()[0]
        
        return {
            "status": "healthy",
            "documents": {
                "count": doc_count,
                "chunks": chunk_count,
                "topics": topic_count,
                "communities": community_count
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "unhealthy", "documents": str(e), "timestamp": datetime.now().isoformat()}

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint without AI"""
    return {
        "message": "Test endpoint working",
        "timestamp": datetime.now().isoformat(),
        "status": "success"
    }

@app.get("/communities")
async def get_communities():
    """Get list of all communities (public endpoint)"""
    print("Communities endpoint called")
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, name 
                    FROM communities 
                    ORDER BY name ASC
                """)
                communities = cur.fetchall()
        
        print(f"Found {len(communities)} communities")
        result = [
            {
                "id": community[0],
                "name": community[1],
                "description": ""
            }
            for community in communities
        ]
        print("Returning communities:", result)
        return result
    except Exception as e:
        print(f"Error in communities endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Admin authentication dependency
def check_admin_auth(request: Request):
    """Check if user is authenticated as admin"""
    if request.cookies.get("admin_auth") != "1":
        raise HTTPException(status_code=401, detail="Admin authentication required")
    return True

# Admin API authentication endpoints
@app.post("/admin/api/login")
def admin_api_login(body: LoginBody, response: Response):
    """Admin login endpoint"""
    if body.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Bad password")
    response.set_cookie(
        key="admin_auth",
        value="1",
        max_age=60*60*8,  # 8 hours
        httponly=True,
        secure=COOKIE_SECURE,  # True in production
        samesite="lax",
    )
    return {"ok": True}

@app.get("/admin/api/me")
def admin_api_me(admin_auth: str | None = Cookie(default=None)):
    """Check admin authentication status"""
    return {"authenticated": admin_auth == "1"}

@app.post("/admin/api/logout")
def admin_api_logout(response: Response):
    """Admin logout endpoint"""
    response.delete_cookie("admin_auth", samesite="lax")
    return {"ok": True}

# Admin routes
@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    """Admin login page"""
    return templates.TemplateResponse("admin_login.html", {"request": request})

@app.post("/admin/login")
async def admin_login(
    request: Request,
    response: Response,
    password: str = Form(...)
):
    """Admin login endpoint"""
    admin_password = os.getenv("ADMIN_PASSWORD", "admin")
    if password == admin_password:
        response = RedirectResponse(url="/admin", status_code=302)
        response.set_cookie(key="admin_auth", value="1", max_age=3600)  # 1 hour
        return response
    else:
        return templates.TemplateResponse("admin_login.html", {"request": request, "error": "Неверный пароль"})

@app.get("/admin", response_class=HTMLResponse)
async def admin_index(request: Request, auth: bool = Depends(check_admin_auth)):
    """Admin main page"""
    return templates.TemplateResponse("admin_index.html", {"request": request})

@app.get("/admin/upload", response_class=HTMLResponse)
async def admin_upload_page(request: Request, auth: bool = Depends(check_admin_auth)):
    """Admin upload page"""
    return templates.TemplateResponse("admin_upload.html", {"request": request})

@app.post("/admin/upload")
async def admin_upload(
    request: Request,
    community_id: int = Form(...),
    title: str = Form(...),
    doc_type: str = Form(...),
    visibility: str = Form(...),
    file: UploadFile = File(...),
    auth: bool = Depends(check_admin_auth)
):
    """Admin upload endpoint"""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.txt')):
            raise HTTPException(status_code=400, detail="Only PDF and TXT files allowed")
        
        # Create unique filename
        file_ext = Path(file.filename).suffix
        unique_filename = f"{secrets.token_hex(8)}{file_ext}"
        file_path = Path("data") / unique_filename
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Import and call save_document
        from ingest.save import save_document
        
        result = save_document(
            community_id=community_id,
            title=title,
            doc_type=doc_type,
            file_path=str(file_path),
            visibility=visibility
        )
        
        return RedirectResponse(url="/admin/documents", status_code=302)
        
    except Exception as e:
        return templates.TemplateResponse("admin_upload.html", {
            "request": request,
            "error": f"Ошибка загрузки: {str(e)}"
        })

@app.get("/admin/documents", response_class=HTMLResponse)
async def admin_documents(request: Request, auth: bool = Depends(check_admin_auth)):
    """Admin documents list page"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT d.id, d.title, d.doc_type, d.created_at, d.visibility,
                           COUNT(c.id) as chunks_count
                    FROM documents d
                    LEFT JOIN chunks c ON d.id = c.document_id
                    GROUP BY d.id, d.title, d.doc_type, d.created_at, d.visibility
                    ORDER BY d.created_at DESC
                """)
                documents = cur.fetchall()
        
        return templates.TemplateResponse("admin_documents.html", {
            "request": request,
            "documents": documents
        })
    except Exception as e:
        return templates.TemplateResponse("admin_documents.html", {
            "request": request,
            "error": str(e),
            "documents": []
        })

@app.get("/admin/logs", response_class=HTMLResponse)
async def admin_logs(request: Request, auth: bool = Depends(check_admin_auth)):
    """Admin logs page"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT created_at, user_role, question, confidence, sources
                    FROM qa_logs
                    ORDER BY created_at DESC
                    LIMIT 100
                """)
                logs = cur.fetchall()
        
        return templates.TemplateResponse("admin_logs.html", {
            "request": request,
            "logs": logs
        })
    except Exception as e:
        return templates.TemplateResponse("admin_logs.html", {
            "request": request,
            "error": str(e),
            "logs": []
        })

# Admin API endpoints
@app.get("/admin/api/community-documents")
async def admin_api_community_documents(
    community_id: int,
    _: bool = Depends(is_admin)
):
    """Get list of documents for a community"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT d.id, d.title, d.doc_type, d.created_at,
                           COUNT(c.id) as chunks_count
                    FROM documents d
                    LEFT JOIN chunks c ON d.id = c.document_id
                    WHERE d.community_id = %s
                    GROUP BY d.id, d.title, d.doc_type, d.created_at
                    ORDER BY d.created_at DESC
                """, (community_id,))
                documents = cur.fetchall()
        
        return [
            {
                "id": doc[0],
                "title": doc[1],
                "doc_type": doc[2],
                "created_at": doc[3].isoformat() if doc[3] else "",
                "chunks": doc[4]
            }
            for doc in documents
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/admin/api/logs")
async def admin_api_logs(
    limit: int = 100,
    _: bool = Depends(is_admin)
):
    """Get recent QA logs"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT created_at, user_role, question, confidence
                    FROM qa_logs
                    ORDER BY id DESC
                    LIMIT %s
                """, (limit,))
                logs = cur.fetchall()
        
        return [
            {
                "created_at": log[0].isoformat() if log[0] else "",
                "user_role": log[1],
                "question": log[2],
                "confidence": float(log[3]) if log[3] else 0.0
            }
            for log in logs
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/admin/api/upload")
async def admin_api_upload(
    community_id: int = Form(...),
    title: str = Form(...),
    doc_type: str = Form(...),
    visibility: str = Form(...),
    file: UploadFile = File(...),
    use_topic_analysis: bool = Form(True),
    _: bool = Depends(is_admin)
):
    """Upload document via API"""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.txt')):
            raise HTTPException(status_code=400, detail="Only PDF and TXT files allowed")
        
        # Check file size and adjust topic analysis if needed
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        
        if file_size_mb > 10 and use_topic_analysis:
            print(f"[info] Large file detected ({file_size_mb:.1f}MB), disabling topic analysis for better performance")
            use_topic_analysis = False
        
        # Check file size (25MB limit)
        MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB in bytes
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 25MB")
        
        # Create unique filename
        file_ext = Path(file.filename).suffix
        unique_filename = f"{secrets.token_hex(8)}{file_ext}"
        file_path = Path("data") / unique_filename
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        # Save file
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        
        # Import and call save_document
        from ingest.save import save_document
        
        # Process document asynchronously to avoid timeout
        import asyncio
        result = await asyncio.get_event_loop().run_in_executor(
            None, 
            save_document,
            community_id,
            title,
            doc_type,
            str(file_path),
            visibility,
            use_topic_analysis
        )
        
        # Also save to document registry for admin management
        try:
            rel_path = f"data/{unique_filename}"
            
            # Count pages for PDF files
            pages = 0
            if file.filename.lower().endswith('.pdf'):
                try:
                    from ingest.chunker import iter_pdf_chunks
                    for chunk in iter_pdf_chunks(file_content, file.filename):
                        pages = max(pages, chunk.get('page', 0))
                except Exception:
                    pages = 0
            
            # Insert document record and get the ID
            doc_id = _exec("""
                INSERT INTO documents (filename, title, doc_type, visibility, file_path, pages, size_bytes, community_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (file.filename, title, doc_type, visibility, rel_path, pages, len(file_content), community_id), fetch=True)[0][0]
            
            # Save chunks to admin_doc_chunks table for admin preview
            if file.filename.lower().endswith('.pdf'):
                try:
                    from ingest.segment import sentence_fragments, pack_sentences, topic_segments, count_tokens
                    from pypdf import PdfReader
                    import io
                    
                    # Extract pages
                    reader = PdfReader(io.BytesIO(file_content))
                    pages = []
                    for i, p in enumerate(reader.pages, start=1):
                        try: 
                            txt = p.extract_text() or ""
                        except Exception: 
                            txt = ""
                        pages.append((i, txt))
                    
                    if segmentation == "topic":
                        # Topic segmentation
                        topics = topic_segments(pages, min_sim_drop=min_sim_drop, max_tokens=max_topic_tokens)
                        
                        # Save topics to doc_topics table
                        for topic in topics:
                            try:
                                _exec("""
                                    INSERT INTO doc_topics (doc_id, topic_index, title, start_page, end_page)
                                    VALUES (%s, %s, %s, %s, %s)
                                """, (doc_id, topic['topic_index'], None, min(topic['pages']), max(topic['pages'])))
                                print(f"Topic {topic['topic_index']} saved successfully")
                            except Exception as e:
                                print(f"[warn] failed to save topic: {e}")
                        
                        # Create chunks from topics
                        for topic in topics:
                            topic_id = topic['topic_index']
                            text = topic['text']
                            pages_list = topic['pages']
                            
                            # If topic is too large, split it further
                            if count_tokens(text) > max_tokens:
                                frags = sentence_fragments([(p, text) for p in pages_list])
                                for chunk in pack_sentences(frags, max_tokens=max_tokens, overlap_sents=overlap):
                                    _exec("""
                                        INSERT INTO admin_doc_chunks (doc_id, page, start_pos, end_pos, text, section, topic_id, token_count)
                                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                    """, (doc_id, min(chunk['pages']), 0, 0, chunk['text'], f"p{min(chunk['pages'])}", topic_id, chunk['token_count']))
                            else:
                                _exec("""
                                    INSERT INTO admin_doc_chunks (doc_id, page, start_pos, end_pos, text, section, topic_id, token_count)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                """, (doc_id, min(pages_list), 0, 0, text, f"p{min(pages_list)}", topic_id, count_tokens(text)))
                    else:
                        # Smart segmentation (default)
                        frags = sentence_fragments(pages)
                        for chunk in pack_sentences(frags, max_tokens=max_tokens, overlap_sents=overlap):
                            _exec("""
                                INSERT INTO admin_doc_chunks (doc_id, page, start_pos, end_pos, text, section, token_count)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """, (doc_id, min(chunk['pages']), 0, 0, chunk['text'], f"p{min(chunk['pages'])}", chunk['token_count']))
                            
                except Exception as e:
                    print(f"[warn] failed to save chunks to admin_doc_chunks: {e}")
            
        except Exception as e:
            print(f"[warn] failed to save document to registry: {e}")
        
        return {
            "document_id": result["document_id"],
            "topics_inserted": result.get("topics_inserted", 0),
            "chunks_inserted": result.get("chunks_inserted", 0),
            "status": "success",
            "message": f"Document '{title}' uploaded and processed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

# Chunk preview endpoint
@app.post("/admin/api/chunk-preview")
async def chunk_preview(
    file: UploadFile = File(...), 
    mode: str = Form("smart"),
    max_tokens: int = Form(500),
    overlap: int = Form(2),
    min_sim_drop: float = Form(0.20),
    max_topic_tokens: int = Form(2000),
    _: bool = Depends(is_admin)
):
    """Stream chunk preview for uploaded file with smart or topic segmentation"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files allowed for preview")
        
        # Read file data
        data = await file.read()
        
        # Import segmentation modules
        from ingest.segment import sentence_fragments, pack_sentences, topic_segments
        from pypdf import PdfReader
        import io
        
        # Extract pages
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for i, p in enumerate(reader.pages, start=1):
            try: 
                txt = p.extract_text() or ""
            except Exception: 
                txt = ""
            pages.append((i, txt))
        
        def gen():
            total = 0
            try:
                if mode == "topic":
                    # Send progress updates for topic segmentation
                    yield {"type": "progress", "message": "Computing embeddings..."}
                    for seg in topic_segments(pages, min_sim_drop=min_sim_drop, max_tokens=max_topic_tokens):
                        total += 1
                        yield {"type": "topic", "payload": seg}
                else:  # smart mode
                    yield {"type": "progress", "message": "Processing text..."}
                    frags = sentence_fragments(pages)
                    yield {"type": "progress", "message": f"Found {len(frags)} sentences, creating chunks..."}
                    for ch in pack_sentences(frags, max_tokens=max_tokens, overlap_sents=overlap):
                        total += 1
                        yield {"type": "chunk", "payload": ch}
                yield {"type": "done", "total": total}
            except Exception as e:
                yield {"type": "error", "message": str(e)}
        
        return StreamingResponse(
            sse_stream(gen()),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview error: {str(e)}")

# Community management endpoints
@app.get("/admin/api/communities")
def list_communities(_: bool = Depends(is_admin)):
    """List all communities"""
    try:
        rows = _exec("""SELECT id, name FROM communities ORDER BY name""", fetch=True)
        return [dict(id=r[0], name=r[1], description="", created_at=None) for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing communities: {str(e)}")

@app.post("/admin/api/communities")
def create_community(request: CommunityCreateRequest, _: bool = Depends(is_admin)):
    """Create a new community"""
    try:
        name = request.name
        
        if not name:
            raise HTTPException(status_code=400, detail="Name is required")
        
        # Check if community with this name already exists
        existing = _exec("SELECT id FROM communities WHERE name = %s", (name,), fetch=True)
        if existing:
            raise HTTPException(status_code=400, detail="Community with this name already exists")
        
        # Insert new community
        community_id = _exec("""
            INSERT INTO communities (name) 
            VALUES (%s) 
            RETURNING id
        """, (name,), fetch=True)
        
        return dict(id=community_id[0][0], name=name, description="", message="Community created successfully")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating community: {str(e)}")

@app.put("/admin/api/communities/{community_id}")
def update_community(community_id: int, request: CommunityUpdateRequest, _: bool = Depends(is_admin)):
    """Update a community"""
    try:
        name = request.name
        
        if not name:
            raise HTTPException(status_code=400, detail="Name is required")
        
        # Check if community exists
        existing = _exec("SELECT id FROM communities WHERE id = %s", (community_id,), fetch=True)
        if not existing:
            raise HTTPException(status_code=404, detail="Community not found")
        
        # Check if another community with this name exists
        name_conflict = _exec("SELECT id FROM communities WHERE name = %s AND id != %s", (name, community_id), fetch=True)
        if name_conflict:
            raise HTTPException(status_code=400, detail="Another community with this name already exists")
        
        # Update community
        _exec("""
            UPDATE communities 
            SET name = %s 
            WHERE id = %s
        """, (name, community_id))
        
        return dict(id=community_id, name=name, description="", message="Community updated successfully")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating community: {str(e)}")

@app.delete("/admin/api/communities/{community_id}")
def delete_community(community_id: int, _: bool = Depends(is_admin)):
    """Delete a community"""
    try:
        # Check if community exists
        existing = _exec("SELECT id FROM communities WHERE id = %s", (community_id,), fetch=True)
        if not existing:
            raise HTTPException(status_code=404, detail="Community not found")
        
        # Check if community has documents
        docs = _exec("SELECT COUNT(*) FROM documents WHERE community_id = %s", (community_id,), fetch=True)
        if docs[0][0] > 0:
            raise HTTPException(status_code=400, detail="Cannot delete community with existing documents")
        
        # Delete community
        _exec("DELETE FROM communities WHERE id = %s", (community_id,))
        
        return dict(message="Community deleted successfully")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting community: {str(e)}")

# Document management endpoints
@app.get("/admin/api/documents")
def list_documents(community_id: int = None, _: bool = Depends(is_admin)):
    """List all documents, optionally filtered by community_id"""
    try:
        if community_id:
            rows = _exec("""SELECT id::text, file_path as filename, title, doc_type, 'public' as visibility, file_path as rel_path, 0 as pages, 0 as size_bytes, NOW() as created_at, community_id
                            FROM documents WHERE community_id = %s ORDER BY id DESC""", (community_id,), fetch=True)
        else:
            rows = _exec("""SELECT id::text, file_path as filename, title, doc_type, 'public' as visibility, file_path as rel_path, 0 as pages, 0 as size_bytes, NOW() as created_at, community_id
                            FROM documents ORDER BY id DESC""", fetch=True)
        return [dict(id=r[0], filename=r[1], title=r[2], doc_type=r[3], visibility=r[4], rel_path=r[5], pages=r[6], size_bytes=r[7], created_at=r[8], community_id=r[9]) for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.get("/admin/api/documents/{doc_id}/chunks")
def get_document_chunks(doc_id: str, _: bool = Depends(is_admin)):
    """Get chunks for a specific document"""
    try:
        import uuid
        rows = _exec("""SELECT id, page, start_pos, end_pos, LEFT(text, 2000)
                        FROM chunks WHERE document_id=%s ORDER BY page NULLS FIRST, id ASC""",
                     (int(doc_id),), fetch=True)
        return [dict(id=r[0], page=r[1], start=r[2], end=r[3], preview=r[4]) for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting document chunks: {str(e)}")

@app.post("/admin/api/generate-topic-title")
def generate_topic_title(request: GenerateTopicTitleRequest, _: bool = Depends(is_admin)):
    """Generate a topic title from text using LLM"""
    try:
        text = request.text
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Import OpenAI client
        from openai import OpenAI
        import os
        from pathlib import Path
        from dotenv import load_dotenv
        
        # Load environment variables
        env_path = Path('/app/.env')
        load_dotenv(env_path)
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        client = OpenAI(api_key=api_key)
        
        # Generate topic title
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты эксперт по анализу документов ТСЖ/ЖКХ. Создай краткое название (2-4 слова) для тематического раздела документа на основе предоставленного текста. Название должно быть понятным и отражать суть темы. Примеры: 'Парковка', 'Платежи и взносы', 'Общие собрания', 'Правила проживания'."},
                {"role": "user", "content": f"Создай название для этого раздела документа:\n\n{text}"}
            ],
            temperature=0.3,
            max_tokens=50
        )
        
        title = response.choices[0].message.content.strip()
        
        return {"title": title}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating topic title: {str(e)}")

@app.get("/admin/api/documents/{doc_id}/topics")
def get_document_topics(doc_id: str, _: bool = Depends(is_admin)):
    """Get topics for a specific document"""
    try:
        # For now, just return empty list since topics table is empty
        # TODO: Implement proper topic retrieval when topics are available
        return []
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting document topics: {str(e)}")

@app.post("/admin/api/fix-topics-table")
def fix_topics_table(_: bool = Depends(is_admin)):
    """Fix doc_topics table structure"""
    try:
        # Drop and recreate table with correct structure
        _exec("DROP TABLE IF EXISTS doc_topics CASCADE")
        _exec("""
            CREATE TABLE doc_topics (
              id BIGSERIAL PRIMARY KEY,
              doc_id UUID NOT NULL REFERENCES admin_documents(id) ON DELETE CASCADE,
              topic_index INT NOT NULL,
              title TEXT NULL,
              start_page INT NULL,
              end_page INT NULL,
              created_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC')
            )
        """)
        _exec("CREATE INDEX IF NOT EXISTS idx_doc_topics_doc ON doc_topics(doc_id, topic_index)")
        
        # Add missing columns to admin_doc_chunks
        _exec("ALTER TABLE admin_doc_chunks ADD COLUMN IF NOT EXISTS section TEXT NULL")
        _exec("ALTER TABLE admin_doc_chunks ADD COLUMN IF NOT EXISTS topic_id INT NULL")
        _exec("ALTER TABLE admin_doc_chunks ADD COLUMN IF NOT EXISTS token_count INT NULL")
        
        return {"status": "success", "message": "Tables fixed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fixing tables: {str(e)}")

@app.delete("/admin/api/documents/{doc_id}")
def delete_document(doc_id: str, _: bool = Depends(is_admin)):
    """Delete document and its chunks"""
    try:
        import os
        
        # Get static root directory
        STATIC_ROOT = os.path.join(os.path.dirname(__file__), "static")
        
        # Read file path from documents table
        rows = _exec("SELECT file_path FROM documents WHERE id=%s", (int(doc_id),), fetch=True)
        if not rows:
            raise HTTPException(404, "Document not found")
        
        file_path = rows[0][0]
        
        # Delete DB rows (cascades delete chunks)
        _exec("DELETE FROM documents WHERE id=%s", (int(doc_id),))
        
        # Remove file from disk (best-effort)
        try:
            abs_path = os.path.join(STATIC_ROOT, file_path.lstrip("/"))
            if abs_path.startswith(STATIC_ROOT) and os.path.exists(abs_path):
                os.remove(abs_path)
        except Exception:
            pass  # Ignore file removal errors
        
        # TODO: Also remove embeddings for this doc if you keep a separate table
        # This would require calling your existing cleanup helper
        
        return {"ok": True}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        timeout_keep_alive=600,  # 10 minutes for large files
        timeout_graceful_shutdown=60
    )