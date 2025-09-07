#!/usr/bin/env python3
"""
FastAPI application for HOA AI Assistant
"""

from fastapi import Cookie, Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from starlette.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
import psycopg
import json
import secrets
import shutil
from pathlib import Path
import os
from dotenv import load_dotenv

# Import local modules
from llm.answer import answer_question
import chat as chat_mod

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

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
    "http://localhost:5173", "http://localhost:5174",
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
STATIC_DIR = (Path(__file__).parent / "backend" / "static").resolve()
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    print(f"[WARN] static dir not found: {STATIC_DIR}; skipping static mount")
templates = Jinja2Templates(directory="backend/templates")

# Initialize chat module and include router
chat_mod.init_chat(app)
app.include_router(chat_mod.router, prefix="/chat", tags=["chat"])

# Pydantic models
class AskRequest(BaseModel):
    community_id: int = Field(..., gt=0, description="Community ID (must be positive)")
    role: Literal["resident", "board", "staff"] = Field(..., description="User role")
    question: str = Field(..., min_length=1, description="User question")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for chat history")

class Source(BaseModel):
    title: str
    section: str

class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: float

# Admin API models
class DocumentInfo(BaseModel):
    id: int
    title: str
    doc_type: str
    created_at: str
    chunks_count: int

class LogEntry(BaseModel):
    created_at: str
    user_role: str
    question: str
    confidence: float

class UploadResponse(BaseModel):
    document_id: int
    chunks_inserted: int

class LoginBody(BaseModel):
    password: str

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
    Ask a question to HOA AI Assistant
    
    Args:
        request: AskRequest with community_id, role, question, and optional conversation_id
        http_request: HTTP request object for accessing session state
        
    Returns:
        AskResponse with answer, sources, confidence, and conversation_id
    """
    try:
        # Validate input (already done by Pydantic, but double-check)
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if request.community_id <= 0:
            raise HTTPException(status_code=400, detail="Community ID must be positive")
        
        # If conversation_id is provided, save user message to chat history
        if request.conversation_id:
            try:
                from chat import _exec
                import uuid
                _exec("""
                    INSERT INTO messages(conversation_id, role, content, meta) 
                    VALUES(%s, %s, %s, %s)
                """, (
                    uuid.UUID(request.conversation_id),
                    "user",
                    request.question,
                    json.dumps({"community_id": request.community_id, "role": request.role})
                ))
            except Exception as e:
                print(f"[warn] failed to save user message to chat history: {e}")
        
        # Call answer_question function
        result = answer_question(
            community_id=request.community_id,
            role=request.role,
            question=request.question,
            k=6
        )
        
        # If conversation_id is provided, save assistant message to chat history
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
            except Exception as e:
                print(f"[warn] failed to save assistant message to chat history: {e}")
        
        # Log the Q&A interaction
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
        except Exception as e:
            # Log error but don't fail the request
            print(f"[warn] failed to write qa_logs: {e}")
        
        # Return response
        return AskResponse(
            answer=result["answer"],
            sources=[Source(title=s["title"], section=s["section"]) for s in result["sources"]],
            confidence=result["confidence"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HOA AI Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "POST /ask": "Ask a question to the HOA assistant"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": str(e)}

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
    response: Response,
    password: str = Form(...),
    admin_password: str = Depends(lambda: os.getenv("ADMIN_PASSWORD", "admin"))
):
    """Admin login endpoint"""
    if password == admin_password:
        response = RedirectResponse(url="/admin", status_code=302)
        response.set_cookie(key="admin_auth", value="1", max_age=3600)  # 1 hour
        return response
    else:
        return templates.TemplateResponse("admin_login.html", {"request": Request, "error": "Неверный пароль"})

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
@app.get("/admin/api/documents")
async def admin_api_documents(
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
    _: bool = Depends(is_admin)
):
    """Upload document via API"""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.txt')):
            raise HTTPException(status_code=400, detail="Only PDF and TXT files allowed")
        
        # Check file size (25MB limit)
        MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB in bytes
        
        # Read file content to check size
        file_content = await file.read()
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
            visibility
        )
        
        return {
            "document_id": result["document_id"],
            "chunks_inserted": result["chunks_inserted"],
            "status": "success",
            "message": f"Document '{title}' uploaded and processed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        timeout_keep_alive=300,  # 5 minutes
        timeout_graceful_shutdown=30
    )