from __future__ import annotations
import os, json, uuid
from typing import Optional, Any
from fastapi import APIRouter, Depends, Request, Response, HTTPException
from pydantic import BaseModel
import psycopg

router = APIRouter()

def _dsn() -> str:
    url = os.getenv("DB_URL")
    if url:
        return url.replace("postgresql+psycopg://", "postgresql://")
    host = os.getenv("DB_HOST","db"); port=os.getenv("DB_PORT","5432")
    name = os.getenv("DB_NAME","hoa"); user=os.getenv("DB_USER","hoa"); pwd=os.getenv("DB_PASS","hoa")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{name}"

def _exec(sql: str, params: Optional[tuple]=None, fetch: bool=False, many: bool=False):
    with psycopg.connect(_dsn(), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            if fetch: 
                return cur.fetchall()
            if many:
                return cur.rowcount
            return None

BOOTSTRAP_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
  id UUID PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC'),
  last_seen_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC'),
  user_agent TEXT NULL,
  ip INET NULL
);
CREATE TABLE IF NOT EXISTS conversations (
  id UUID PRIMARY KEY,
  session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  title TEXT NULL,
  archived BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC'),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC')
);
CREATE INDEX IF NOT EXISTS idx_conversations_session_updated ON conversations(session_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
  id BIGSERIAL PRIMARY KEY,
  conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
  content TEXT NOT NULL,
  meta JSONB NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() AT TIME ZONE 'UTC')
);
CREATE INDEX IF NOT EXISTS idx_messages_conv_created ON messages(conversation_id, created_at ASC);
"""

def init_chat(app):
    # run DDL once at startup
    _exec(BOOTSTRAP_SQL)

# ------------ models ------------
class StartBody(BaseModel):
    title: Optional[str] = None

class MessageBody(BaseModel):
    role: str
    content: str
    meta: Optional[dict[str, Any]] = None

# ------------ helpers ------------
def _ensure_session(sid: str, ua: str, ip: Optional[str]):
    rows = _exec("SELECT 1 FROM sessions WHERE id=%s", (uuid.UUID(sid),), fetch=True)
    if not rows:
        _exec("INSERT INTO sessions(id,user_agent,ip) VALUES(%s,%s,%s)",
              (uuid.UUID(sid), ua[:400] if ua else None, ip))
    else:
        _exec("UPDATE sessions SET last_seen_at=NOW() AT TIME ZONE 'UTC', user_agent=%s WHERE id=%s",
              (ua[:400] if ua else None, uuid.UUID(sid)))

def _get_sid(request: Request) -> str:
    sid = getattr(request.state, "sid", None) or request.cookies.get("sid")
    if not sid:
        raise HTTPException(401, "no session")
    return sid

# ------------ endpoints ------------
@router.post("/start")
def start_conv(body: StartBody, request: Request):
    sid = _get_sid(request)
    _ensure_session(sid, request.headers.get("user-agent",""), request.client.host if request.client else None)
    cid = uuid.uuid4()
    _exec("INSERT INTO conversations(id, session_id, title) VALUES(%s,%s,%s)",
          (cid, uuid.UUID(sid), body.title))
    return {"conversation_id": str(cid)}

@router.get("/list")
def list_conversations(request: Request):
    sid = _get_sid(request)
    rows = _exec("""
      SELECT c.id::text, COALESCE(c.title,''), c.created_at, c.updated_at,
             (SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.id) AS message_count
      FROM conversations c
      WHERE c.session_id=%s AND NOT c.archived
      ORDER BY c.updated_at DESC
    """, (uuid.UUID(sid),), fetch=True)
    return [{"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3], "message_count": r[4]} for r in rows]

@router.get("/{cid}/messages")
def get_messages(cid: str, limit: int = 100, after_id: Optional[int] = None, before_id: Optional[int] = None, request: Request = None):
    sid = _get_sid(request)
    # authorization: cid must belong to sid
    owner = _exec("SELECT 1 FROM conversations WHERE id=%s AND session_id=%s", (uuid.UUID(cid), uuid.UUID(sid)), fetch=True)
    if not owner: raise HTTPException(404, "conversation not found")
    sql = "SELECT id, role, content, meta, created_at FROM messages WHERE conversation_id=%s"
    params = [uuid.UUID(cid)]
    if after_id:
        sql += " AND id > %s"; params.append(after_id)
    if before_id:
        sql += " AND id < %s"; params.append(before_id)
    sql += " ORDER BY id ASC LIMIT %s"; params.append(max(1, min(limit, 500)))
    rows = _exec(sql, tuple(params), fetch=True)
    return [{"id": r[0], "role": r[1], "content": r[2], "meta": r[3], "created_at": r[4]} for r in rows]

@router.post("/{cid}/messages")
def add_message(cid: str, body: MessageBody, request: Request):
    sid = _get_sid(request)
    owner = _exec("SELECT 1 FROM conversations WHERE id=%s AND session_id=%s", (uuid.UUID(cid), uuid.UUID(sid)), fetch=True)
    if not owner: raise HTTPException(404, "conversation not found")
    _exec("INSERT INTO messages(conversation_id,role,content,meta) VALUES(%s,%s,%s,%s)",
          (uuid.UUID(cid), body.role, body.content, json.dumps(body.meta) if body.meta is not None else None))
    _exec("UPDATE conversations SET updated_at=NOW() AT TIME ZONE 'UTC' WHERE id=%s", (uuid.UUID(cid),))
    return {"ok": True}

@router.post("/{cid}/title")
def set_title(cid: str, body: StartBody, request: Request):
    sid = _get_sid(request)
    owner = _exec("SELECT 1 FROM conversations WHERE id=%s AND session_id=%s", (uuid.UUID(cid), uuid.UUID(sid)), fetch=True)
    if not owner: raise HTTPException(404, "conversation not found")
    _exec("UPDATE conversations SET title=%s, updated_at=NOW() AT TIME ZONE 'UTC' WHERE id=%s",
          (body.title, uuid.UUID(cid)))
    return {"ok": True}
