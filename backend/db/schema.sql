-- сообщества (пока на демо может быть одно)
CREATE TABLE IF NOT EXISTS communities (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL
);

-- загруженные документы
CREATE TABLE IF NOT EXISTS documents (
  id SERIAL PRIMARY KEY,
  community_id INT REFERENCES communities(id) ON DELETE CASCADE,
  title TEXT,
  doc_type TEXT,                 -- CC&R, Bylaws, Rules, Policy, Guidelines
  effective_from DATE NULL,
  effective_to DATE NULL,
  version TEXT NULL,
  visibility TEXT DEFAULT 'resident',  -- 'resident' | 'board' | 'staff'
  file_path TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- кусочки текста + вектор эмбеддинга
-- длина вектора 1536 под text-embedding-3-small (можно изменить позже)
CREATE TABLE IF NOT EXISTS chunks (
  id BIGSERIAL PRIMARY KEY,
  document_id INT REFERENCES documents(id) ON DELETE CASCADE,
  section_ref TEXT,
  text TEXT,
  embedding VECTOR(1536),
  token_count INT,
  visibility TEXT DEFAULT 'resident'
);

-- логи вопросов-ответов
CREATE TABLE IF NOT EXISTS qa_logs (
  id BIGSERIAL PRIMARY KEY,
  community_id INT,
  user_role TEXT,
  question TEXT,
  answer TEXT,
  confidence NUMERIC,
  sources JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- индексы
CREATE INDEX IF NOT EXISTS idx_documents_community ON documents(community_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

-- если доступен HNSW/IVFFLAT для pgvector, можно добавить специализированный индекс позже
