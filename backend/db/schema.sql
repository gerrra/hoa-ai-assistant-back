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

-- топики документов (тематические разделы)
CREATE TABLE IF NOT EXISTS topics (
  id BIGSERIAL PRIMARY KEY,
  document_id INT REFERENCES documents(id) ON DELETE CASCADE,
  title TEXT NOT NULL,                    -- название топика (например "Парковка", "Платежи")
  description TEXT,                       -- краткое описание топика
  content TEXT NOT NULL,                  -- весь контент топика из документа
  embedding VECTOR(1536),                 -- эмбеддинг топика для поиска
  token_count INT,                        -- количество токенов в топике
  page_numbers TEXT,                      -- номера страниц, откуда взят контент (JSON array)
  visibility TEXT DEFAULT 'resident',
  created_at TIMESTAMPTZ DEFAULT now()
);

-- кусочки текста + вектор эмбеддинга (оставляем для совместимости)
-- длина вектора 1536 под text-embedding-3-small (можно изменить позже)
CREATE TABLE IF NOT EXISTS chunks (
  id BIGSERIAL PRIMARY KEY,
  document_id INT REFERENCES documents(id) ON DELETE CASCADE,
  topic_id INT REFERENCES topics(id) ON DELETE SET NULL,  -- связь с топиком
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
CREATE INDEX IF NOT EXISTS idx_topics_document ON topics(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_topic ON chunks(topic_id);

-- если доступен HNSW/IVFFLAT для pgvector, можно добавить специализированный индекс позже
