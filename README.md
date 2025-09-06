# HOA AI Assistant - Backend

FastAPI backend for HOA AI Assistant with document processing and AI-powered Q&A.

## Quick Start

### Local Development

1. **Setup environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your values
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start database:**
   ```bash
   docker run -d --name hoa-db -e POSTGRES_USER=hoa -e POSTGRES_PASSWORD=hoa -e POSTGRES_DB=hoa -p 5432:5432 pgvector/pgvector:pg16
   ```

4. **Run application:**
   ```bash
   uvicorn app:app --reload
   ```

### Docker

1. **Using docker-compose:**
   ```bash
   docker compose -f docker-compose.local.yml up --build
   ```

2. **Manual build:**
   ```bash
   docker build -t hoa-backend .
   docker run -p 8000:8000 --env-file .env hoa-backend
   ```

## API Endpoints

- `GET /` - Health check
- `POST /ask` - Ask question
- `GET /admin` - Admin panel
- `POST /admin/api/upload` - Upload document
- `GET /admin/api/documents` - List documents
- `GET /admin/api/logs` - View logs

## Environment Variables

See `.env.example` for required variables.
