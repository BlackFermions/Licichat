BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS ai_ingestion_jobs (
    id BIGSERIAL PRIMARY KEY,
    tender_id VARCHAR(255) NOT NULL REFERENCES tenders(id) ON DELETE CASCADE,
    source_signature CHAR(64) NOT NULL,
    selected_documents JSONB NOT NULL DEFAULT '[]'::jsonb,
    document_count SMALLINT NOT NULL,
    reason VARCHAR(80) NOT NULL DEFAULT 'pilot_seed',
    priority SMALLINT NOT NULL DEFAULT 50,
    status VARCHAR(24) NOT NULL DEFAULT 'pending',
    attempts SMALLINT NOT NULL DEFAULT 0,
    max_attempts SMALLINT NOT NULL DEFAULT 4,
    available_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    locked_at TIMESTAMPTZ,
    locked_by VARCHAR(120),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    last_error_code VARCHAR(80),
    last_error_message TEXT,
    pipeline_version VARCHAR(40) NOT NULL DEFAULT 'pilot-v1',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ai_ingestion_jobs_status_check CHECK (
        status IN ('pending', 'processing', 'retry', 'ready', 'failed', 'skipped')
    ),
    CONSTRAINT ai_ingestion_jobs_document_count_check CHECK (document_count BETWEEN 1 AND 2),
    CONSTRAINT ai_ingestion_jobs_attempts_check CHECK (
        attempts >= 0 AND max_attempts BETWEEN 1 AND 20
    ),
    CONSTRAINT ai_ingestion_jobs_priority_check CHECK (priority BETWEEN 0 AND 100),
    CONSTRAINT ai_ingestion_jobs_source_unique UNIQUE (
        tender_id, source_signature, pipeline_version
    )
);

CREATE INDEX IF NOT EXISTS idx_ai_ingestion_jobs_claim
    ON ai_ingestion_jobs (priority DESC, available_at, id)
    WHERE status IN ('pending', 'retry');

CREATE INDEX IF NOT EXISTS idx_ai_ingestion_jobs_tender
    ON ai_ingestion_jobs (tender_id, created_at DESC);

CREATE TABLE IF NOT EXISTS ai_document_assets (
    id BIGSERIAL PRIMARY KEY,
    tender_id VARCHAR(255) NOT NULL REFERENCES tenders(id) ON DELETE CASCADE,
    job_id BIGINT REFERENCES ai_ingestion_jobs(id) ON DELETE SET NULL,
    source_document_id VARCHAR(255) REFERENCES documents(id) ON DELETE SET NULL,
    document_role VARCHAR(24) NOT NULL,
    source_title TEXT,
    source_url TEXT NOT NULL,
    source_format VARCHAR(100),
    source_published_at TIMESTAMP WITHOUT TIME ZONE,
    extracted_filename TEXT,
    detected_mime_type VARCHAR(160),
    sha256 CHAR(64),
    byte_size BIGINT,
    storage_provider VARCHAR(24) NOT NULL DEFAULT 'azure_blob',
    original_object_key TEXT,
    markdown_object_key TEXT,
    status VARCHAR(24) NOT NULL DEFAULT 'pending',
    page_count INTEGER,
    native_text_pages INTEGER NOT NULL DEFAULT 0,
    ocr_pages INTEGER NOT NULL DEFAULT 0,
    text_char_count BIGINT NOT NULL DEFAULT 0,
    error_code VARCHAR(80),
    error_message TEXT,
    pipeline_version VARCHAR(40) NOT NULL DEFAULT 'pilot-v1',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ai_document_assets_role_check CHECK (
        document_role IN ('bases', 'buena_pro')
    ),
    CONSTRAINT ai_document_assets_status_check CHECK (
        status IN ('pending', 'downloading', 'extracting', 'ocr', 'ready', 'failed', 'skipped')
    ),
    CONSTRAINT ai_document_assets_size_check CHECK (byte_size IS NULL OR byte_size >= 0),
    CONSTRAINT ai_document_assets_pages_check CHECK (
        (page_count IS NULL OR page_count >= 0)
        AND native_text_pages >= 0
        AND ocr_pages >= 0
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_ai_document_assets_source
    ON ai_document_assets (
        tender_id,
        source_document_id,
        COALESCE(extracted_filename, ''),
        pipeline_version
    );

CREATE INDEX IF NOT EXISTS idx_ai_document_assets_tender
    ON ai_document_assets (tender_id, document_role, status);

CREATE INDEX IF NOT EXISTS idx_ai_document_assets_sha256
    ON ai_document_assets (sha256)
    WHERE sha256 IS NOT NULL;

CREATE TABLE IF NOT EXISTS ai_tender_chunks (
    id BIGSERIAL PRIMARY KEY,
    tender_id VARCHAR(255) NOT NULL REFERENCES tenders(id) ON DELETE CASCADE,
    asset_id BIGINT NOT NULL REFERENCES ai_document_assets(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    section_title TEXT,
    content TEXT NOT NULL,
    token_count INTEGER,
    embedding vector(1536),
    embedding_model VARCHAR(80) NOT NULL DEFAULT 'text-embedding-3-small',
    pipeline_version VARCHAR(40) NOT NULL DEFAULT 'pilot-v1',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ai_tender_chunks_index_check CHECK (chunk_index >= 0),
    CONSTRAINT ai_tender_chunks_pages_check CHECK (
        (page_start IS NULL OR page_start >= 1)
        AND (page_end IS NULL OR page_end >= COALESCE(page_start, 1))
    ),
    CONSTRAINT ai_tender_chunks_tokens_check CHECK (token_count IS NULL OR token_count > 0),
    CONSTRAINT ai_tender_chunks_unique UNIQUE (
        asset_id, chunk_index, embedding_model, pipeline_version
    )
);

CREATE INDEX IF NOT EXISTS idx_ai_tender_chunks_tender
    ON ai_tender_chunks (tender_id, asset_id, chunk_index);

CREATE INDEX IF NOT EXISTS idx_ai_tender_chunks_embedding
    ON ai_tender_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE TABLE IF NOT EXISTS ai_tender_profiles (
    tender_id VARCHAR(255) PRIMARY KEY REFERENCES tenders(id) ON DELETE CASCADE,
    source_signature CHAR(64) NOT NULL,
    title TEXT,
    summary TEXT NOT NULL,
    search_text TEXT NOT NULL,
    key_info JSONB NOT NULL DEFAULT '{}'::jsonb,
    document_count SMALLINT NOT NULL DEFAULT 0,
    embedding vector(1536),
    chat_model VARCHAR(80),
    embedding_model VARCHAR(80) NOT NULL DEFAULT 'text-embedding-3-small',
    pipeline_version VARCHAR(40) NOT NULL DEFAULT 'pilot-v1',
    ready_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ai_tender_profiles_document_count_check CHECK (document_count BETWEEN 1 AND 2)
);

CREATE INDEX IF NOT EXISTS idx_ai_tender_profiles_embedding
    ON ai_tender_profiles USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE TABLE IF NOT EXISTS ai_pipeline_events (
    id BIGSERIAL PRIMARY KEY,
    job_id BIGINT REFERENCES ai_ingestion_jobs(id) ON DELETE CASCADE,
    tender_id VARCHAR(255) NOT NULL REFERENCES tenders(id) ON DELETE CASCADE,
    asset_id BIGINT REFERENCES ai_document_assets(id) ON DELETE CASCADE,
    stage VARCHAR(40) NOT NULL,
    status VARCHAR(24) NOT NULL,
    duration_ms INTEGER,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_code VARCHAR(80),
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ai_pipeline_events_status_check CHECK (
        status IN ('started', 'succeeded', 'failed', 'skipped')
    ),
    CONSTRAINT ai_pipeline_events_duration_check CHECK (
        duration_ms IS NULL OR duration_ms >= 0
    )
);

CREATE INDEX IF NOT EXISTS idx_ai_pipeline_events_job
    ON ai_pipeline_events (job_id, created_at);

CREATE INDEX IF NOT EXISTS idx_ai_pipeline_events_tender
    ON ai_pipeline_events (tender_id, created_at DESC);

CREATE OR REPLACE FUNCTION set_ai_pilot_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_ai_ingestion_jobs_updated_at ON ai_ingestion_jobs;
CREATE TRIGGER trg_ai_ingestion_jobs_updated_at
    BEFORE UPDATE ON ai_ingestion_jobs
    FOR EACH ROW EXECUTE FUNCTION set_ai_pilot_updated_at();

DROP TRIGGER IF EXISTS trg_ai_document_assets_updated_at ON ai_document_assets;
CREATE TRIGGER trg_ai_document_assets_updated_at
    BEFORE UPDATE ON ai_document_assets
    FOR EACH ROW EXECUTE FUNCTION set_ai_pilot_updated_at();

DROP TRIGGER IF EXISTS trg_ai_tender_profiles_updated_at ON ai_tender_profiles;
CREATE TRIGGER trg_ai_tender_profiles_updated_at
    BEFORE UPDATE ON ai_tender_profiles
    FOR EACH ROW EXECUTE FUNCTION set_ai_pilot_updated_at();

COMMIT;
