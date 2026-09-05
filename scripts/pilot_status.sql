SELECT status, count(*) AS jobs,
       count(*) FILTER (WHERE available_at = 'infinity'::timestamptz) AS awaiting_staging
FROM ai_ingestion_jobs WHERE pipeline_version = 'pilot-v1'
GROUP BY status ORDER BY status;

SELECT count(*) AS ready_documents, sum(page_count) AS pages_reviewed,
       sum(native_text_pages) AS native_pages, sum(ocr_pages) AS ocr_pages,
       sum((extraction_coverage->>'skipped_pages')::int) AS skipped_pages,
       count(*) FILTER (WHERE extraction_coverage->>'partial' = 'true') AS partial_documents,
       round(sum(byte_size) / 1048576.0, 2) AS original_mib
FROM ai_document_assets WHERE pipeline_version = 'pilot-v1' AND status = 'ready';

SELECT count(*) AS chunks, count(*) FILTER (WHERE embedding IS NOT NULL) AS vectors,
       (SELECT count(*) FROM ai_tender_profiles WHERE pipeline_version='pilot-v1') AS profiles
FROM ai_tender_chunks WHERE pipeline_version = 'pilot-v1';

SELECT tender_id, status, last_error_code
FROM ai_ingestion_jobs WHERE pipeline_version='pilot-v1' AND status IN ('retry', 'failed');

SELECT sum(duration_ms)/1000.0 AS successful_processing_seconds,
       sum((metrics->'usage'->>'embedding_tokens')::bigint) AS metered_embedding_tokens,
       sum((metrics->'usage'->>'chat_input_tokens')::bigint) AS metered_chat_input_tokens,
       sum((metrics->'usage'->>'chat_output_tokens')::bigint) AS metered_chat_output_tokens,
       count(*) FILTER (WHERE metrics ? 'usage') AS metered_jobs,
       count(*) AS successful_jobs
FROM ai_pipeline_events WHERE stage = 'job' AND status = 'succeeded';
