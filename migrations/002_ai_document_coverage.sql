BEGIN;
ALTER TABLE ai_document_assets
    ADD COLUMN IF NOT EXISTS extraction_coverage JSONB NOT NULL DEFAULT '{}'::jsonb;

WITH latest AS (
    SELECT DISTINCT ON (asset_id) asset_id, metrics
    FROM ai_pipeline_events
    WHERE stage = 'extract_embed' AND status = 'succeeded' AND asset_id IS NOT NULL
    ORDER BY asset_id, id DESC
)
UPDATE ai_document_assets a
SET extraction_coverage = jsonb_build_object(
    'pages', latest.metrics->'pages',
    'native_pages', latest.metrics->'native_pages',
    'ocr_pages', latest.metrics->'ocr_pages',
    'skipped_pages', latest.metrics->'skipped_pages',
    'warnings', latest.metrics->'warnings',
    'partial', COALESCE((latest.metrics->>'skipped_pages')::int, 0) > 0
        OR jsonb_array_length(COALESCE(latest.metrics->'warnings', '[]'::jsonb)) > 0
)
FROM latest WHERE a.id = latest.asset_id;

UPDATE ai_tender_profiles p
SET key_info = jsonb_set(key_info, '{document_coverage}', coverage.value)
FROM (
    SELECT tender_id, pipeline_version, jsonb_agg(
        jsonb_build_object('asset_id', id, 'role', document_role, 'coverage', extraction_coverage)
        ORDER BY id
    ) AS value
    FROM ai_document_assets WHERE status = 'ready'
    GROUP BY tender_id, pipeline_version
) coverage
WHERE p.tender_id = coverage.tender_id AND p.pipeline_version = coverage.pipeline_version;
COMMIT;
