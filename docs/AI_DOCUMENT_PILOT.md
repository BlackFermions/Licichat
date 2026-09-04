# Piloto de inteligencia documental de LiciGob

## Objetivo

Validar durante una semana de licitaciones que LiciGob puede preparar documentos
por adelantado y reutilizarlos en chatbot, recomendaciones y busqueda hibrida.
El piloto no modifica ni reemplaza los recorridos actuales de produccion.

## Alcance

- Seleccionar licitaciones por `tenders.first_seen_at` dentro de un intervalo
  cerrado-abierto: `inicio <= first_seen_at < fin`.
- Procesar como maximo una version de bases por licitacion:
  1. La ultima `Bases Integradas`, si existe.
  2. En su defecto, la ultima `Bases Administrativas`.
- Incorporar el ultimo `Documentos de Otorgamiento de Buena Pro` cuando exista.
- Ejecutar primero lotes de 50, luego 200 y finalmente el resto de la semana.
- Mantener el chatbot Lite actual como fallback.

No se incluye el backfill historico, el reemplazo del buscador actual ni la
eliminacion de tablas o servicios existentes.

## Arquitectura del piloto

```text
Importador remoto de SEACE
        |
        | metadatos y URLs
        v
PostgreSQL: ai_ingestion_jobs
        |
        v
licigob-ai-ingestion-job
        |-- descarga por el proxy Cloudflare existente
        |-- inspeccion segura de PDF/ZIP/RAR/7z
        |-- extraccion nativa con PyMuPDF
        |-- OCR Tesseract solo en paginas deficientes
        |-- secciones, paginas y chunks
        |-- embeddings con Azure OpenAI
        |
        +--> Azure Blob: originales temporales y Markdown
        +--> PostgreSQL/pgvector: chunks, ficha y estados
                           |
                           v
                    licigob-ai-lite
```

MinIO seguira siendo una implementacion valida para desarrollo local. En Azure,
la implementacion recomendada de `ObjectStorage` sera Blob Storage para evitar
operar una VM de almacenamiento.

## Recursos

Se reutilizan:

- PostgreSQL Azure y la extension `vector`.
- Azure OpenAI.
- `licigob-ai-lite`.
- Cloudflare Worker de documentos SEACE.
- Azure Container Registry `licigobregistry`.

Recurso nuevo recomendado:

- Azure Container Apps Job: `licigob-ai-ingestion-job`.
- Contenedor privado de Blob Storage: `licigob-ai-documents`. Puede residir en
  una Storage Account existente.

No se necesita otro PostgreSQL, Redis, API, frontend ni MinIO en produccion.

## Almacenamiento

Claves propuestas:

```text
originals/{tender_id}/{source_document_id}/{sha256}/{filename}
markdown/{tender_id}/{asset_id}/{pipeline_version}.md
```

Los originales se conservaran 30 dias durante el piloto. El Markdown y los
registros vectoriales se conservaran para medir busqueda y calidad. El job usa
disco efimero y elimina temporales al terminar.

## Estados

Un trabajo pasa por:

```text
pending -> processing -> ready
                    \-> retry -> processing
                    \-> failed
                    \-> skipped
```

Cada trabajo contiene la seleccion documental y una firma SHA-256. Esto permite
reintentar sin duplicar resultados y volver a procesar cuando cambie una URL,
un documento o la version del pipeline.

## Seguridad documental

- Permitir descargas solo desde hosts y rutas SEACE aprobados.
- Validar formato mediante magic bytes, no solo extension.
- Limitar bytes descargados, bytes descomprimidos, cantidad de archivos,
  profundidad y ratio de compresion.
- Bloquear rutas absolutas, `..`, ejecutables, enlaces y archivos anidados no
  permitidos.
- No exponer Blob Storage al navegador; usar identidad administrada o SAS de
  corta duracion solo desde servicios internos.
- Nunca registrar contenido completo, URLs firmadas ni credenciales.

## Estrategia de extraccion

1. Descargar el documento completo seleccionado.
2. Si es un archivo comprimido, extraer solamente PDF/DOCX relevantes.
3. Extraer todas las paginas con texto nativo.
4. Aplicar OCR solo a paginas sin texto suficiente.
5. Conservar documento, pagina y seccion en cada chunk.
6. Dividir por estructura, con objetivo de 700 a 1,000 tokens y solapamiento de
   100 tokens.
7. Generar un embedding por chunk y un embedding consolidado por licitacion.

## Ejecucion inicial

La migracion es aditiva y no se ejecuta automaticamente:

```powershell
psql "$env:DATABASE_URL" -f migrations/001_ai_document_pilot.sql
```

El sembrador es `dry-run` por defecto:

```powershell
python scripts/seed_pilot_jobs.py --start 2026-08-24 --end 2026-08-31
python scripts/seed_pilot_jobs.py --start 2026-08-24 --end 2026-08-31 --limit 50 --apply
```

`--end` es exclusivo. Antes de usar `--apply`, deben revisarse el conteo y la
muestra mostrados por el primer comando.

La primera ejecucion recomendada es exactamente el lote de 50. El lote de 200
reutiliza la misma seleccion; la restriccion de idempotencia conserva los 50
anteriores e inserta solamente los 150 siguientes. El procesamiento completo de
la semana se habilita despues de revisar errores, tiempos, OCR y gasto de esos
dos lotes.

## Criterios de exito

- Al menos 90% de documentos seleccionados termina en `ready` o en un estado
  explicable como `skipped`.
- Ningun archivo se procesa dos veces con la misma firma y version.
- Las respuestas citan documento y pagina.
- Se registra tiempo, paginas OCR, bytes, tokens y errores por etapa.
- La importacion diaria no espera descargas, OCR ni OpenAI.
- El chatbot actual sigue funcionando si el piloto esta apagado.
