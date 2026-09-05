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
        | staging secuencial desde una red aceptada por SEACE
        v
Azure Blob: originales privados
        |
        v
licigob-ai-ingestion-job
        |-- recuperacion del original desde Blob
        |-- inspeccion segura de PDF/ZIP/RAR/7z
        |-- extraccion nativa con PyMuPDF
        |-- OCR Tesseract solo en paginas deficientes
        |-- secciones, paginas y chunks
        |-- embeddings con Azure OpenAI
        |
        +--> Azure Blob: Markdown normalizado
        +--> PostgreSQL/pgvector: chunks, ficha y estados
                           |
                           v
                    Prueba de recuperacion semantica
                           |
                           v (integracion pendiente)
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

Recursos del piloto desplegados:

- Azure Container Apps Job: `licigob-ai-ingestion-job`.
- Contenedor privado de Blob Storage: `licigob-ai-documents`. Puede residir en
  la Storage Account existente `licigobstorage`.

No se necesita otro PostgreSQL, Redis, API, frontend ni MinIO en produccion.

SEACE bloquea de forma intermitente las salidas de centros de datos. Por eso el
Job no reclama un trabajo hasta que todos sus originales hayan sido guardados
en Blob. El staging es liviano: descarga y transmite archivos de uno en uno; no
ejecuta OCR, embeddings ni mantiene los documentos en memoria.

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
3. Extraer texto nativo hasta el limite de 300 paginas.
4. Aplicar OCR solo a paginas sin texto suficiente, hasta 60 paginas por PDF.
5. Conservar documento, pagina y seccion en cada chunk.
6. Dividir por estructura, con objetivo de 700 a 1,000 tokens y solapamiento de
   100 tokens.
7. Generar un embedding por chunk y un embedding consolidado por licitacion.

## Ejecucion inicial

La migracion es aditiva y no se ejecuta automaticamente:

```powershell
psql "$env:DATABASE_URL" -f migrations/001_ai_document_pilot.sql
psql "$env:DATABASE_URL" -f migrations/002_ai_document_coverage.sql
```

El sembrador es `dry-run` por defecto:

```powershell
python scripts/seed_pilot_jobs.py --start 2026-08-24 --end 2026-08-31
python scripts/seed_pilot_jobs.py --start 2026-08-24 --end 2026-08-31 --limit 50 --apply
```

`--end` es exclusivo. Antes de usar `--apply`, deben revisarse el conteo y la
muestra mostrados por el primer comando.

Los trabajos sembrados quedan con `available_at = infinity` para impedir que el
Job intente descargar desde Azure. Tras configurar las variables de PostgreSQL,
Blob y el proxy, el staging habilita solamente los trabajos completos:

```powershell
python -m scripts.stage_pilot_documents --limit 10
az containerapp job start `
  --name licigob-ai-ingestion-job `
  --resource-group licigobrsg
```

En el piloto se ejecutan grupos de 10. El staging puede correr en la misma
maquina del importador o en una estacion operativa ubicada en una red aceptada
por SEACE; su uso de CPU y RAM es pequeno frente al OCR, que permanece en Azure.

Para staging basta Python 3.11 con `requirements-staging.txt`. Configurar
`DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `AI_BLOB_ACCOUNT_URL` y
`AI_BLOB_CONTAINER`, junto con una identidad que pueda escribir en ese contenedor.
El Job utiliza identidad administrada para Blob. El modo alternativo
`AI_BLOB_CONNECTION_STRING` es exclusivamente para un entorno operativo seguro;
no almacenarlo en Git. `SEACE_PROXY_FIRST=false` permite intentar SEACE directamente.

El staging probado en este piloto se ejecuto desde la estacion local. Su
instalacion y programacion en el servidor remoto aun estan pendientes.

## Cobertura y consulta

`ready` significa que hay texto y vectores consultables; no garantiza que se
haya leido el PDF completo. `ai_document_assets.extraction_coverage` y
`ai_tender_profiles.key_info.document_coverage` indican paginas omitidas,
advertencias y lectura parcial. Los archivos que superan 50 MiB quedan fuera.
En DOCX los numeros de pagina son logicos; no equivalen a paginacion de un PDF.

Consultar estados y consumo medido:

```powershell
psql "$env:DATABASE_URL" -f scripts/pilot_status.sql
python -m scripts.check_pilot_retrieval "Transporte de medicamentos" --limit 3
```

La consulta usa un embedding y PostgreSQL en modo solo lectura. Requiere las
variables `OPENAI_API_BASE` y `OPENAI_API_KEY`. Devuelve fragmentos, documento,
paginas y similitud; no llama al modelo de chat. Los tokens se registran para
trabajos ejecutados desde la imagen `pilot-v1.4`; los primeros 11 trabajos no
tienen medicion completa de tokens.

La ficha generada utiliza hasta 55.000 caracteres iniciales de los fragmentos.
Debe evaluarse su cobertura de requisitos y Buena Pro antes de usarla en
recomendaciones publicas; la recuperacion vectorial consulta todos los
fragmentos almacenados.

La primera ejecucion recomendada es exactamente el lote de 50. El lote de 200
reutiliza la misma seleccion; la restriccion de idempotencia conserva los 50
anteriores e inserta solamente los 150 siguientes. El procesamiento completo de
la semana se habilita despues de revisar errores, tiempos, OCR y gasto de esos
dos lotes.

Para ejecutar varios lotes ya preparados, con limites explicitos:

```powershell
./scripts/run_pilot_batches.ps1 -MaxExecutions 4 -MaxConcurrent 2
```

Cada ejecucion conserva 2 vCPU y 4 GiB. Con dos ejecuciones simultaneas el pico
es de 4 vCPU y 8 GiB en Azure. El script no reintenta automaticamente trabajos
fallidos ni crea una programacion permanente.

## Criterios de exito

- Al menos 90% de documentos seleccionados termina en `ready` o en un estado
  explicable como `skipped`.
- Ningun archivo se procesa dos veces con la misma firma y version.
- Las respuestas citan documento y pagina.
- Se registra tiempo, paginas OCR, bytes, tokens y errores por etapa.
- La importacion diaria no espera descargas, OCR ni OpenAI.
- El chatbot actual sigue funcionando si el piloto esta apagado.
