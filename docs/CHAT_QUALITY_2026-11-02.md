# Primera mejora del chatbot preparado — 02/11/2026 (America/Lima)

## Alcance y fuentes

Se reutiliza el corpus existente `pilot-v2`: 47 licitaciones, 58 assets ready y
443.782.369 bytes registrados de originales en la consulta inicial. No se inicia
ningun Job, backfill, migracion, OCR ni cambio de embeddings documentales.

`evals/prepared_chat_cases.json` contiene 50 preguntas en 14 familias explicitas
de parafrasis sobre cinco IDs (1231657, 1243819, 1241981, 1244018, 1243812).
46 preguntas tienen anclas factuales; cuatro son negativas. No son 50 hechos
independientes ni una muestra estadistica de toda la BD. Las anclas se contrastaron
con los chunks almacenados, incluyendo paginas y miembro de archivo.

Comprobacion adicional del original: PDF de refrigerios 1241981, paginas fisicas
1-2, mediante el fetcher/proxy ya configurado. Confirma precios unitarios,
subcontratacion no admitida, S/200.000 de experiencia, cinco anos y maximo diez
servicios. La lectura directa devolvia 403 y una consulta al original en Blob
404; no se modificaron permisos ni retencion. La comprobacion limitada del archivo
original de Buena Pro 1231657 no pudo completarse (`unexpected_content`). No se
presenta esta bateria como una auditoria integral de los originales ni asesoria legal.

## Medicion reproducible

```powershell
python scripts/evaluate_prepared_chat.py --live --azure-readonly --label baseline --output evals/results/otra-medicion.json
```

Requiere autorizacion explicita `--live`. Carga configuracion Azure solo en memoria,
fuerza sesiones SQL read-only y limita a 50 recuperaciones/14 respuestas; bloquea
el descargador durante la evaluacion. No usa cuentas ni cuotas de usuarios,
correos o pagos. Se invoca el codigo/prompt/modelo real localmente, con servicios
DB/OpenAI reales. No mide el navegador ni el arranque en frio de Container Apps.
`runtime_image` del informe identifica el entorno del que se leyo la configuracion,
no afirma que las llamadas se hayan hecho al contenedor de esa imagen.

| Comprobacion | Antes | Release |
|---|---:|---:|
| Anclas recuperadas + familia documental correcta | 38/46 | 46/46 |
| Respuestas representativas con anclas esperadas | 12/14 | 14/14 |
| Mediana recuperacion desde estacion local | 3.629 ms | 1.475 ms |
| P95 recuperacion desde estacion local | 4.044 ms | 1.729 ms |

Informes `evals/results/prepared-before.json` y `prepared-release.json`. Se conservan
tambien iteraciones intermedias para no ocultar regresiones: la primera candidata
mejoraba recuperacion pero empeoraba algunas respuestas. Se corrigieron
interpretaciones de consorcios y la reescritura excesiva de preguntas de resultado.
La ultima correccion de historial solo usa preguntas previas del usuario como
contexto de recuperacion; se verifica con regresion unitaria, sin reingesta.

Los checks de anclas y referencias NO equivalen a precision semantica completa.
Una cita permitida no demuestra por si sola que cada afirmacion este sustentada.
Se revisaron respuestas representativas (ganador, experiencia/MYPE, preguntas
mixtas y datos privados). Algunas respuestas aun pueden ser mas extensas de lo
necesario o no citar cada frase; falta un conjunto independiente con revision
experta para certificar exhaustividad y fidelidad de cada afirmacion.
No se promete exactitud del 100% ni un SLA de latencia.

## Cambios

- Intencion mixta consulta bases y Buena Pro con cupos separados; sinonimos de
  adjudicatario y equipamiento. No se mezclan IDs ni versiones.
- Candidatos hibridos, fusion de rankings, vecinos del mismo asset y fragmentos
  limitados por parrafos/oraciones completos. Se declara si hubo omisiones y si
  la pregunta exige una lista exhaustiva que no puede certificarse con fragmentos.
- IDs de evidencia E1/E2 resueltos por codigo a documento/paginas reales. Se
  reemplazan referencias inventadas por advertencias, sin inventar una pagina.
- Pool de hasta cuatro conexiones read-only con rollback al devolverlas y descarte
  de conexiones rotas. Cache acotada de vectores de consulta (128 entradas/300s),
  solo en memoria; no se almacenan respuestas ni se comparten historiales.
- Comprobacion de documento seleccionado mas reciente, ademas del ID/URL. Estados
  de version obsoleta y fallo temporal no disparan el descargador como respaldo.
  Cambios de contenido conservando mismo ID/URL requieren futura comprobacion de
  hash en la ingesta; no se descarga cada documento en cada pregunta.
- Reglas de respuesta conservan periodo, numero maximo de contratos/servicios,
  excepciones MYPE y condiciones de consorcios; no agregan factores de evaluacion
  sin evidencia. Rechazo explicito de datos privados y garantias de adjudicacion.
- Historial de recuperacion usa preguntas del usuario, no afirmaciones generadas
  por el asistente como si fueran evidencia. Historial y prompts acotados.
- Protocolo opt-in `ndjson-v1`: status/meta/content/error/done. Texto plano sigue
  disponible para clientes anteriores. La finalizacion no se deduce de una frase.
- Tiempos por etapa sin registrar preguntas, documentos completos o secretos.

## Validacion y despliegue

87 tests IA aprobados, una prueba de descarga deliberadamente omitida; cuatro
subtests negativos. Backend: 64 tests. Frontend: 12 de permisos y siete de
documentos; ESLint de los JS/JSX modificados y build aprobados.
Edge headless con APIs simuladas y trafico externo bloqueado: exito estructurado,
respuesta truncada, error temporal, historial sin respuestas incompletas, sin
descargas repetidas; regresion del drawer en desktop/movil/tablet y permisos.

Orden operativo: construir/push fuente y nueva imagen Lite, activar revision,
publicar backend compatible, luego frontend. Conservar imagen/revision previas.
No cambiar modelo, dimension 1536, cuotas, minReplicas=0 ni escalado existente.
No se cambia la imagen del Job de ingesta: los chunks actuales se reutilizan.

## Siguiente puerta antes de ampliar

1. Validar un conjunto independiente y preguntas de seguimiento con usuario experto.
2. Auditar las paginas parciales/OCR y contrastar cada cifra ambigua con el original.
3. Solo reprocesar documentos donde falte texto necesario, preservando versiones.
4. Autorizar despues el lote ampliado; no programar una ingesta masiva por defecto.

La separacion de evaluaciones de recuperacion y respuesta sigue
[OpenAI Docs: evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices).
