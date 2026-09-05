# Licitaciones para probar el chatbot del buscador

Muestra preparada: 46 licitaciones. Para encontrarlas en el buscador, usar
Filtros avanzados > Nomenclatura con el codigo de esta tabla y quitar filtros
de fecha restrictivos. Comprobar el ID en el popup antes de preguntar.

## Pruebas recomendadas

| ID | Consulta |
| --- | --- |
| 1243819 | Que vehiculos y personal se requieren para transportar los medicamentos? |
| 1241981 | Que condiciones piden para el servicio de alimentacion y refrigerios? |
| 1243812 | Que obra se requiere realizar? Cita las paginas. |
| 1244018 | Que requisitos debe acreditar el postor? Separa los factores con puntaje. |

Empezar con "Analiza las bases". El piloto debe responder sin descargar otra
vez los originales. Las siguientes preguntas consultan los fragmentos de esa
licitacion; no buscan en documentos de otros procesos. Donde haya lectura
parcial o referencias internas de archivos comprimidos se mostrara un aviso.

Comprobar cifras y requisitos contra las paginas citadas. El sistema no debe
afirmar que un requisito no existe solo porque no aparece en los fragmentos.

## Lista completa

| ID | Nomenclatura | Lectura parcial |
| --- | --- | --- |
| 1223598 | CP-ABR-2-2026-GRPUNO-DREP-OEC-1 | No |
| 1235196 | LP-ABR-3-2026-MDSR/C-1 | No |
| 1235776 | CP-ABR-13-2026-RED SALUD JAUJA-2 | No |
| 1238303 | DIRECTA-DIRECTA-77-2026-GR.LAMB/PEOT-1 | No |
| 1239671 | COMPRE-COMPRE-9-2026-GRJ-1 | No |
| 1240630 | CP SER-SM-1-2026-GR.CAJ/PROREGI-1 | Si |
| 1241929 | COMPRE-COMPRE-3-2026-GRA - CMFB-1 | Si |
| 1241981 | ABR-PROC-51-2026-OTL / PETROPERU-1 | No |
| 1242325 | LP-ABR-42-2026-MDCC-1 | No |
| 1242360 | LP-ABR-1-2026-MDC/C-1 | No |
| 1242631 | LP-ABR-22-2026-MPH/C-1 | No |
| 1242635 | SIE-SIE-1-2026-OEC-CSJAY/PJ-1 | No |
| 1242675 | LP-ABR-4-2026-INCN-2 | Si |
| 1242700 | CP-ABR-2-2026-MDP/CS-1 | No |
| 1242754 | LP-SM-1-2026-MDA/CS-1 | No |
| 1242808 | CP-ABR-2-2026-MDA/CS-1 | No |
| 1242826 | LP-ABR-9-2026-MDHCC/CS.-1 | No |
| 1243041 | LP-ABR-1-2026-MDP/CS-1 | No |
| 1243089 | LP-ABR-6-2026-MDS/C-1 | No |
| 1243274 | CP-ABR-5-2026-MDM/CS-1 | No |
| 1243283 | COMPRE-COMPRE-12-2026-GRH/OC-1 | No |
| 1243294 | CP-ABR-9-2026-RSH-1 | No |
| 1243328 | LP-ABR-3-2026-MDSL/CS-1 | No |
| 1243351 | LP-ABR-6-2026-CS/MDH-1 | No |
| 1243380 | LP-ABR-1-2026-ESSALUD/RACAJ-1 | No |
| 1243391 | LP-ABR-1-2026-MDLF/CS-1 | No |
| 1243395 | COMPRE-COMPRE-45-2026-OC-MPC-1 | No |
| 1243446 | CP-ABR-41-2026-SUNAT/7Z0010-1 | No |
| 1243462 | LP-ABR-11-2026-CS-MDM-1 | No |
| 1243467 | CONV-PROC-28-2026-UNSA-1 | No |
| 1243473 | LP-ABR-3-2026-MUDIAR-CS-1 | No |
| 1243477 | SIE-SIE-4-2026-RED-SSCH-1 | No |
| 1243479 | CP-ABR-2-2026-EPS SEDAJULIACA S.A.-1 | No |
| 1243488 | LP-ABR-2-2026-MDFOE/CS-1 | No |
| 1243771 | CP-ABR-1-2026-MDP/CS-1 | No |
| 1243812 | LP-SM-26-2026-GRU-GR-C-1 | Si |
| 1243819 | CP-ABR-4-2026-RSDM-1 | No |
| 1243863 | CONV-PROC-42-2026-SCI-BID-1 | No |
| 1243869 | CP-ABR-19-2026-ELECTRO UCAYALI-1 | No |
| 1243909 | LP-ABR-4-2026-MDCH-1 | Si |
| 1243910 | CONV-PROC-1-2026-MCEBS/CI-1 | No |
| 1243937 | LP-SM-1-2026-MDP/CS-1 | No |
| 1244018 | LP-ABR-25-2026-MML-OGA-OL-1 | Si |
| 1244064 | LP-ABR-1-2026-RSVM/CS-1 | No |
| 1244125 | CP CON-SM-1-2026-MDLO/CS-1 | Si |
| 1244129 | CP-ABR-52-2026-MDNCH/C-1 | Si |

No usar como casos de exito del piloto: 1243339, 1243485, 1243490 (tamano),
1242994 (archivos anidados). Estos permanecen en el flujo anterior.

## Activacion y respaldo

- El componente `search/ChatbotHelper` envia `use_document_pilot: true`.
- El proxy mantiene autenticacion y cuotas; consulta `/api/v1/pilot-status`
  antes de exigir un PDF o emitir el ticket de Cloudflare.
- Lite usa `LITE_DOCUMENT_PILOT_ENABLED=true`, version `pilot-v1`, y el
  deployment de embeddings `text-embedding-3-small`, dimension 1536.
- La consulta vectorial esta restringida al ID solicitado, a documentos
  listos y a URLs de origen que siguen coincidiendo con `documents`.
- Si el piloto esta deshabilitado, no hay datos listos o falla la consulta,
  se conserva el flujo anterior. No se dispara ninguna ingesta desde el chat.
- Reversion inmediata: `LITE_DOCUMENT_PILOT_ENABLED=false`. Las imagenes
  anteriores eran Lite `v20` y backend `v74`; no se eliminaron.
- La imagen backend `pilot-chat-v1` hereda `v74` y reemplaza solamente el
  modulo del proxy. Conserva dependencias, runtime y configuracion existente.

## Verificacion del 5 de septiembre

- Lite desplegado: `licigob-ai-lite:pilot-chat-v1`, revision
  `licigob-ai-lite--0000022`, version de servicio `1.0.7`.
- Backend desplegado: `licigob-backend:pilot-chat-v1`.
- Frontend publicado por GitHub Actions; el bundle servido en `www.licigob.pe`
  contiene la activacion `use_document_pilot` y el bypass de preparacion.
- Prueba real del servicio: preparar 1243812 tomo 0,65 s y mostro el aviso de
  lectura parcial. Preguntar por vehiculos y personal en 1243819 tomo 3,38 s,
  devolviendo furgones, camionetas, personal y cita de paginas 81-82.
  Son muestras puntuales con el servicio disponible, no un SLA de latencia.
- 38 pruebas del servicio: 37 aprobadas y una prueba de descarga omitida.
  Cuatro pruebas del proxy y dos pruebas unitarias del frontend aprobadas.
- Navegador Edge headless, APIs simuladas: piloto a 1440 y 390 px sin llamar
  al descargador; caso fuera del piloto conserva llamada al edge; las
  preguntas posteriores pasan al stream. Estas pruebas no usan cuentas
  reales ni consumen cuotas de usuarios.
- Build frontend y compilacion Python aprobadas. El endpoint publico sigue
  rechazando solicitudes anonimas con `401 AUTH_REQUIRED`.
- No se ejecuto otra ingesta ni OCR al consultar el chat. Lite conserva
  0,5 vCPU / 1 GiB; no se crearon nuevos servicios.
