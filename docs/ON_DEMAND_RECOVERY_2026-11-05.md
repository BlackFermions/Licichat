# Recuperacion economica del chatbot (05/11/2026)

No se activa ingesta diaria/masiva, OCR, vectorizacion nueva, otro servicio ni mas
replicas. Se conserva min=0/max=1 y el modelo existente. El piloto preparado y las
recomendaciones no cambian.

## Recorridos

1. Ficha: frontend responde con datos visibles, identificados como ficha.
2. Preparado: `/api/v1/document-status` consulta piloto vigente y cache de texto
   on-demand, sin descargar. El proxy reintenta una lectura acotada durante cold start.
3. Bajo demanda: navegador -> Worker `/prepare` -> SEACE -> Lite -> cache de texto
   existente, TTL 14 dias. No se generan embeddings para este recorrido.
4. Si SEACE falla: usuario descarga por su cuenta y elige **Aportar PDF**.
   Backend autentica, revisa cuota sin consumir mensaje y comprueba licitacion.
   Ruta `/api/chatbot/lite/private-document/upload?tender_id=...` transmite PDF
   crudo a `/api/v1/private-document` con user_id de JWT y clave de servicio.
   Limite Flask POR PETICION de 50 MiB solo en esta ruta; global JSON sigue 2 MiB.
   No depende del Worker, de SEACE ni de autorizacion nueva en Cloudflare.

No reintentar el fallo Worker descargando desde Azure, cuya salida ya fue rechazada
en las pruebas del 03/11. No confundir indisponibilidad de cache con documento ausente.

## Sesiones PDF privadas

- PDF <=50 MiB, magic bytes y parser; sin OCR, ZIP ni conversiones.
- Extraccion acotada a limites Lite (120 paginas / 600000 caracteres), aviso parcial.
- Original en temporal local eliminado al acabar el intento; nunca en Blob.
- Texto exclusivamente en memoria, token aleatorio, ligado a usuario y licitacion.
- Caducidad absoluta 30 minutos; acceso caducado rechazado. Limpieza de expirados
  al recibir peticiones. Reinicio/escalado a cero o desalojo por capacidad termina sesion.
- Maximo 12 sesiones/proceso, 2 por usuario; una extraccion simultanea y un intento
  por usuario cada 30 segundos. No son cuotas globales de un despliegue multireplica.
- No escribe tender_documents_text, ai_document_assets, chunks ni otros datos compartidos.
- Token solo en memoria de la UI; historial nuevo al aportar/retirar PDF o cambiar cuenta.
- **Retirar PDF** borra la sesion del propietario sin afectar documentos oficiales.
- Cada respuesta avisa procedencia aportada/no verificada. No certifica que el archivo
  corresponda a la licitacion o a su version vigente. No permite contaminar fuentes compartidas.
- Cuota de mensajes del backend sigue aplicando; user_id se deriva de autenticacion.

Los extractos privados se pasan como mensaje de usuario, nunca dentro del mensaje
de sistema. Se conservan restricciones/citas y no hay herramientas externas del modelo.
Esto reduce, no elimina, el riesgo de prompt injection. Referencia usada:
[OpenAI Docs: untrusted inputs](https://developers.openai.com/api/docs/guides/agent-builder-safety).

## Validacion y publicacion

Pruebas de aislamiento, caducidad, limites, borrado, PDF escaneado/invalido, no escrituras
compartidas, contenido no privilegiado, cuotas y propietarios en proxy, CORS/firma/stream
reintento de estado sin descargar, subida de mas de 2 MiB manteniendo limite global
y navegador simulado de upload/chat/delete.

Orden: Lite -> backend -> frontend. No publicar frontend antes de rutas.
Rollback: revision/imagen anterior de Lite e imagen previa backend. Worker no se modifica:
la sesion Wrangler estaba vencida; los cambios provisionales de Worker se retiraron
antes de publicar. No es necesario reautorizar Cloudflare para la carga privada.
La cache on-demand 1241981 se preparo en prueba acotada real sin modelos ni cuotas de usuario.
No se inicia el Job de ingesta ni se cambia retencion o recursos del piloto.
