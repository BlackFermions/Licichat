# Infraestructura del piloto documental

Este despliegue reutiliza recursos de LiciGob y crea solamente:

- El contenedor Blob privado `licigob-ai-documents` dentro de
  `licigobstorage`.
- El Container Apps Job manual `licigob-ai-ingestion-job` dentro del entorno
  existente.
- Una identidad administrada para el job. Sus permisos de Blob y ACR se
  habilitan por separado con una cuenta autorizada.
- Una regla de ciclo de vida que elimina `originals/` despues de 30 dias.

La plantilla conserva una imagen de arranque para crear infraestructura sin
guardar credenciales de PostgreSQL. La instancia desplegada usa actualmente
`licigobregistry.azurecr.io/licigob-ai-ingestion:pilot-v1.4`; sus secretos y
variables se configuraron fuera de Bicep para que no entren al repositorio ni al
historial de despliegues.

## Validacion y despliegue

La plantilla Bicep es solo para el alta inicial. Reaplicarla sobre el Job
configurado reemplazaria su imagen y variables por el bootstrap. Para actualizar
el worker existente, construir una etiqueta nueva en ACR y ejecutar
`az containerapp job update --image <imagen>`; revisar primero el estado de las
ejecuciones.

```powershell
az deployment group what-if `
  --resource-group licigobrsg `
  --template-file infra/ai-document-pilot.bicep

az deployment group create `
  --name licigob-ai-document-pilot `
  --resource-group licigobrsg `
  --template-file infra/ai-document-pilot.bicep
```

No se debe iniciar manualmente el job hasta que tenga la imagen definitiva y
las variables protegidas requeridas. Este despliegue tampoco ejecuta la
migracion PostgreSQL.

El Job tiene 2 vCPU, 4 GiB, disparador manual y lote maximo de 10. Los trabajos
solo quedan disponibles despues de que `scripts/stage_pilot_documents.py`
guarda todos sus originales en Blob. Esto evita depender de descargas SEACE
desde una IP de centro de datos.

## Permisos de identidad

La identidad se crea siempre. Las asignaciones RBAC estan desactivadas por
defecto porque requieren autorizacion RBAC. En el Job ya configurado, usar
asignaciones directas para preservar imagen, secretos y variables:

```powershell
$principalId = az containerapp job show -n licigob-ai-ingestion-job -g licigobrsg --query identity.principalId -o tsv
$storageScope = az storage account show -n licigobstorage -g licigobrsg --query id -o tsv
$registryScope = az acr show -n licigobregistry -g licigobrsg --query id -o tsv
az role assignment create --assignee-object-id $principalId --assignee-principal-type ServicePrincipal --role "Storage Blob Data Contributor" --scope $storageScope
az role assignment create --assignee-object-id $principalId --assignee-principal-type ServicePrincipal --role AcrPull --scope $registryScope
```

Esto concede solamente `Storage Blob Data Contributor` sobre
`licigobstorage` y `AcrPull` sobre `licigobregistry` a la identidad del job.
No se deben reemplazar estos permisos por llaves permanentes en variables de
entorno.
