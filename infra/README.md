# Infraestructura del piloto documental

Este despliegue reutiliza recursos de LiciGob y crea solamente:

- El contenedor Blob privado `licigob-ai-documents` dentro de
  `licigobstorage`.
- El Container Apps Job manual `licigob-ai-ingestion-job` dentro del entorno
  existente.
- Una identidad administrada para el job. Sus permisos de Blob y ACR se
  habilitan por separado con una cuenta autorizada.
- Una regla de ciclo de vida que elimina `originals/` despues de 30 dias.

El job inicial usa una imagen de arranque que no procesa documentos. Esto
reserva y valida la infraestructura sin guardar credenciales de PostgreSQL. La
imagen se reemplazara por `licigobregistry.azurecr.io/licigob-ai-ingestion`
cuando el worker este implementado y probado.

## Validacion y despliegue

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

## Permisos de identidad

La identidad se crea siempre. Las asignaciones RBAC estan desactivadas por
defecto porque requieren una cuenta con rol `Owner` o
`User Access Administrator`. Una cuenta autorizada debe ejecutar despues:

```powershell
az deployment group create `
  --name licigob-ai-document-pilot-rbac `
  --resource-group licigobrsg `
  --template-file infra/ai-document-pilot.bicep `
  --parameters createRoleAssignments=true
```

Esto concede solamente `Storage Blob Data Contributor` sobre
`licigobstorage` y `AcrPull` sobre `licigobregistry` a la identidad del job.
No se deben reemplazar estos permisos por llaves permanentes en variables de
entorno.
