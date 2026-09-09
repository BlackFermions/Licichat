targetScope = 'resourceGroup'

@description('Existing Container Apps managed environment.')
param containerAppsEnvironmentName string = 'managedEnvironment-licigobrsg-b354'

@description('Existing StorageV2 account used by LiciGob.')
param storageAccountName string = 'licigobstorage'

@description('Existing Azure Container Registry.')
param containerRegistryName string = 'licigobregistry'

@description('Container Apps Job reserved for the document ingestion worker.')
param ingestionJobName string = 'licigob-ai-ingestion-job'

@description('Private blob container for source documents and normalized text.')
param blobContainerName string = 'licigob-ai-documents'

@description('Temporary image used until the ingestion worker is implemented.')
param bootstrapImage string = 'mcr.microsoft.com/azure-cli:2.76.0'

@description('Create RBAC assignments. Requires Owner or User Access Administrator.')
param createRoleAssignments bool = false

var storageBlobDataContributorRoleId = subscriptionResourceId(
  'Microsoft.Authorization/roleDefinitions',
  'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
)
var acrPullRoleId = subscriptionResourceId(
  'Microsoft.Authorization/roleDefinitions',
  '7f951dda-4ed3-4680-a7ca-43fe172d538d'
)
var commonTags = {
  project: 'licigob'
  component: 'ai-document-ingestion'
  environment: 'pilot'
  sourceBranch: 'feature-ai-document-pilot'
}

resource managedEnvironment 'Microsoft.App/managedEnvironments@2024-03-01' existing = {
  name: containerAppsEnvironmentName
}

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' existing = {
  name: storageAccountName
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' existing = {
  parent: storageAccount
  name: 'default'
}

resource documentContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: blobContainerName
  properties: {
    publicAccess: 'None'
  }
}

resource storageLifecycle 'Microsoft.Storage/storageAccounts/managementPolicies@2023-05-01' = {
  parent: storageAccount
  name: 'default'
  properties: {
    policy: {
      rules: [
        {
          enabled: true
          name: 'delete-ai-pilot-originals-after-30-days'
          type: 'Lifecycle'
          definition: {
            actions: {
              baseBlob: {
                delete: {
                  daysAfterModificationGreaterThan: 30
                }
              }
            }
            filters: {
              blobTypes: [
                'blockBlob'
              ]
              prefixMatch: [
                '${blobContainerName}/originals/'
              ]
            }
          }
        }
      ]
    }
  }
}

resource ingestionJob 'Microsoft.App/jobs@2024-03-01' = {
  name: ingestionJobName
  location: resourceGroup().location
  tags: commonTags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    environmentId: managedEnvironment.id
    configuration: {
      triggerType: 'Manual'
      replicaTimeout: 7200
      replicaRetryLimit: 1
      manualTriggerConfig: {
        parallelism: 1
        replicaCompletionCount: 1
      }
    }
    template: {
      containers: [
        {
          name: 'ingestion'
          image: bootstrapImage
          command: [
            '/bin/sh'
          ]
          args: [
            '-c'
            'echo "The LiciGob ingestion worker image has not been deployed yet."'
          ]
          env: [
            {
              name: 'AI_PIPELINE_VERSION'
              value: 'pilot-v2'
            }
            {
              name: 'AI_STORAGE_PROVIDER'
              value: 'azure_blob'
            }
            {
              name: 'AI_BLOB_ACCOUNT_URL'
              value: storageAccount.properties.primaryEndpoints.blob
            }
            {
              name: 'AI_BLOB_CONTAINER'
              value: blobContainerName
            }
            {
              name: 'AI_JOB_BATCH_SIZE'
              value: '10'
            }
          ]
          resources: {
            cpu: json('2.0')
            memory: '4Gi'
          }
        }
      ]
    }
  }
}

resource storageRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (createRoleAssignments) {
  name: guid(storageAccount.id, ingestionJob.id, storageBlobDataContributorRoleId)
  scope: storageAccount
  properties: {
    principalId: ingestionJob.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: storageBlobDataContributorRoleId
  }
}

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: containerRegistryName
}

resource acrPullRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (createRoleAssignments) {
  name: guid(containerRegistry.id, ingestionJob.id, acrPullRoleId)
  scope: containerRegistry
  properties: {
    principalId: ingestionJob.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: acrPullRoleId
  }
}

output ingestionJobResourceName string = ingestionJob.name
output ingestionJobPrincipalId string = ingestionJob.identity.principalId
output blobContainerResourceName string = documentContainer.name
output storageAccountResourceName string = storageAccount.name
