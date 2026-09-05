param(
    [string]$ResourceGroup = "licigobrsg",
    [string]$JobName = "licigob-ai-ingestion-job",
    [ValidateRange(1, 2)][int]$MaxConcurrent = 1,
    [ValidateRange(1, 10)][int]$MaxExecutions = 1
)

$ErrorActionPreference = "Stop"
$requiredSettings = @("DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME")
foreach ($setting in $requiredSettings) {
    if (-not [Environment]::GetEnvironmentVariable($setting)) {
        throw "Missing environment variable: $setting"
    }
}
$previousPassword = $env:PGPASSWORD
$previousTimeout = $env:PGCONNECT_TIMEOUT
$env:PGPASSWORD = $env:DB_PASSWORD
$env:PGCONNECT_TIMEOUT = "15"
$launched = 0
try {
    do {
        $records = az containerapp job execution list -n $JobName -g $ResourceGroup -o json | ConvertFrom-Json
        if ($LASTEXITCODE -ne 0) { throw "Cannot read execution status" }
        $active = @($records | Where-Object {
            $_.properties.status -notin @("Succeeded", "Failed", "Stopped", "Degraded")
        })
        $query = "SELECT count(*) FROM ai_ingestion_jobs WHERE pipeline_version='pilot-v1' AND status='pending' AND available_at<=NOW();"
        $pendingOutput = & psql -h $env:DB_HOST -U $env:DB_USER -d $env:DB_NAME -v ON_ERROR_STOP=1 -At -c $query
        if ($LASTEXITCODE -ne 0) { throw "Cannot inspect the ingestion queue" }
        $pending = [int]$pendingOutput
        Write-Output ("active={0} pending={1} launched={2}/{3}" -f $active.Count, $pending, $launched, $MaxExecutions)
        if ($pending -gt 0 -and $active.Count -lt $MaxConcurrent -and $launched -lt $MaxExecutions) {
            $execution = az containerapp job start -n $JobName -g $ResourceGroup --query name -o tsv
            if ($LASTEXITCODE -ne 0) { throw "Cannot start the next execution" }
            $launched++
            Write-Output ("STARTED " + $execution)
        } elseif ($active.Count -eq 0) {
            break
        }
        Start-Sleep -Seconds 30
    } while ($true)
    Write-Output "Executions finished. Check scripts/pilot_status.sql for individual failures or pending jobs."
} finally {
    $env:PGPASSWORD = $previousPassword
    $env:PGCONNECT_TIMEOUT = $previousTimeout
}
