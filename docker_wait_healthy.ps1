param(
    [Parameter(Mandatory = $true)]
    [string]$ContainerName
)

$ErrorActionPreference = 'Stop'
$logProcess = $null

function Get-ContainerState {
    $stateJson = & docker inspect --format '{{json .State}}' $ContainerName 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $stateJson) {
        throw "Cannot inspect container '$ContainerName'."
    }

    return $stateJson | ConvertFrom-Json
}

try {
    # Capture the timestamp before inspecting to avoid missing a health event
    # that occurs between the initial inspection and the event subscription.
    $since = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ss.fffffffZ')
    $state = Get-ContainerState

    if (-not $state.Running) {
        throw "Container '$ContainerName' exited before becoming healthy."
    }
    if ($null -eq $state.Health) {
        throw "Container '$ContainerName' has no healthcheck configured."
    }
    if ($state.Health.Status -eq 'healthy') {
        exit 0
    }
    if ($state.Health.Status -eq 'unhealthy') {
        & docker logs --timestamps --tail 100 $ContainerName
        throw "Container '$ContainerName' is unhealthy."
    }

    Write-Host "Streaming container logs until the health check passes..."
    $logProcess = Start-Process `
        -FilePath 'docker' `
        -ArgumentList @('logs', '--timestamps', '--follow', '--tail', '100', $ContainerName) `
        -NoNewWindow `
        -PassThru

    # Docker publishes health transitions as container events. Select-Object
    # closes the event stream as soon as the first terminal event arrives.
    $terminalEvent = & docker events `
        --since $since `
        --filter "container=$ContainerName" `
        --format '{{.Action}}' |
        Where-Object {
            $_ -in @(
                'health_status: healthy',
                'health_status: unhealthy',
                'die',
                'stop',
                'destroy'
            )
        } |
        Select-Object -First 1

    if ($terminalEvent -eq 'health_status: healthy') {
        $state = Get-ContainerState
        if ($state.Running -and $state.Health.Status -eq 'healthy') {
            exit 0
        }
    }

    if ($terminalEvent) {
        throw "Container '$ContainerName' reported '$terminalEvent'."
    }
    throw "Docker event stream ended before container '$ContainerName' became healthy."
}
catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
finally {
    # Stop only the local log-following client; the container keeps running.
    if ($null -ne $logProcess -and -not $logProcess.HasExited) {
        $logProcess.Kill()
        $logProcess.WaitForExit()
    }
}
