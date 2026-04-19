$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$asciiRepoRoot = "C:\Tez"

if (-not (Test-Path $asciiRepoRoot)) {
    New-Item -ItemType Junction -Path $asciiRepoRoot -Target $repoRoot | Out-Null
}

if (Test-Path $asciiRepoRoot) {
    $repoRoot = $asciiRepoRoot
}

$pythonExe = Join-Path $repoRoot ".venv311\Scripts\python.exe"
$launcher = Join-Path $repoRoot "tools\pybullet_mcp_stdio.py"
$serverName = "pybullet-local"

codex mcp get $serverName *> $null
if ($LASTEXITCODE -eq 0) {
    codex mcp remove $serverName | Out-Null
}

codex mcp add $serverName -- $pythonExe $launcher
Write-Host ""
Write-Host "PyBullet MCP registered as '$serverName'."
Write-Host "If the current Codex session does not refresh the tool list immediately, restart Codex once."
