#!/usr/bin/env pwsh
# Gemma 4 E4B Edge target only (no MTP / no assistant draft).
# Override MAIN_GGUF or LLAMA_SERVER via env vars, or any flag via -ExtraArgs.

[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments=$true)] [string[]] $ExtraArgs)

$ErrorActionPreference = 'Stop'

function Env-Or($name, $default) {
    $v = [Environment]::GetEnvironmentVariable($name)
    if ([string]::IsNullOrEmpty($v)) { $default } else { $v }
}

$Root   = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Server = Env-Or 'LLAMA_SERVER' (Join-Path $Root 'build\bin\llama-server.exe')
$Main   = Env-Or 'MAIN_GGUF'    (Join-Path $Root '.scratch\gemma-4-e4b\gemma-4-E4B-it-Q4_K_M.gguf')

$Ctx            = Env-Or 'CTX'             '16384'
$Ngl            = Env-Or 'NGL'             '99'
$Ctk            = Env-Or 'CTK'             'turbo3'
$Ctv            = Env-Or 'CTV'             'turbo3'
$BindHost       = Env-Or 'HOST'            '127.0.0.1'
$Port           = Env-Or 'PORT'            '8080'
$Fa             = Env-Or 'FA'              'on'
$Temp           = Env-Or 'LLAMA_TEMP'      ''   # NOT 'TEMP' — Windows pre-sets that to a path
$EnableMetrics  = Env-Or 'ENABLE_METRICS'  '1'
$EnableSlots    = Env-Or 'ENABLE_SLOTS'    '1'
$LogTimestamps  = Env-Or 'LOG_TIMESTAMPS'  '1'
$LogPrefix      = Env-Or 'LOG_PREFIX'      '1'
$NoWarmup       = Env-Or 'NO_WARMUP'       '0'
$Parallel       = Env-Or 'PARALLEL'        '1'

if (-not (Test-Path -LiteralPath $Server)) {
    Write-Error "missing $Server (build with: cmake --build build --target llama-server)"
}
if (-not (Test-Path -LiteralPath $Main)) {
    Write-Error "main GGUF not found: $Main"
}

$serverArgs = @(
    '-m', $Main,
    '-c', $Ctx,
    '-ngl', $Ngl,
    '-ctk', $Ctk,
    '-ctv', $Ctv,
    '-fa', $Fa,
    '--host', $BindHost,
    '--port', $Port,
    '--parallel', $Parallel,
    '-np', $Parallel,
    '--cont-batching',
    '--reasoning', 'off',
    '--chat-template-kwargs', '{"enable_thinking": false}'
)

if ($Temp)                       { $serverArgs += @('--temp', $Temp) }
if ($EnableMetrics -ne '0')      { $serverArgs += '--metrics' }
if ($EnableSlots   -ne '0')      { $serverArgs += '--slots' }
if ($LogTimestamps -ne '0')      { $serverArgs += '--log-timestamps' }
if ($LogPrefix     -ne '0')      { $serverArgs += '--log-prefix' }
if ($NoWarmup      -ne '0')      { $serverArgs += '--no-warmup' }
if ($ExtraArgs)                  { $serverArgs += $ExtraArgs }

Write-Host "info: baseline E4B (no MTP) CTX=$Ctx NGL=$Ngl FA=$Fa CTK=$Ctk" -ForegroundColor Cyan
Write-Host "info: MAIN=$Main" -ForegroundColor Cyan

& $Server @serverArgs
exit $LASTEXITCODE
