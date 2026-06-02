#!/usr/bin/env pwsh
# Gemma 4 E4B Edge target + gemma4_assistant MTP draft
# (centroid LM head when use_ordered_embeddings=true).
# Override paths via MAIN_GGUF, DRAFT_GGUF, or LLAMA_SERVER env vars.
#
# MTP_PRESET (ordered-embeddings MTP is heavy on Edge):
#   throughput - block 2, max 6
#   lift       - block 3, max 8
#   balanced   - block 3, max 8
#   quality    - block 4, max 16 (max depth; best when acceptance is high)
# Override with DRAFT_BLOCK_SIZE, DRAFT_MAX. Optional: set LLAMA_MTP_SKIP_STREAK_THRESHOLD=1..32
# to skip drafting after consecutive zero-accept MTP batches (off by default in the binary).

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

# Prefer quantized MTP head when present (see scripts/quantize-gemma4-edge-assistant-mtp.sh).
$DraftQ4  = Join-Path $Root '.scratch\gemma-e4b-assistant-mtp-Q4_K_M.gguf'
$DraftF16 = Join-Path $Root '.scratch\gemma-e4b-assistant-mtp.gguf'

$DraftEnv = [Environment]::GetEnvironmentVariable('DRAFT_GGUF')
if (-not [string]::IsNullOrEmpty($DraftEnv)) {
    $Draft = $DraftEnv
} elseif (Test-Path -LiteralPath $DraftQ4) {
    $Draft = $DraftQ4
} else {
    $Draft = $DraftF16
}

$VerifyAssistantGguf = Env-Or 'VERIFY_ASSISTANT_GGUF' '1'

$MtpPreset = Env-Or 'MTP_PRESET' 'throughput'
switch ($MtpPreset) {
    'throughput' { $DefBlock = '2'; $DefMax = '6' }
    'lift'       { $DefBlock = '3'; $DefMax = '8' }
    'balanced'   { $DefBlock = '3'; $DefMax = '8' }
    'quality'    { $DefBlock = '4'; $DefMax = '16' }
    default {
        Write-Warning "unknown MTP_PRESET=$MtpPreset; using throughput defaults"
        $DefBlock = '2'; $DefMax = '6'
    }
}
$DraftBlockSize = Env-Or 'DRAFT_BLOCK_SIZE' $DefBlock
$DraftMax       = Env-Or 'DRAFT_MAX'        $DefMax
$DraftMin       = Env-Or 'DRAFT_MIN'        '0'

$Ctx            = Env-Or 'CTX'             '16384'
$Ngl            = Env-Or 'NGL'             '99'
$NglDraft       = Env-Or 'NGL_DRAFT'       '99'
$Ctk            = Env-Or 'CTK'             'turbo3'
$Ctv            = Env-Or 'CTV'             'turbo3'
$Ctkd           = Env-Or 'CTKD'            'turbo3'
$Ctvd           = Env-Or 'CTVD'            'turbo3'
$BindHost       = Env-Or 'HOST'            '127.0.0.1'
$Port           = Env-Or 'PORT'            '8080'
$Fa             = Env-Or 'FA'              'on'
$Spec           = Env-Or 'SPEC'            'mtp'
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
if ($Spec -eq 'mtp') {
    if (-not (Test-Path -LiteralPath $Draft)) {
        Write-Host "error: draft (assistant) GGUF not found: $Draft" -ForegroundColor Red
        Write-Host "hint: hf download google/gemma-4-E4B-it-assistant --local-dir $Root\.scratch\gemma-4-E4B-it-assistant"
        Write-Host "hint: `$env:PYTHONPATH='$Root\gguf-py'; python $Root\convert_hf_to_gguf.py $Root\.scratch\gemma-4-E4B-it-assistant --outfile $DraftF16 --outtype f16"
        Write-Host "hint: bash $Root\scripts\quantize-gemma4-edge-assistant-mtp.sh e4b   # then re-run (prefers Q4_K_M draft automatically)"
        exit 1
    }
    if ($VerifyAssistantGguf -ne '0') {
        & python (Join-Path $Root 'scripts\verify-gemma4-assistant-gguf.py') $Draft
        if ($LASTEXITCODE -ne 0) { Write-Error 'assistant GGUF verification failed' }
    }
}

$serverArgs = @(
    '-m', $Main,
    '-c', $Ctx,
    '-ngl', $Ngl,
    '-ngld', $NglDraft,
    '-ctk', $Ctk,
    '-ctv', $Ctv,
    '-ctkd', $Ctkd,
    '-ctvd', $Ctvd,
    '-fa', $Fa,
    '--host', $BindHost,
    '--port', $Port,
    '--parallel', $Parallel,
    '-np', $Parallel,
    '--cont-batching',
    '--reasoning', 'off',
    '--jinja',
    '--chat-template-kwargs', '{\"enable_thinking\": false}'
)

if ($Spec -eq 'mtp') {
    $serverArgs += @(
        '-md', $Draft,
        '--spec-type', 'mtp',
        '--spec-draft-n-max', 2
    )
} else {
    Write-Host "info: speculative decoding disabled (SPEC=$Spec); running baseline" -ForegroundColor Yellow
}

if ($Temp)                       { $serverArgs += @('--temp', $Temp) }
if ($EnableMetrics -ne '0')      { $serverArgs += '--metrics' }
if ($EnableSlots   -ne '0')      { $serverArgs += '--slots' }
if ($LogTimestamps -ne '0')      { $serverArgs += '--log-timestamps' }
if ($LogPrefix     -ne '0')      { $serverArgs += '--log-prefix' }
if ($NoWarmup      -ne '0')      { $serverArgs += '--no-warmup' }
if ($ExtraArgs)                  { $serverArgs += $ExtraArgs }

$skipStreak = [Environment]::GetEnvironmentVariable('LLAMA_MTP_SKIP_STREAK_THRESHOLD')
Write-Host "info: SPEC=$Spec MTP_PRESET=$MtpPreset DRAFT_BLOCK_SIZE=$DraftBlockSize DRAFT_MAX=$DraftMax LLAMA_MTP_SKIP_STREAK_THRESHOLD=$skipStreak" -ForegroundColor Cyan
Write-Host "info: CTX=$Ctx NGL=$Ngl FA=$Fa CTK=$Ctk CTKD=$Ctkd" -ForegroundColor Cyan
Write-Host "info: MAIN=$Main" -ForegroundColor Cyan
Write-Host "info: DRAFT=$Draft" -ForegroundColor Cyan

& $Server @serverArgs
exit $LASTEXITCODE
