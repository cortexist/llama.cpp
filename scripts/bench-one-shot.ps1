#!/usr/bin/env pwsh
# Hit a running llama-server with one fixed prompt, fixed token count,
# greedy sampling. Prints decode tk/s and (for MTP) accept rate.
#
# Usage: .\scripts\bench-one-shot.ps1 [-Port 8080] [-MaxTokens 256] [-Prompt "..."]

[CmdletBinding()]
param(
    [int]    $Port      = 8080,
    [int]    $MaxTokens = 256,
    [string] $Prompt    = 'Write a clear, single-paragraph explanation of how a transformer language model produces one output token, step by step. Be precise.'
)

$ErrorActionPreference = 'Stop'

$body = @{
    model        = 'local'
    messages     = @(@{ role = 'user'; content = $Prompt })
    max_tokens   = $MaxTokens
    temperature  = 0
    top_k        = 1
    stream       = $false
    cache_prompt = $false
    timings_per_token = $false
} | ConvertTo-Json -Depth 5 -Compress

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$resp = Invoke-RestMethod -Method Post `
    -Uri "http://127.0.0.1:$Port/v1/chat/completions" `
    -ContentType 'application/json' `
    -Body $body
$sw.Stop()

$text = $resp.choices[0].message.content
$t    = $resp.timings   # server returns timings block when present

Write-Host "`n--- response (first 400 chars) ---" -ForegroundColor Cyan
Write-Host ($text.Substring(0, [Math]::Min(400, $text.Length)))
Write-Host "`n--- timing ---" -ForegroundColor Cyan
Write-Host ("wall            : {0,8:N2} s" -f ($sw.Elapsed.TotalSeconds))

if ($t) {
    Write-Host ("prompt tokens   : {0,8} ({1,7:N2} tk/s)" -f $t.prompt_n, $t.prompt_per_second)
    Write-Host ("decode tokens   : {0,8} ({1,7:N2} tk/s)" -f $t.predicted_n, $t.predicted_per_second)
    if ($t.PSObject.Properties.Name -contains 'draft_n') {
        Write-Host ("drafted         : {0,8}" -f $t.draft_n)
        Write-Host ("accepted        : {0,8}" -f $t.draft_n_accepted)
        if ($t.draft_n -gt 0) {
            $rate = 100.0 * $t.draft_n_accepted / $t.draft_n
            Write-Host ("accept rate     : {0,8:N1} %" -f $rate)
        }
    } else {
        Write-Host "(no draft/accept fields - server is not running MTP, or this build doesn't surface them)" -ForegroundColor Yellow
    }
} else {
    Write-Host "(server returned no 'timings' block; check server stderr for 'n_decoded'/'n_accept')" -ForegroundColor Yellow
}
Write-Host ("usage           : prompt={0} completion={1} total={2}" -f `
    $resp.usage.prompt_tokens, $resp.usage.completion_tokens, $resp.usage.total_tokens)
