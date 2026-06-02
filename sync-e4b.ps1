#!/usr/bin/env pwsh
# Sync e4b-assistant with upstream stacks.
#
# Stack (bottom-up):
#   ggml-org/llama.cpp        (origin/master, mainline llama.cpp — very active)
#     -> TheTom/llama-cpp-turboquant       (turbo/master, TurboQuant KV/weight infra)
#       -> BoFan/...MTP-TurboQuant         (bofan/merge-mtp-turboquant — MTP + TurboQuant
#                                            + Gemma 4 E4B/assistant/vision + multimodal MTP fix)
#         -> cortexist e4b-assistant       (our branch)
#
# Strategy: pull bofan first (the upstream we're directly forked from as of
# 2026-06-01). BoFan tracks mainline llama.cpp via merges, so syncing bofan
# transitively brings in mainline updates. Use -Mainline/-Turbo only when
# bofan is stale relative to what we want, or to apply a specific upstream
# fix before bofan absorbs it.
#
# Historical: e4b-assistant was originally based on atomic/feature/gemma-mtp
# until 2026-06-01. Atomic was inactive and lacked the multimodal MTP fix;
# we reset to bofan/merge-mtp-turboquant. The `atomic` remote remains
# available for reference / comparison via `git log atomic/feature/gemma-mtp`.

param(
    [switch]$DryRun,
    [switch]$Mainline,   # also merge origin/master directly (use if BoFan is stale)
    [switch]$Turbo,      # also merge turbo/master directly
    [switch]$Push
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

function Run($cmd) {
    Write-Host "+ $cmd" -ForegroundColor Cyan
    if (-not $DryRun) { Invoke-Expression $cmd; if ($LASTEXITCODE) { throw "Failed: $cmd" } }
}

# Safety
$branch = git branch --show-current
if ($branch -ne 'e4b-assistant') { throw "Expected branch 'e4b-assistant', got '$branch'. Checkout first." }
if (git status --porcelain --untracked-files=no) { throw "Working tree has uncommitted tracked changes. Commit or stash first." }

Run 'git fetch --all --prune'

Write-Host "`n=== Position before sync ===" -ForegroundColor Yellow
git log --oneline -1 HEAD
Write-Host "bofan/merge-mtp-turboquant: " -NoNewline; git log --oneline -1 bofan/merge-mtp-turboquant
Write-Host "turbo/master:               " -NoNewline; git log --oneline -1 turbo/master
Write-Host "origin/master (mainline):   " -NoNewline; git log --oneline -1 origin/master

$behindBofan = (git rev-list --count HEAD..bofan/merge-mtp-turboquant).Trim()
Write-Host "`nBehind bofan/merge-mtp-turboquant by $behindBofan commits."

if ([int]$behindBofan -gt 0) {
    Run 'git merge --no-edit bofan/merge-mtp-turboquant'
} else {
    Write-Host 'Already current with bofan.' -ForegroundColor Green
}

if ($Turbo) {
    $behindTurbo = (git rev-list --count HEAD..turbo/master).Trim()
    Write-Host "`nBehind turbo/master by $behindTurbo commits."
    if ([int]$behindTurbo -gt 0) { Run 'git merge --no-edit turbo/master' }
}

if ($Mainline) {
    $behindMain = (git rev-list --count HEAD..origin/master).Trim()
    Write-Host "`nBehind origin/master (llama.cpp mainline) by $behindMain commits."
    if ([int]$behindMain -gt 0) {
        Write-Warning 'Merging mainline directly. Expect conflicts in ggml/, src/llama-*, and KV cache code.'
        Run 'git merge --no-edit origin/master'
    }
}

if ($Push -and -not $DryRun) { Run 'git push' }

Write-Host "`n=== Done ===" -ForegroundColor Green
Write-Host 'Next: rebuild + run E4B assistant smoke test (scripts/bench-one-shot.ps1) before pushing if you skipped -Push.'
