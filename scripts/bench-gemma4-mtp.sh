#!/usr/bin/env bash
# Formal Gemma 4 MTP throughput matrix: {no-MTP, MTP} x {f16 KV, turbo3 KV}.
#
# Launches a fresh llama-server for each cell (no-MTP = head NOT loaded, so it is
# a true baseline with embeddings off), benches it with scripts/bench-gemma4-mtp.py,
# then prints a table with MTP speedup per KV type. Reproducible on any machine;
# point MODEL/HEAD at local GGUFs.
#
# Usage:
#   scripts/bench-gemma4-mtp.sh
# Override via env, e.g.:
#   MODEL=... HEAD=... NGL=99 BLOCK=2 SERVER=./build/bin/llama-server \
#   scripts/bench-gemma4-mtp.sh
set -euo pipefail

SERVER="${SERVER:-./build/bin/llama-server}"          # Windows: ./build/bin/Release/llama-server.exe
MODEL="${MODEL:-.scratch/gemma-4-12b/gemma-4-12b-it-Q4_K_M.gguf}"
HEAD="${HEAD:-.scratch/gemma-4-12b-it-assistant-Q4_K_M.gguf}"
PY="${PY:-python3}"
NGL="${NGL:-99}"
CTX="${CTX:-4096}"
FA="${FA:-on}"
BLOCK="${BLOCK:-2}"            # MTP draft block size B (head proposes B-1 tokens/round)
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
MAXTOK="${MAXTOK:-256}"

declare -A GEN ACC
SRV_PID=""
cleanup() { [[ -n "$SRV_PID" ]] && kill "$SRV_PID" 2>/dev/null || true; }
trap cleanup EXIT

run_cell() {  # $1=kv (f16|turbo3)  $2=mtp (off|on)
  local kv="$1" mtp="$2" label="${1}/${2}-MTP"
  local args=( -m "$MODEL" -c "$CTX" -ngl "$NGL" -ctk "$kv" -ctv "$kv" -fa "$FA"
               --host "$HOST" --port "$PORT" --parallel 1 -np 1 --cont-batching --metrics --jinja )
  if [[ "$mtp" == "on" ]]; then
    args+=( -ngld "$NGL" -ctkd "$kv" -ctvd "$kv" --mtp-head "$HEAD"
            --spec-type draft-mtp --draft-block-size "$BLOCK" )
  fi

  "$SERVER" "${args[@]}" >".bench-${kv}-${mtp}.log" 2>&1 &
  SRV_PID=$!

  # wait for health (max 180s)
  for _ in $(seq 1 180); do
    if curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q 200; then break; fi
    sleep 1
  done

  local out; out="$($PY scripts/bench-gemma4-mtp.py "$PORT" "$label" "$MAXTOK")"
  echo "$out"
  local line; line="$(echo "$out" | grep '^RESULT')"
  GEN["${kv}/${mtp}"]="$(echo "$line" | cut -f3)"
  ACC["${kv}/${mtp}"]="$(echo "$line" | cut -f5)"

  kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; SRV_PID=""
  sleep 1
}

for kv in f16 turbo3; do
  for mtp in off on; do
    run_cell "$kv" "$mtp"
  done
done

sp() { awk -v a="$1" -v b="$2" 'BEGIN{ if (b>0) printf "%.2fx", a/b; else printf "n/a" }'; }
echo
echo "================ Gemma 4 12B MTP matrix (B=$BLOCK, -fa $FA, $MAXTOK tok x3, temp 0) ================"
printf "%-10s %12s %12s %10s %12s\n" "KV" "no-MTP tk/s" "MTP tk/s" "accept%" "MTP speedup"
for kv in f16 turbo3; do
  printf "%-10s %12s %12s %10s %12s\n" "$kv" "${GEN[$kv/off]}" "${GEN[$kv/on]}" "${ACC[$kv/on]:-}" \
    "$(sp "${GEN[$kv/on]}" "${GEN[$kv/off]}")"
done
