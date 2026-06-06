#!/usr/bin/env python3
"""Formal MTP throughput bench client for Gemma 4 (target + assistant head).

Hits an already-running llama-server and measures generation throughput on a
fixed prompt set. Deterministic by construction: temperature 0 + ignore_eos +
fixed max_tokens => every config generates the exact same number of tokens, so
tk/s differences reflect compute cost only (not output-length variance). One
warm-up request (graph capture / first alloc) is excluded from the timing.

Reports generation tk/s, prompt-eval tk/s, and (when a draft/MTP head is loaded)
the MTP acceptance rate. Speedup is computed by the driver by comparing the
no-MTP and MTP runs at the SAME KV-cache type.

Usage: bench-gemma4-mtp.py <port> <label> [max_tokens]
Emits a machine-readable trailer line: RESULT\t<label>\t<gen_tps>\t<pp_tps>\t<accept|>
"""
import sys, json, urllib.request

PORT   = int(sys.argv[1])
LABEL  = sys.argv[2]
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 256

# Mixed workload: explanatory prose, code, and long-form history — acceptance
# (and therefore MTP payoff) is workload-dependent, so we average across types.
PROMPTS = [
    "Explain in detail how a modern CPU executes a single instruction, from fetch to retire.",
    "Write a complete, well-commented Python implementation of a binary search tree supporting insert, search, in-order traversal, and delete.",
    "Describe the main causes and lasting consequences of the Industrial Revolution across several paragraphs.",
]

def call(prompt):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": MAXTOK,
        "ignore_eos": True,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)

call("Say hello.")  # warm-up, not timed

pn = pms = qn = qms = 0.0
dn = da = 0
for p in PROMPTS:
    t = call(p).get("timings", {})
    pn  += t.get("predicted_n", 0);   pms += t.get("predicted_ms", 0)
    qn  += t.get("prompt_n", 0);      qms += t.get("prompt_ms", 0)
    dn  += t.get("draft_n", 0) or 0;  da  += t.get("draft_n_accepted", 0) or 0

gen = 1000.0 * pn / pms if pms else 0.0
pp  = 1000.0 * qn / qms if qms else 0.0
acc = (100.0 * da / dn) if dn else None

print(f"\n=== {LABEL} ===")
print(f"gen tokens      : {int(pn)}")
print(f"generation tk/s : {gen:.2f}")
print(f"prompt eval tk/s: {pp:.1f}")
print(f"MTP acceptance  : {acc:.1f}%  (draft_n={dn}, accepted={da})" if acc is not None else "MTP             : (none)")
print(f"RESULT\t{LABEL}\t{gen:.2f}\t{pp:.1f}\t{acc if acc is not None else ''}")
