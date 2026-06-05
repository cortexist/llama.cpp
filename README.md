<p align="center">
  <img src="media/cortexist-fork.png" alt="Cortexist llama.cpp fork" width="100%">
</p>

<h1 align="center">Cortexist llama.cpp</h1>
<p align="center"><b>Gemma 4 E4B / E2B MTP speculative decoding — <i>with multimodal</i> — on top of TurboQuant.</b></p>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Buy Me A Coffee](https://img.shields.io/badge/-buy_me_a_coffee-gray?logo=buy-me-a-coffee)](https://www.buymeacoffee.com/cortexist)

This is a downstream fork of [`llama.cpp`](https://github.com/ggml-org/llama.cpp). It exists for **one specific job**: running **Gemma 4 E4B / E2B** with the **`gemma4_assistant` MTP draft head** *and* **multimodal inputs** (vision + audio via `--mmproj`), accelerated by **TurboQuant** KV-cache/weight compression — fast enough to be useful on edge devices like the **Jetson Orin NX**.

If that exact combination isn't what you need, one of the sibling forks below is a better fit.

---

## When to Use This Fork?

Pick the smallest fork that covers what you actually need — each adds scope (and surface area) on top of the previous one.

| You need… | Use | Repo |
|---|---|---|
| Gemma 4 MTP **+ multimodal** (vision/audio) | **this fork** 👈 | [`cortexist/llama.cpp`](https://github.com/cortexist/llama.cpp) |
| Gemma 4 MTP, **no multimodal** | **atomic** | [`AtomicBot-ai/atomic-llama-cpp-turboquant`](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant) |
| TurboQuant, **no Gemma 4 MTP** | **bofan** | [`BoFan-tunning/llama.cpp-MTP-TurboQuant`](https://github.com/BoFan-tunning/llama.cpp-MTP-TurboQuant) |
| Just the base **TurboQuant** quantization | **turbo** (upstream of all of these) | [`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant) |

> In short: **need multimodal → here. Don't need multimodal → atomic. Don't need Gemma 4 MTP → bofan.**

### What this fork adds over `atomic`

- **Multimodal + MTP together**: the Gemma 4 `gemma4_assistant` draft head runs alongside an `--mmproj` projector, so speculative decoding works on image/audio prompts, not just text.
- Tuned and validated for **Jetson Orin NX** (Ampere `sm_87`) edge deployment.

---

## Benchmark — MTP vs. baseline

**Jetson Orin NX**, Gemma 4 E4B (`Q4_K_M`) target + `gemma4_assistant` MTP draft (`Q4_K_M`) + `mmproj` (f16), `-c 4096`, f16 KV, `-fa off`, draft block size `B = 2`, greedy / `temperature 0` (deterministic, so MTP is lossless). Throughput and draft-acceptance are read from the server's own `timings` (`predicted_per_second`, `draft_n_accepted / draft_n`).

| Test | Baseline (tk/s) | **MTP (tk/s)** | Speedup | Draft accept |
|------|:---:|:---:|:---:|:---:|
| Short (~40 tok) | 13.73 | **19.11** | 1.39× | 65.2% |
| Medium (256 tok) | 13.44 | **18.75** | 1.39× | 63.9% |
| Long (512 tok) | 13.08 | **18.77** | 1.44× | 64.5% |
| Image / vision (200 tok) | 12.76 | **18.44** | 1.45× | 61.0% |

MTP holds a steady **~1.4×** on long-form text *and* on vision generation. On the long answer that's a **39.1 s → 27.3 s** wall-clock win; on the image, **15.7 s → 10.8 s** — multimodal speculative decoding at essentially the same acceptance as text.

> **Acceptance is workload-dependent.** The draft-accept rate is how often the assistant head guesses the target's *next greedy token*, so it tracks how predictable the output is: explanatory prose accepts high (~61–65% above), while short / cold-start, code, or numeric spans accept lower (≈37–49% on terser prompts). Treat it as a range, not a single figure.

### Tuning — draft block size

`--spec-draft-block-size B` makes the head draft `B − 1` tokens per round. On the Edge centroid head, **`B = 2` is the sweet spot** — it maximizes *both* acceptance and throughput. Larger blocks append low-probability tail tokens that drag the aggregate accept rate down *and* spend extra draft compute, so they lose on this head (same text prompts as above):

| `--spec-draft-block-size` | Draft tokens / round | Accept | tk/s |
|:---:|:---:|:---:|:---:|
| **2** (default) | 1 | **64.3%** | **18.8** |
| 3 | 2 | 52.7% | 18.1 |
| 4 | 3 | 41.4% | 16.0 |

### Desktop / big-GPU? Temper expectations — these are *bandwidth-bound* wins

MTP and `turbo*` KV pay off when the target's decode is **memory-bandwidth-bound** — the regime of edge devices like the Orin. A fast desktop GPU runs a model this small **compute-bound**: it's already near its compute ceiling, so there's little headroom for speculation to recover, and the turbo dequant just adds compute the GPU wasn't waiting on.

Same harness on an **RTX A5000** (24 GB, Ampere `sm_86`), E4B `Q4_K_M`, `B = 2`, greedy, median tk/s:

| Config | tk/s | vs f16 base | Draft accept |
|---|:---:|:---:|:---:|
| f16, baseline | 108.4 | — | — |
| **f16 + MTP** | **115.9** | 1.07× | 65.6% |
| turbo3, baseline | 94.6 | 0.87× | — |
| turbo3 + MTP | 91.1 | 0.84× | 64.9% |

On a big GPU, MTP buys only ~**1.07×** (vs **1.4×** on the Orin), and `turbo3` is a net *loss* — its 3-bit unpack costs compute the GPU isn't bandwidth-starved for. Note draft **acceptance is the same ~65% as on the Orin**: acceptance is a property of the model + prompt, *not* the hardware — only the throughput *payoff* is hardware-dependent. **If you have a high-end discrete GPU and don't need the KV-memory savings, plain f16 (optionally + MTP) is the simplest choice; the speculative/turbo machinery is built for the bandwidth-bound edge.**

> **Scope of these numbers.** All benchmarks above are **Gemma 4 E4B**, this fork's primary target. The larger `gemma4_assistant` heads (**26B-A4B / 31B**) load and run, but their MTP throughput/acceptance haven't been measured yet — those numbers are **pending**.

---

## Quick start

Build (CUDA example — see upstream [build guide](docs/build.md) for other backends):

```bash
cmake -S . -B build -DGGML_CUDA=ON
cmake --build build --target llama-server -j
```

Run Gemma 4 E4B with the MTP draft head **and** multimodal, OpenAI-compatible server on `:8080`:

```bash
./build/bin/llama-server \
  -m   gemma-4-E4B-it-Q4_K_M.gguf \
  -md  gemma-4-E4B-it-assistant-mtp-Q4_K_M.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-block-size 2 \
  --mmproj mmproj-gemma-4-e4b-f16.gguf \
  -c 4096 -ngl 99 -ngld 99 \
  -ctk f16 -ctv f16 -ctkd f16 -ctvd f16 \
  -fa off \
  --host 0.0.0.0 --port 8080
```

> **CLI note — different from `atomic`, aligned with mainline.** This fork follows **mainline
> llama.cpp** speculative conventions: load the MTP assistant as a **draft model** with `-md`
> (`--model-draft`), enable it with `--spec-type draft-mtp`, and size the draft with `--spec-draft-n-max`
> / `--spec-draft-n-min`. The `atomic` fork instead uses `--mtp-head` and `--draft-max`/`--draft-min`.
> If you're porting a command from atomic, translate `--mtp-head → -md` and `--draft-max → --spec-draft-n-max`.

Pre-built `gemma4_assistant` MTP heads (E2B / E4B / 26B-A4B / 31B) are published in the
[**AtomicChat / Gemma 4 Assistant GGUF collection**](https://huggingface.co/collections/AtomicChat/gemma-4-assistant-gguf).
Full details on how MTP works here — graph, KV sharing, the speculative driver, presets — are in **[MTP.md](MTP.md)**.

---

## Flash attention — work in progress

The Gemma 4 MTP cross-attention has an unusual shape (**head_dim 512, gqa_ratio 2**) for which no stock CUDA flash-attention kernel is compiled, so historically the MTP path required **`-fa off`**.

- ✅ **Landed:** `-fa on` now works with **f16 KV** — the tiny single-token MTP op is routed through the non-FA path while the rest of the model keeps full flash attention.
- 🚧 **In progress:** **turbo3 KV + flash attention** at D=512 (the real prize — memory savings → much longer context). This needs a turbo-WHT-aware 512-dim kernel.

Until that lands, run MTP with `-fa off` (as above) for the proven path. Full diagnosis and the implementation plan are in **[MTP-flash-attention.md](MTP-flash-attention.md)**.

---

## TurboQuant — KV cache & weight compression

WHT-rotated low-bit quantization with backend-native kernels (Metal `TurboFlash`, CUDA, Vulkan, HIP).

| KV type (`-ctk`/`-ctv`) | Bits | Compression vs F16 | Notes |
|---|---:|---:|---|
| `turbo2` | 2 | ~6.4× | maximum compression, large-context budgets |
| `turbo3` | 3 | ~4.3× | recommended default |
| `turbo4` | 4 | ~3.8× | highest accuracy of the family |

Weights can also be quantized to `TQ3_1S` / `TQ4_1S` via `llama-quantize`. See the upstream forks above for the full TurboQuant documentation.

> **Note:** MTP runs on **`-ctk f16` KV** by default. `turbo*` KV + MTP **also works** (the MTP cross-attn dequantizes the turbo V on its non-FA path), giving ~4× smaller KV — but since f16 KV already fits 128 K context on a 16 GB Orin, reach for `turbo3` with MTP only when you need very long context or are memory-tight. Plain (non-MTP) inference uses `turbo*` KV with `-fa on` as usual. Details in **[MTP-flash-attention.md](MTP-flash-attention.md)** (Option 3).

---

## Credits

TurboQuant in this lineage is built on the excellent work of **[@TheTom](https://github.com/TheTom)** in
[`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant) — the original WHT-rotated
quantization design, reference kernels, and backend ports. ❤️ Gemma 4 MTP and the assistant-head pipeline
come from the **atomic** / **bofan** forks above. This fork adds the multimodal + MTP integration on top.

Everything here remains MIT-licensed, like upstream [`llama.cpp`](https://github.com/ggml-org/llama.cpp).
