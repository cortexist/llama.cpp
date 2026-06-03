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

**Jetson Orin NX**, Gemma 4 E4B (`Q4_K_M`) + `gemma4_assistant` MTP draft + `mmproj` (f16), `-c 4096`, f16 KV, greedy (deterministic, so MTP is lossless). Generation throughput is read from the server's own `timings`.

| Test | Baseline (tk/s) | **MTP (tk/s)** | Speedup | Draft accept |
|------|:---:|:---:|:---:|:---:|
| Cold start | 13.33 | **15.42** | 1.16× | 37.5% |
| Short | 13.65 | **16.26** | 1.19× | 37.5% |
| Medium (256 tok) | 13.20 | **16.47** | 1.25× | 41.1% |
| Long (512 tok) | 12.83 | **17.65** | 1.38× | 48.8% |
| Photo / vision (200 tok) | 12.55 | **17.50** | 1.39× | 48.5% |

MTP speedup scales with output length as draft acceptance climbs (37% → 49%), reaching **~1.4×** on long-form *and* vision generation. On the long answer that's a **40.4 s → 29.6 s** wall-clock win; on the photo, **18.3 s → 14.1 s**.

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
  --spec-type mtp --spec-draft-n-max 2 \
  --mmproj mmproj-gemma-4-e4b-f16.gguf \
  -c 4096 -ngl 99 -ngld 99 \
  -ctk f16 -ctv f16 -ctkd f16 -ctvd f16 \
  -fa off \
  --host 0.0.0.0 --port 8080
```

> **CLI note — different from `atomic`, aligned with mainline.** This fork follows **mainline
> llama.cpp** speculative conventions: load the MTP assistant as a **draft model** with `-md`
> (`--model-draft`), enable it with `--spec-type mtp`, and size the draft with `--spec-draft-n-max`
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

> **Note:** `turbo*` KV with the Gemma 4 MTP draft currently requires `-fa off` (see the flash-attention section). Plain (non-MTP) inference can use `turbo*` KV with `-fa on` as usual.

---

## Credits

TurboQuant in this lineage is built on the excellent work of **[@TheTom](https://github.com/TheTom)** in
[`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant) — the original WHT-rotated
quantization design, reference kernels, and backend ports. ❤️ Gemma 4 MTP and the assistant-head pipeline
come from the **atomic** / **bofan** forks above. This fork adds the multimodal + MTP integration on top.

Everything here remains MIT-licensed, like upstream [`llama.cpp`](https://github.com/ggml-org/llama.cpp).
