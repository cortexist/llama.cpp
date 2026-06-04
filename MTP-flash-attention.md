# MTP + CUDA Flash-Attention — status & memo

Companion to [`MTP.md`](MTP.md). Covers why flash-attention crashed with the Gemma-4-E4B MTP
draft, the fix that landed (option 1), and the plan for the real prize (option 3: turbo3 + FA).

## TL;DR

- **Symptom:** `llama-server -fa on` aborted at the first MTP draft step:
  `ggml/src/ggml-cuda/fattn.cu:109: fatal error` (GGML_ABORT in the MMA-f16 dispatch).
  Backtrace: `decode_mtp_wait` → `common_speculative_state_mtp::draft`.
- **Root cause:** the Gemma-4-E4B MTP cross-attention is **head_dim 512, gqa_ratio 2**. No CUDA
  flash-attention kernel is compiled for that shape — both the **MMA** and **TILE** large-head
  paths only cover `gqa_ratio % 4 == 0` (ncols2 ≥ 4). gqa=2 at D=512 has no kernel and aborts.
  This is **not** Ampere-specific — it aborts on any arch (Ada included) with `-fa on`.
- **Fixed (option 1):** route just the MTP cross-attention op through the **non-FA** path
  (it's a single query token — negligible cost) while the rest of the model keeps full FA.
  `-fa on` with **f16 KV** now runs correctly. turbo3 + FA at D=512 is still unsupported → see
  "Option 3" below.

## The shapes (measured on Orin NX, from the load log)

Target `supergemma-4-E4B-it-Q4_K_M` (arch gemma4): full-attention layers
`n_head=8, n_head_kv=2, n_embd_head_k=v=512`; SWA layers `n_embd_head=256`. The MTP assistant
(`gemma4_assistant`, 4 blocks) has `head_count=4, head_count_kv=2, key/value_length=512`.

MTP cross-attention reads ONE shared KV per attention type from the target's **last** layer of
that type (`gemma4_mtp_kv_layer_last_in_range`). For the full-attention type that KV is **D=512**,
and Q (MTP, 4 heads) vs KV (target, 2 heads) ⇒ **gqa_ratio = 2**.

## Why no kernel exists (D=512, gqa=2)

- `fattn.cu` `ggml_cuda_get_best_fattn_kernel`: D>256 ⇒ `can_use_vector_kernel` is false (VEC only
  for D ∈ {64,128,256}) ⇒ falls to `BEST_FATTN_KERNEL_MMA_F16`.
- `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512,512>`: handles only `gqa_ratio > 4` (ncols2=8)
  and `gqa_ratio > 2` (ncols2=4); else `GGML_ABORT` (fattn.cu:109). gqa=2 falls into the abort.
  Instances confirm it: `DECL_FATTN_MMA_F16_CASE(512,512,*,4)` and `(*,8)` exist; **no `(*,2)`**.
- `fattn-tile.cuh` `launch_fattn_tile_switch_ncols2`: the ncols2∈{1,2} fallbacks are gated behind
  `if constexpr (DV <= 256)`, so D=512/gqa=2 hits `GGML_ABORT` (tile.cuh:1263) too.
- The main model self-attention survives FA because it is gqa=4 (8/2) ⇒ ncols2=4 path.

## Option 1 — what landed (this change)

`src/llama-graph.h` / `src/llama-graph.cpp`: add a defaulted `bool force_no_flash_attn = false`
to `build_attn_mha`; wire it into `use_flash_attn`. `build_attn_mtp` passes `true`.

- When `-fa off`: no-op (MTP was already non-FA).
- When `-fa on`: MTP cross-attn uses the standard `mul_mat`+softmax+`mul_mat` path (handles any
  D/gqa with f16 KV); everything else keeps flash attention.
- Output shape is identical (`build_attn_mha` returns the same 2D tensor in both paths), so
  `build_attn_mtp`'s downstream code is unchanged.

**Validated on Orin NX (incremental libllama rebuild, 67 s, no CUDA recompile):** `-fa on` + f16 KV
runs the full benchmark with no abort, identical token counts to `-fa off` (greedy). Generation
throughput is ~neutral-to-slightly-slower for batch-1 @ 4 K ctx (FA wins are long-ctx prefill +
memory, not exercised here):

| test | -fa off (tk/s) | -fa on (tk/s) |
|------|:---:|:---:|
| cold | 15.42 | 14.47 |
| short | 16.26 | 15.24 |
| medium | 16.47 | 16.61 |
| long | 17.65 | 16.80 |
| photo | 17.50 | 16.80 |

## Option 3 — turbo3 KV + MTP — LANDED (non-FA dequant); turbo-VEC FA at D=512 still open

**Status (2026-06-03, RTX A5000 sm_86): turbo3 KV now works with the Gemma 4 MTP draft.** The
crash is fixed and turbo3 + MTP produces correct output. The fix is a graph-side dequant plus a
small `cpy.cu` kernel — **not** the big turbo-WHT FA kernel the rest of this memo discusses (that
is still future work, but the single-token MTP op does not need it).

### Root cause (instrumented 2026-06-03)

A **layout mismatch**, not a generically missing feature:
- `turbo*` KV **forces `cparams.flash_attn = 1`** even with `-fa off` (see `llama-context.cpp`
  "turbo cache types require flash_attn", plus the hard `V cache quantization requires flash_attn`
  check). With FA on, the V cache is stored **non-transposed** (`v_trans = v->nb[1] > v->nb[2]` is
  false).
- The **main model** self-attn takes the FA path and consumes non-transposed turbo V fine (why
  plain non-MTP turbo3 works).
- The **MTP cross-attn** is pinned **non-FA** (`build_attn_mtp` → `build_attn_mha` with
  `force_no_flash_attn = true`, since D=512/gqa=2 has no FA kernel). The non-FA path needs V
  transposed, so it hit `if (!v_trans) { v = ggml_cont(ggml_transpose(v)); }` — a same-type
  `turbo3→turbo3` `ggml_cpy` → `cpy.cu:551` abort. (Every MTP layer logged `v_trans=0 fa=1
  force_no_fa=1` right before the abort.) You **cannot** "just add a turbo3→turbo3 copy kernel":
  transposing 128-element 3-bit WHT blocks shatters them — it is ill-defined.

### The fix that landed

1. `ggml/src/ggml-cuda/cpy.cu`: a `turbo3_0 → f32` dequant copy (`cpy_blck_turbo3_0_f32`,
   mirroring the q8_0 dequant-copy and reusing `dequantize_turbo3_0`, so values match the FA /
   mul_mat paths) + a dispatch entry; `ggml-cuda.cu` marks CPY `turbo3→f32` supported.
2. `src/llama-graph.cpp` `build_attn_mha`: when this attn will run non-FA and V is turbo,
   **dequantize V to f32 up front** (while still contiguous, before the permute), capture
   `v_was_turbo`, and key the output inverse-WHT on that flag (the WHT group still comes from K,
   which stays turbo). Only the forced-non-FA MTP path is affected; the main model's FA path is
   untouched.

**Validation (A5000):** no crash; coherent output; draft acceptance **64.9% ≈ f16's 65.6%** (a
wrong dequant would collapse acceptance); throughput ~88–97 tk/s vs ~95–117 f16 (the dequant +
3-bit read cost a little on this compute-bound GPU; on a bandwidth-bound Orin the KV-bandwidth
saving may offset — untested). KV at 4 K ≈ 32 MiB vs 124 MiB f16 (~4×). **Pending an Orin
(sm_87) rebuild to validate there.**

### When to use it

turbo3 + MTP is now an *option*, not the default. f16 KV already fits **128 K** context on a 16 GB
Orin (≈3.9 GB KV; weights 4.6 + mmproj 0.9 → 9.4 GB) and doesn't run out until ~200 K+, so the
~4× KV saving only matters at extreme contexts. Ship MTP on **f16 KV** unless you specifically
need very long context or are tight on memory; reach for turbo3 + MTP when you do.

### Still open — turbo-VEC FA at D=512 (the original "Option 3")

The turbo-WHT FA kernel below is still unimplemented. It is **not** needed for MTP (single-token,
forced non-FA, now handled by the dequant above); it would only matter for turbo3 + FA on the
*main model's* long-context prefill. The original open problems:

1. **turbo3 is coupled to the VEC kernel.** `build_attn_mtp` WHT-transforms Q (`ggml_turbo_wht`,
   llama-graph.cpp ~2521) and stores K in the turbo/WHT domain. Only the VEC turbo path
   (`vec_dot_fattn_vec_KQ_turbo3_0` + `dequantize_V_turbo3_0` in `fattn-common.cuh`) computes that
   dot product correctly. A plain `to_fp16` reconstruct + standard dot (what MMA/TILE do via
   `launch_fattn`'s K_f16/V_f16 conversion) would be **mathematically wrong** vs the WHT-domain Q.
2. **VEC turbo kernels are only instantiated for D ∈ {64,128,256}.** D=512 needs a new turbo VEC
   instance (`FATTN_VEC_CASE`/`FATTN_VEC_CASES_ALL_D` in `fattn.cu` cap at 256; the dispatch in
   `get_best_fattn_kernel` also gates turbo on `K->ne[0] % 64 == 0` but VEC max D is 256). Adding
   D=512 turbo VEC: check shared-memory/register budget for 512-wide heads.
3. **Alternative:** a turbo-WHT-aware TILE/MMA kernel for D=512 (handle the WHT/innerq scale inside
   the kernel rather than pre-converting to f16). Larger, but unlocks gqa=2 batched paths too.
4. **Main-model self-attn with turbo3 + FA (D=512, gqa=4) is untested** — currently always run with
   `-fa off`. Verify whether the main `build_attn` path WHT-transforms Q for turbo (the forward WHT
   was only seen in `build_attn_mtp`; the FA/non-FA paths in `build_attn_mha` only do the *inverse*
   WHT on the output). If the main path needs forward-Q-WHT for turbo+FA, that's additional work.
5. **Validation is slow:** turbo VEC/tile changes are `.cu` files ⇒ full CUDA recompile on the Orin
   (tens of minutes), expect several iterations for smem/correctness. Budget accordingly.

Files in play: `ggml/src/ggml-cuda/{fattn.cu, fattn-common.cuh, fattn-vec.cuh, fattn-tile.cuh,
fattn-mma-f16.cuh}`, `ggml/src/ggml-cuda/template-instances/`, and the turbo helpers
`ggml/src/ggml-cuda/{turbo-wht.cu, turbo-innerq.cu, turbo-quant.cuh}`; graph side
`src/llama-graph.cpp` (`build_attn_mha`, `build_attn_mtp`).
