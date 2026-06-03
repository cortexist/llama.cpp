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

## Option 3 — turbo3 KV + FA at D=512 (FUTURE WORK)

The real payoff: turbo KV compression *with* flash attention lets context grow well past 4 K on the
Orin. This is a genuine CUDA-kernel project, not a graph tweak. Open problems:

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
