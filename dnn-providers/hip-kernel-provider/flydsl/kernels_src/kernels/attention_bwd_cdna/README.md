# attention_bwd_cdna

The gfx950 SDPA backward kernels: prep, dK/dV, dQ, merged, dBias reduce and dSink reduce, with the modules they share. Every file is written for this provider and has no upstream (`# Not vendored` header).

## Origin

The FlyDSL SDPA backward sweep at commit `e196d0d` (its sources are unchanged since `8142d08`). The kernel logic is byte-for-byte the sweep's; only the headers and the import lines differ (flat module imports became `kernels.attention_bwd_cdna` package imports, and the `sys.path` insertions of the prep, dBias and dSink modules were dropped).

## Modules

- Kernels: `fmha_bwd_prep_gfx950`, `fmha_bwd_dkdv_gfx950`, `fmha_bwd_dkdv_m16_gfx950`, `fmha_bwd_dq_gfx950`, `fmha_bwd_dq_m16_gfx950`, `fmha_bwd_merged_gfx950`, `fmha_bwd_dbias_reduce_gfx950`, `fmha_bwd_dsink_reduce_gfx950`.
- Tuning and traits: `fmha_tuning_gfx950`, `fmha_tuning_bwd_dkdv_gfx950`, `fmha_tuning_bwd_dq_gfx950`, `fmha_traits_gfx950`.
- Shared: `fmha_common_gfx1201`, `fmha_abi_gfx1201`, `fmha_dualwave_gfx950`, `fmha_wide_gfx950`, `fmha_mfma16_gfx950`, `philox`, `flyc_polyfill`, `gfx950_standalone`.

## `flash_attn_utils`

`gfx950_standalone.dualwave` resolves to `kernels.attention_cdna.flash_attn_utils`, the forward kernels' copy. It differs from the one the sweep built against (bias constants and `_no_keys`, generic bias and softmax paths, dual-wave scaled-score branches). Spike P compiled the sampled specs against it: none of the backward objects changed, and every non-flaky one reproduced its pinned sha256.

## Build nondeterminism

LLVM AMDGPU codegen can turn identical sources into different objects, or a dump holding two different objects. Five specs are known to be bimodal: `1a114be37069`, `4691c39657a8`, `5f6f40c7b88a`, `065be7bccb37`, `d86ab8bde882`. A shipped object is pinned by hash; it is not rebuilt blindly.
