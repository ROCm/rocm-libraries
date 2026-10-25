# attention_bwd_cdna

The gfx950 SDPA backward kernels: prep, dK/dV, dQ, merged, dBias reduce and dSink reduce, in a 16-bit and an fp32 (native f32 MFMA) family, with the modules they share. Every file is written for this provider and has no upstream (`# Not vendored` header). "The sweep" is the AMD-internal FlyDSL backward kernel sweep these sources and pins come from.

## Origin

The FlyDSL SDPA backward sweep (an AMD-internal repository), at the commit whose pins are recorded in `tests/data/sdpa_bwd_reference.csv`. The kernel sources are the sweep snapshot's with three differences: the headers, the import lines (flat module imports became `kernels.attention_bwd_cdna` package imports, and the `sys.path` insertions were dropped), and black formatting. Docstring and comment references to the sweep's own files (`tools/`, `parity/`, `RP/`) name paths in that repository.

The deliberate additions (each normalized source diff against the sweep shows these and nothing else):

- The merged launcher and both dQ launchers (16-bit and fp32) take `dbias_reduce=False`: a second launch of the dBias reduce kernel after the main one, so a dB-storing bias object holds two kernels (the main kernel and `fmha_bwd_dbias_reduce[_f32]_gfx950_kernel_1`). Each launcher gains the reduce launch's arguments (`DBias`, `reduce_batch`, `reduce_head`, `stride_ws_*`, `stride_out_*`, after `rt_flags`); they serve compilation only. The dQ launchers expose `.jit`; the dQ `_launch` / `_compile` of the 16-bit module pass placeholders for those arguments, and refuse a `dbias_reduce` build.
- The 16-bit launchers (merged, dQ) reset the module's fast-math compile hints around the reduce launch; the fp32 dQ launcher does not (the fp32 reduce has no hints, and none reach it).
- The gpu module is compiled under one set of LLVM options, so the reduce kernel carries the main module's: its bytes differ from the standalone reduce pin's (the 16-bit resource row is equal; the fp32 dQ object's `unroll-max-count=1` also changes the fp32 reduce's resource row, so an fp32 combined object is gated against a standalone fp32 reduce built under the dQ options).
- `_launch.kernel` on both reduce modules (the kernel the dQ and merged launchers launch).

## Modules

- Kernels: `fmha_bwd_prep_gfx950`, `fmha_bwd_dkdv_gfx950`, `fmha_bwd_dkdv_m16_gfx950`, `fmha_bwd_dq_gfx950`, `fmha_bwd_dq_m16_gfx950`, `fmha_bwd_merged_gfx950`, `fmha_bwd_dbias_reduce_gfx950`, `fmha_bwd_dsink_reduce_gfx950`.
- fp32 kernels (prep, dK/dV, dQ, dBias reduce only): `fmha_bwd_prep_f32_gfx950`, `fmha_bwd_dkdv_f32_gfx950`, `fmha_bwd_dq_f32_gfx950`, `fmha_bwd_dbias_reduce_f32_gfx950`, on the native f32 MFMA core `fmha_f32_common_gfx950`, `fmha_f32_dsplit_gfx950`, `fmha_f32_perwave_gfx950`.
- Tuning and traits: `fmha_tuning_gfx950`, `fmha_tuning_bwd_dkdv_gfx950`, `fmha_tuning_bwd_dq_gfx950`, `fmha_tuning_bwd_dkdv_f32_gfx950`, `fmha_tuning_bwd_dq_f32_gfx950`, `fmha_traits_gfx950`.
- Shared: `fmha_common_gfx1201`, `fmha_abi_gfx1201`, `fmha_dualwave_gfx950`, `fmha_wide_gfx950`, `fmha_mfma16_gfx950`, `philox`, `flyc_polyfill`, `gfx950_standalone`.

## `flash_attn_utils`

`gfx950_standalone.dualwave` resolves to `kernels.attention_cdna.flash_attn_utils`, the forward kernels' copy. It differs from the one the sweep built against (bias constants and `_no_keys`, generic bias and softmax paths, dual-wave scaled-score branches). The sampled specs (28, one per kind and dtype plus the known-flaky and special cases) were compiled against it: every non-flaky one reproduced its pinned sha256; the others are known-flaky builds.

## Build nondeterminism

LLVM AMDGPU codegen can turn identical sources into different objects, or a dump holding two different objects. `tests/data/sdpa_bwd_known_flaky.csv` lists the 22 `dq` causal specs that were nondeterministic in the sweep (a `source` column says how: `bimodal`, `rmir-retry`, `attempts`). Five of them are bimodal: `1a114be37069`, `4691c39657a8`, `5f6f40c7b88a`, `065be7bccb37`, `d86ab8bde882`. A shipped object is pinned by hash; it is not rebuilt blindly.
