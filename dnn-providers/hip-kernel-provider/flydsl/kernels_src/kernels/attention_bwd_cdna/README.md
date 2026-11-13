# attention_bwd_cdna

The gfx950 SDPA backward kernels: prep, dK/dV, dQ, merged, dBias reduce, dSink reduce and tail-zero, in a 16-bit and an fp32 (native f32 MFMA) family, with the modules they share. The files are written for this provider except `fmha_common_gfx1201.py` and `fmha_abi_gfx1201.py`, which are AOTriton's (see *Origin*); each file's header says which. "The sweep" is the AMD-internal FlyDSL backward kernel sweep these sources and pins come from.

## Origin

The FlyDSL SDPA backward sweep (an AMD-internal repository), whose pinned objects `tests/data/sdpa_bwd_reference.csv` lists (hash, spec, object SHA-256 and resource row; the sweep commit is not recorded in the tree). The kernel sources are the sweep snapshot's with these differences: the headers, the import lines (flat module imports became `kernels.attention_bwd_cdna` package imports, and the `sys.path` insertions were dropped), black formatting, comment and docstring text (below), and the additions listed under *Origin*. Docstring and comment references to the sweep's own files (`tools/`, `parity/`, `RP/`) name paths in that repository.

`fmha_common_gfx1201.py` and `fmha_abi_gfx1201.py` are AOTriton's `modules/flash/flyc/` files (MIT; Apache-2.0 header, FlyDSL Project Contributors), taken at AOTriton commit `b5e8cfb909e554c62081f56055b11d6ba3fd6bfa` (recorded in each header and in `REGEN.md`). They differ from the `attention_rdna4/` copies in the package imports; `fmha_common_gfx1201.py` also differs in helper bodies (`stats_ws_base`, `ptr_is_set`, the two ragged-origin helpers `tensor_origin` (gfx950) and `tensor_origin_gated` (gfx11), and a different `make_addr_pair`, where the rdna4 copy has `lse_row_addressing`) and imports `_load_u64_or_zero` from `attention_rdna4/`; `fmha_abi_gfx1201.py` differs in `prep_tensors`, which has no `v_heads_own` option (V's heads equal K's), in `row_tensor_arg`, in `ragged_args`, which only this copy has, and in `dropout_args`, whose threshold and scale are this module's host `dropout_threshold` and `dropout_scale` (float32, as the engines evaluate them); it has none of the rdna4 copy's varlen LSE layout: no `VARLEN_LSE_LAYOUT_*`, no `lse_layout` or `lse_tokens` parameter and no `lse_args`. `fmha_common_gfx1201.py` is the one copy of this file: `attention_bwd_rdna` imports it. The snapshot's `fmha_bwd_dq_gfx950.py` and `fmha_dualwave_gfx950.py` also differ in comment text: references to the sweep's issue numbers and phases are restated as the observation; `gfx950_standalone.py`'s module docstring and import fallback are this provider's. The prep, dK/dV, dQ and merged kernels port AOTriton's `bwd_preprocess`, `bwd_kernel_dk_dv`, `bwd_kernel_dq` and `bwd_kernel_fuse` (`THIRD_PARTY_LICENSES.md`, AOTriton).

The deliberate additions (each normalized source diff against the sweep shows these and nothing else):

- The merged launcher and both dQ launchers (16-bit and fp32) take `dbias_reduce=False`: a second launch of the dBias reduce kernel after the main one, so a dB-storing bias object holds two kernels (the main kernel and `fmha_bwd_dbias_reduce[_f32]_gfx950_kernel_1`). Each launcher gains the reduce launch's arguments (`DBias`, `reduce_batch`, `reduce_head`, `stride_ws_*`, `stride_out_*`, after `rt_flags`); they serve compilation only. The dQ launchers expose `.jit`; the dQ `_launch` / `_compile` of the 16-bit module pass placeholders for those arguments, and refuse a `dbias_reduce` build.
- The 16-bit launchers (merged, dQ) reset the module's fast-math compile hints around the reduce launch; the fp32 dQ launcher does not (the fp32 reduce has no hints, and none reach it).
- The gpu module is compiled under one set of LLVM options, so the reduce kernel carries the main module's: its bytes differ from the standalone reduce pin's (the 16-bit resource row is equal; the fp32 dQ object's `unroll-max-count=1` also changes the fp32 reduce's resource row, so an fp32 combined object is gated against a standalone fp32 reduce built under the dQ options).
- `flyc_polyfill._if_then` (a guard helper) and `fmha_common_gfx1201.py`'s five `with common_utils._if_then(...)` sites, which the AOTriton file spells with `kernels_common._if_then`; the `kernels.common` this tree vendors does not define it.
- `_launch.kernel` on both reduce modules (the kernel the dQ and merged launchers launch).

## Modules

- Kernels: `fmha_bwd_prep_gfx950`, `fmha_bwd_dkdv_gfx950`, `fmha_bwd_dkdv_m16_gfx950`, `fmha_bwd_dq_gfx950`, `fmha_bwd_dq_m16_gfx950`, `fmha_bwd_merged_gfx950`, `fmha_bwd_dbias_reduce_gfx950`, `fmha_bwd_dsink_reduce_gfx950`, `fmha_bwd_tailzero_gfx950`.
- fp32 kernels (prep, dK/dV, dQ, dBias reduce only): `fmha_bwd_prep_f32_gfx950`, `fmha_bwd_dkdv_f32_gfx950`, `fmha_bwd_dq_f32_gfx950`, `fmha_bwd_dbias_reduce_f32_gfx950`, on the native f32 MFMA core `fmha_f32_common_gfx950`, `fmha_f32_dsplit_gfx950`, `fmha_f32_perwave_gfx950`.
- Tuning and traits: `fmha_tuning_gfx950`, `fmha_tuning_bwd_dkdv_gfx950`, `fmha_tuning_bwd_dq_gfx950`, `fmha_tuning_bwd_dkdv_f32_gfx950`, `fmha_tuning_bwd_dq_f32_gfx950`, `fmha_traits_gfx950`.
- Shared: `fmha_common_gfx1201`, `fmha_abi_gfx1201`, `fmha_bwd_tailzero_addressing` (the tail-zero kernels' row addressing), `fmha_dualwave_gfx950`, `fmha_mfma16_gfx950`, `fmha_bwd_bias_exp_gfx950` (the bias exponent's fast-math flags and LSE floor), `flyc_polyfill`, and from `attention_rdna4` the `philox` module (Philox, `keep_mask`), `gfx950_standalone`.

## `flash_attn_utils`

`gfx950_standalone.dualwave` resolves to `kernels.attention_cdna.flash_attn_utils`, the forward kernels' copy. It differs from the one the sweep built against (bias constants and `_no_keys`, generic bias and softmax paths, dual-wave scaled-score branches). The sampled specs (28, one per kind and dtype plus the known-flaky and special cases) were compiled against it: every non-flaky one reproduced its pinned sha256; the others are known-flaky builds.

## Build nondeterminism

LLVM AMDGPU codegen can turn identical sources into different objects, or a dump holding two different objects. `tests/data/sdpa_bwd_nondeterministic.csv` lists the rows sampled as nondeterministic, and the policy for them is the "Nondeterministic builds" section of `generators/_sdpa_bwd_cdna_build.py`. A shipped object is pinned by hash; it is not rebuilt blindly.
