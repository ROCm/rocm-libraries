# attention_bwd_rdna

The gfx11-generic (`gfx11-generic`, wave32 WMMA) SDPA backward kernels: prep, dK/dV, dQ and dBias reduce, in a 16-bit and an fp32 (VALU) family, with the modules they share. Every file is written for this provider except `fmha_common_gfx1201.py`, which is AOTriton's; each file's header says which. "The sweep" is the AMD-internal FlyDSL backward kernel sweep these sources come from.

## Origin

The gfx11-generic sources come from a snapshot of AMD's FlyDSL SDPA backward kernel development repository. They are that snapshot's with three differences: the headers, the import lines (flat module imports became `kernels.attention_bwd_rdna` package imports, and the `sys.path` insertions and the `os`/`sys` imports only they used were dropped), and black formatting. Attribution that the snapshot carried (AITER, AOTriton notices) is kept under the header. Docstring and comment references to the sweep's own files (`tools/`, `h/`) name paths in that repository.

`fmha_common_gfx1201.py` is AOTriton's `modules/flash/flyc/` file (MIT; Apache-2.0 header, FlyDSL Project Contributors). It is code-identical to `../attention_bwd_cdna/fmha_common_gfx1201.py` except for the `flyc_polyfill` it imports, and differs from `../attention_rdna4/` only in the package imports.

## Modules

- Kernels: `fmha_bwd_prep_gfx11`, `fmha_bwd_dkdv_gfx11`, `fmha_bwd_dq_gfx11`, `fmha_bwd_dbias_reduce_gfx11`.
- fp32 kernels: `fmha_bwd_prep_f32_gfx11`, `fmha_bwd_dkdv_f32_gfx11`, `fmha_bwd_dq_f32_gfx11`, `fmha_bwd_dbias_reduce_f32_gfx11`.
- Tuning: `fmha_tuning_bwd_dkdv_gfx11`, `fmha_tuning_bwd_dq_gfx11`, `fmha_tuning_bwd_dkdv_f32_gfx11`, `fmha_tuning_bwd_dq_f32_gfx11`.
- Shared: `fmha_common_gfx1201`, `fmha_common_gfx11` (guarded-staging helpers over `_if_then`, which the pinned FlyDSL's `kernels.common` lacks; `flyc_polyfill` defines it for `fmha_common_gfx1201`), `rdna_layout_gfx11` (WMMA layout algebra), `flyc_polyfill`.

## Shared-module choices

| Module | Used from | Why |
|---|---|---|
| `kernels.common.{buffer_ops, kernels_common, mem_ops}` | in-tree `kernels/common` | byte-identical to the sweep's |
| `philox` | in-tree `kernels.attention_rdna4.philox` | code-identical to the sweep's |
| `fmha_common_gfx1201` | this package | `attention_rdna4` differs in the package imports only; `attention_bwd_cdna` binds its own polyfill |
| `flyc_polyfill` | this package | differs from `attention_rdna4` (a trait field) and from `attention_bwd_cdna` |

The gfx11 builders do not import `fmha_abi_gfx1201`, any `*_gfx950` module or `kernels.attention_cdna.flash_attn_utils`. The generic-target shim and `arch_families.json` are generator inputs, not kernel imports; they live in `generators/`.

## Build

- Set `SWEEP_ARCH=gfx11-generic` before FlyDSL is imported; the generator installs the generic-target shim (`generators/_generic_targets.py`), which FlyDSL 0.3.4 needs to emit a generic object.
- The sampled build is the 432 specs the sweep ships; each object is pinned by hash. A build from this package reproduces them byte for byte under the same toolchain.
- Caller preconditions for the kernels (pad8 storage for `hd % 8 != 0`, empty shapes handled on the host, launch-grid limits, mandatory prep) are the caller's. The dK/dV module docstring states its `Scope` and `Padded head`; the other kernels' docstrings do not state them all.
