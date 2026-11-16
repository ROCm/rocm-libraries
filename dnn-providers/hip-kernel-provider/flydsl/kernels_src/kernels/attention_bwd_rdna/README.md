# attention_bwd_rdna

The gfx11-generic (`gfx11-generic`, wave32 WMMA) SDPA backward kernels: prep, dK/dV, dQ, dBias reduce and tail-zero, in a 16-bit and an fp32 (VALU) family, with the modules they share. Every file is written for this provider; the AOTriton-derived `fmha_common_gfx1201.py` is `../attention_bwd_cdna/`'s, which this package imports. "The sweep" is the AMD-internal FlyDSL backward kernel sweep these sources come from.

## Origin

The gfx11-generic sources come from a snapshot of AMD's FlyDSL SDPA backward kernel development repository. They are that snapshot's with three differences: the headers, the import lines (flat module imports became `kernels.attention_bwd_rdna` package imports, and the `sys.path` insertions and the `os`/`sys` imports only they used were dropped), and black formatting. Attribution that the snapshot carried (AITER, AOTriton notices) is kept under the header. Docstring and comment references to the sweep's own files (`tools/`, `h/`) name paths in that repository.

The kernels import `../attention_bwd_cdna/fmha_common_gfx1201.py` and `fmha_bwd_tailzero_addressing.py`; see that package's README for their origin.

## Modules

- Kernels: `fmha_bwd_prep_gfx11`, `fmha_bwd_dkdv_gfx11`, `fmha_bwd_dq_gfx11`, `fmha_bwd_dbias_reduce_gfx11`, `fmha_bwd_tailzero_gfx11`.
- fp32 kernels: `fmha_bwd_prep_f32_gfx11`, `fmha_bwd_dkdv_f32_gfx11`, `fmha_bwd_dq_f32_gfx11`, `fmha_bwd_dbias_reduce_f32_gfx11`.
- Tuning: `fmha_tuning_bwd_dkdv_gfx11`, `fmha_tuning_bwd_dq_gfx11`, `fmha_tuning_bwd_dkdv_f32_gfx11`, `fmha_tuning_bwd_dq_f32_gfx11`.
- Shared: `fmha_common_gfx11` (guarded-staging helpers over `_if_then`, which the pinned FlyDSL's `kernels.common` lacks; `flyc_polyfill` defines it), `rdna_layout_gfx11` (WMMA layout algebra), `flyc_polyfill`. Imported from `attention_bwd_cdna`: `fmha_common_gfx1201`, `fmha_bwd_tailzero_addressing`.

## Shared-module choices

| Module | Used from | Why |
|---|---|---|
| `kernels.common.{buffer_ops, kernels_common, mem_ops}` | in-tree `kernels/common` | byte-identical to the sweep's |
| `philox` | in-tree `kernels.attention_rdna4.philox` | code-identical to the sweep's |
| `fmha_common_gfx1201`, `fmha_bwd_tailzero_addressing` | `kernels.attention_bwd_cdna` | one copy for both arches; it carries the helper variants each arch's bytes need |
| `flyc_polyfill` | this package | differs from `attention_rdna4` and `attention_bwd_cdna` (it carries no dual-wave trait record); supplies `_if_then` to `fmha_common_gfx11` |

The gfx11 builders do not import `fmha_abi_gfx1201`, any `*_gfx950` module or `kernels.attention_cdna.flash_attn_utils`. The generic-target shim and `arch_families.json` are generator inputs, not kernel imports; they live in `generators/`.

## Build

- Set `SWEEP_ARCH=gfx11-generic` before FlyDSL is imported; the generator installs the generic-target shim (`generators/_generic_targets.py`), which FlyDSL 0.3.4 needs to emit a generic object.
- The sampled build is the 504 specs in `tests/data/sdpa_bwd_gfx11_reference.csv`; each object is pinned by hash. A build from this package reproduces them byte for byte under the same toolchain.
- Caller preconditions for the kernels (pad8 storage for `hd % 8 != 0`, empty shapes handled on the host, launch-grid limits, mandatory prep) are the caller's. The dK/dV module docstring states its `Scope` and `Padded head`; the other kernels' docstrings do not state them all.
