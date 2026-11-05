# ASM Kernel Binaries — Provenance

This directory holds vendored GPU code-object (`.co`) binaries from the AITER
project for both forward and backward FMHA kernels. The forward and backward
snapshots were taken from **different** AITER commits (see table below).

| Kernel group | AITER commit | Platforms | Per-arch SOURCE.md |
|---|---|---|---|
| `fmha_v3_fwd` (gfx942, gfx950) | `17d4a33b6f9535e820353ebc6217769efc3766d6` | gfx942, gfx950 | — |
| `fmha_v3_bwd` (gfx942, gfx950) | `cdcfa833bdf554ca75594c90dde4316ea9b50199` (v0.1.13) | gfx942, gfx950 | `gfx942/fmha_v3_bwd/SOURCE.md`, `gfx950/fmha_v3_bwd/SOURCE.md` |

The forward POC snapshot (`17d4a33b`) pre-dates the backward snapshot
(`cdcfa833`, AITER v0.1.13). There is no single unified AITER commit that
contains both the forward and backward kernels used here. When performing an
AITER refresh, both groups must be updated together (or the commit mismatch
documented explicitly) so reviewers can verify provenance from a single source
of truth.

## Directory layout notes

- **Forward kernels** (`fmha_v3_fwd/`): AITER has separate `MI300/` and `MI308/` subdirectories
  because MI308 (MI300A APU) has different fp8 support. We only include MI300 for the POC.
- **Backward kernels** (`fmha_v3_bwd/`): AITER stores these flat (no MI300/MI308 split) because
  backward kernels only support bf16/fp16 (no fp8), so the same binary works on both MI300X and
  MI300A. SHA256 manifests for each arch are recorded in the per-arch `SOURCE.md` files.

## Data-type contract

`SdpaFwdPlanBuilder` and `SdpaBwdPlanBuilder` decline (in `isApplicable`, with a logged
reason) any graph outside this contract:

| Tensor / attribute | Forward (`SdpaFwdPlanBuilder`, gfx942/gfx950) | Backward (`SdpaBwdPlanBuilder`, gfx942) |
|---|---|---|
| Q, K, V | BF16, or FP8_E4M3_FNUZ (gfx942 kernels only, scalar q/k/v descales required) | BF16 or FP16 |
| O | BF16 | same dtype as Q (not yet checked by `isApplicable`; tracked in ALMIOPEN-3050) |
| dO, dQ, dK, dV | — | same dtype as Q |
| Stats (LSE) | FP32 (optional output) | FP32 (required input) |
| Node `compute_data_type` | FLOAT | FLOAT |
| `mma_core_mode` | UNSET, or HALF/BFLOAT16 for BF16 inputs | — |
| Descale Q, K, V (FP8 only) | FLOAT scalars, non-virtual, non-pass-by-value device tensors | — |

- **Accumulation.** QKᵀ, the online softmax and PV accumulate in fp32 inside the kernels;
  no intermediate tensor is exposed. Backward dQ accumulates through an fp32 workspace by
  default (`a32`). The `sdpa.bwd.accumulator_type=a16` knob opts into the 16-bit path.
- **Intermediate data type.** Neither builder reads the graph's `intermediate_data_type`.
  The frontend uses it only to type unset virtual tensors, and it is `(cache_ignore)` in
  `graph.fbs`. These single-node kernels expose no intermediate tensor, so both builders
  decline a virtual operand tensor instead (forward: Q, K, V, O, Stats and the FP8 descales;
  backward: every operand). A virtual tensor has no device buffer, and execute needs a pointer
  for every operand.
- **Enforcement.** Any other combination is declined in `isApplicable` with a logged reason,
  except the backward O dtype noted in the table.

## Verified Kernel Matrix

The backward dispatch (`SdpaBwdPlanBuilder`) can resolve kernels for any
`(dtype, head-dim)` combination present in the CSV registry, but only some of
those combinations have been calibrated against a CPU backward reference. The
table below records which are **verified** (numerically validated) and which are
**unverified** (dispatch succeeds, but results have not been checked against a
reference).

| dtype | hd64 | hd128 | hd192 |
|---|---|---|---|
| bf16 | unverified | **verified** | unverified |
| fp16 | unverified | **verified** | unverified |

This matrix is the source of truth for the `verified` flag computed in
`SdpaBwdPlanBuilder.cpp` (`computeDispatchTuples`). At runtime, dispatch emits a
one-time log per `(dtype, hdim, mask)`:

- **verified** kernel → `INFO` line confirming the configuration is validated.
- **unverified** kernel → `WARN` line stating the results are not validated
  against a CPU reference.

When a configuration is calibrated, update both this table and the `verified`
predicate in `computeDispatchTuples` so the documentation and code stay in sync.
