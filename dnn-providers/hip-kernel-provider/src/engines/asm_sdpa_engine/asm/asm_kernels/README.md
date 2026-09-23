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
