# ASM Kernel Binaries — Provenance

This directory holds vendored GPU code-object (`.co`) binaries from the AITER
project for both forward and backward FMHA kernels. The forward and backward
snapshots were taken from **different** AITER commits (see table below).

| Kernel group | AITER commit | Platforms | Per-arch SOURCE.md |
|---|---|---|---|
| `fmha_v3_fwd` (gfx942, gfx950) | `commit` (objects) and `abi_commit` (host code) of each `<arch>/fmha_v3_fwd` module in `aiter_pins.json` | gfx942, gfx950 | `gfx942/fmha_v3_fwd/SOURCE.md`, `gfx950/fmha_v3_fwd/SOURCE.md` |
| `fmha_v3_bwd` (gfx942, gfx950) | `cdcfa833bdf554ca75594c90dde4316ea9b50199` (v0.1.13) | gfx942, gfx950 | `gfx942/fmha_v3_bwd/SOURCE.md`, `gfx950/fmha_v3_bwd/SOURCE.md` |

Pinned modules are recorded in `aiter_pins.json`, keyed `<arch>/<module>`. For
each forward or backward module it records the AITER commit of its objects, the
AITER host code the engine mirrors, and the upstream rows or CSVs left out with
the reason. The `SOURCE.md` of every pinned module is generated from it by the
`hipdnn-aiter-asm-sdpa-fwd-sync` skill. The forward modules are pinned; the
backward modules have no pin, and their `SOURCE.md` files are written by hand.
The forward and backward kernels come from different AITER commits, so check
each group against its own row above.

## Directory layout notes

- **Forward kernels** (`fmha_v3_fwd/`): gfx942 keeps AITER's separate `MI300/` and `MI308/`
  subdirectories, because MI308X needs its own objects. The engine picks `MI308/` at runtime
  when the device's PCI chip ID is an MI308X ID (`isMi308Device` in `SdpaFwdPlanBuilder.cpp`)
  and `MI300/` for every other gfx942 device, including MI300X and MI300A. gfx950 stores its
  objects flat.
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
