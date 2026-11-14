# ASM Kernel Binaries — Provenance

This directory holds vendored GPU code-object (`.co`) binaries from the AITER
project for both forward and backward FMHA kernels. Each module's release is in
`aiter_pins.json` and its `SOURCE.md`.

| Kernel group | AITER commit | Platforms | Per-arch SOURCE.md |
|---|---|---|---|
| `fmha_v3_fwd` (gfx942, gfx950) | `commit` (objects) and `abi_commit` (host code) of each `<arch>/fmha_v3_fwd` module in `aiter_pins.json` | gfx942, gfx950 | `gfx942/fmha_v3_fwd/SOURCE.md`, `gfx950/fmha_v3_fwd/SOURCE.md` |
| `fmha_v3_bwd` (gfx942, gfx950) | `a0c4b69c146d94e393984bbe553b704e8c89a5e4` (v0.1.29) | gfx942, gfx950 | `gfx942/fmha_v3_bwd/SOURCE.md`, `gfx950/fmha_v3_bwd/SOURCE.md` |

Every module is pinned in `aiter_pins.json`, keyed `<arch>/<module>`. For each
forward and backward module the pin records the AITER release and commit of its
objects, the AITER host code the engine mirrors, and the upstream rows or CSVs
left out, with the reason. The `hipdnn-aiter-asm-sdpa-fwd-sync` skill writes the
pin and generates every module's `SOURCE.md` from it; edit neither by hand.
Change what a module leaves out with the skill's `plan --exclude`,
`--drop-exclude` or `--include`, then `apply`.

## Directory layout notes

- **Forward kernels** (`fmha_v3_fwd/`): gfx942 keeps AITER's separate `MI300/` and `MI308/`
  subdirectories, because MI308X needs its own objects. The engine picks `MI308/` at runtime
  when the device's PCI chip ID is an MI308X ID (`isMi308Device` in `SdpaFwdPlanBuilder.cpp`)
  and `MI300/` for every other gfx942 device, including MI300X and MI300A. gfx950 stores its
  objects flat.
- **Backward kernels** (`fmha_v3_bwd/`): AITER stores these flat (no MI300/MI308 split) because
  backward kernels only support bf16/fp16 (no fp8), so the same binary works on every gfx942
  device, MI308X included. SHA256 manifests for each arch are recorded in the per-arch
  `SOURCE.md` files.

## Forward precision

The gfx950 bf16 hd256 forward kernels (group mode only) multiply Q by
scale × log2(e) and round the product to bf16 before QK^T, where the hd128 and
hd192x128 kernels keep that factor in fp32. Their error grows with the
attention scale. For inputs in [-1, 1], a model of this rounding keeps their
error inside the 1e-2 bf16 tolerance of the GPU registry tests at the standard
scale 1/sqrt(256); at scale 1.0 a group-mode case reaches about 0.016. The
limit is in AITER's kernels, which AITER launches at any scale too, so the
engine serves them at any scale and does not decline them. The GPU registry
tests run every forward row at scale 1/sqrt(hdim_q).

## Backward tables

Each `fmha_v3_bwd/` directory holds AITER's backward CSV tables, one per kernel
stage: ODO (computes D from O and dO), DQDKDV (computes dQ, dK and dV), and the
post stages DQ_CONVERT and DQ_SHUFFLE (write dQ from the dQ accumulator). The
engine builds its kernel registry from these tables. Column terms used below:

- `atomic32` selects the dQ accumulator: 1 is **A32** (FP32, the default) and
  0 is **A16** (16-bit). The engine knob `hipkernel.sdpa.bwd.accumulator_type`
  picks it (`a32` or `a16`).
- `pssk` is 1 for a kernel that handles any sequence length (padded K/V
  sequence), and `pddv` is 1 for one that handles a head dim padded up to the
  row's head dim. A **psskddv** row has both set.
- `mode` is 0 for batch mode and 1 for group (varlen) mode.

Not every row is dispatched:

- **`fmha_bwd_dq_shuffle.csv` (gfx950 only):** vendored and used. With the A16
  accumulator, gfx950 writes dQ through the DQ_SHUFFLE post stage for every
  D_qk other than 64. gfx942 does not vendor this table: its A16 kernels write
  dQ themselves, and AITER never runs a shuffle stage there.
- **Group (varlen) rows (`mode = 1`):** vendored but not dispatched. The engine
  declines group mode, so it serves batch-mode rows (`mode = 0`) only.
- **hd256 rows (gfx950):** vendored but not dispatched. Every hd256 row is a
  group row, and the engine declines head dims above 128 on gfx950 except
  D_qk = 192 with D_v = 128.
- **hd64 and hd192 x hd192 batch rows (gfx950):** vendored but not
  dispatched. AITER's gfx950 gate (`can_impl_fmha_v3_bwd_gfx950`) sends these
  head dims to CK, and `isBatchShapeSupported` declines them too. The hd192
  dq_convert and dq_shuffle rows are used: D_qk = 192 with D_v = 128 keys the
  post stage on hdim_q 192. The hd192 odo rows are not used, because odo keys
  on padded hdim_v (128).

## Verified Kernel Matrix

Every graph the backward engine accepts runs a kernel AITER ships. The engine
follows AITER's choice of kernel (the gate in `aiter/ops/mha.py` and
`get_heuristic_kernel()` in `mha_bwd.cu`) with these deliberate differences:

- A `pssk = 0` DQDKDV row also needs every (batch, head) offset its kernel
  forms in 32 bits to fit (`psskOffOffsetsFitU32`), which AITER does not check.
  Otherwise the engine takes a `pssk = 1` row.
- When no row has exactly the requested `pssk` and `pddv`, the engine takes the
  smallest qualifying (`pssk`, `pddv`); AITER takes whichever its map yields last.
- The engine serves two cases AITER sends to CK: Sq ≤ 16, and gfx942 bands with
  left bound 0 and right bound 0 or unbounded
  ([SDPA backward test plan §4.6](../../../../../../integration-tests/docs/test-plans/sdpa-bwd.md#46-asm_sdpa_engine-scope)).

Only some of those kernels have been checked on a GPU against the CPU backward
reference. A checked kernel is **verified**.

The verified list is `K_VERIFIED_KERNELS` in `plans/SdpaBwdPlanBuilder.cpp`.
Each entry names an arch, a dtype and the head dims of a DQDKDV registry row.
`isVerifiedDispatch()` counts a dispatch as verified only when all of these
hold:

1. The DQDKDV row matches an entry on arch, dtype, row `hdim_q` and row `hdim_v`.
2. The row is a psskddv row (`pssk = 1` and `pddv = 1`). Calibration ran these
   rows only.
3. The graph's head dims equal the row's head dims, so no head-dim padding
   applies.

| Arch | dtype | D_qk x D_v | Row | Accumulator | Masks |
|---|---|---|---|---|---|
| gfx942 | bf16 | 128 x 128 | psskddv | A32 | 0-3 |
| gfx942 | fp16 | 128 x 128 | psskddv | A32 | 0-3 |

Every gfx942 hd128 psskddv row is an A32 row, so the entries cover A32 only.
The mask is not part of the key; the entries cover each mask those rows ship.

Every other configuration the engine accepts is unverified. That includes
gfx942 at D = 64 or 192, any padded head dim, any `pssk = 0` or `pddv = 0` row
and the A16 accumulator, and every gfx950 configuration.

Dispatch logs once per (arch, dtype, row head dims, mask, accumulator,
verified):

- **Verified** kernel: an `INFO` line confirming the configuration is checked.
- **Unverified** kernel: a `WARN` line stating the results are not validated
  against a CPU reference.

An unverified configuration keeps its `WARN` until it is calibrated on a GPU.
When a configuration is calibrated, add its entry to `K_VERIFIED_KERNELS` and
its row to the table above in the same change.

## Declared tensor alignment

The backend rejects any device pointer that is not a multiple of its tensor's declared alignment
(`TensorAttributes::alignment()`, default 16). The vendored kernels have no documented alignment
contract, so `SdpaFwdPlanBuilder` and `SdpaBwdPlanBuilder` decline a graph whose declared
alignment does not cover the row-load width each tensor is expected to need: 16 bytes for Q, K,
V, O (forward) and Q, K, V, O, dO, dQ, dK, dV (backward), 8 bytes for fp8 K/V, 16 bytes for the
LSE (stats) tensor, 4 bytes for the fp8 descales. The values are the named constants at the top of
`plans/SdpaPlanUtils.hpp` (`K_*_ALIGNMENT_BYTES`) and are to be confirmed by exact-placement
probes.
