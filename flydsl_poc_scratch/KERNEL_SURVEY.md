# flyDSL kernel survey — what ships, what's pullable, where authors keep kernels

Answering the three questions: (1) what kernels flyDSL ships, (2) what's here to
pull into hipDNN, (3) whether kernel authors contribute into flyDSL or keep
kernels in some other repo/location.

## 1. What does flyDSL itself ship? → **No kernels. It's a compiler.**
Verified against **both** the pinned 0.1.6 and the current latest **0.3.2**:

| version | top-level subpackages | kernel modules shipped |
|---------|----------------------|------------------------|
| 0.1.6 (installed) | `compiler, expr, runtime, autotune, utils, _mlir` | **0** |
| 0.3.2 (latest, wheel inspected) | `compiler, expr, runtime, extension, utils, _mlir` | **0** |

No `flydsl.kernels`, no `examples/`, no attention/flash/gemm/moe/conv modules in
either wheel. flyDSL is purely the tile DSL + MLIR/LLVM-23 backend (bundles its
own `flir`/`fly` dialect and LLVM). The only version-to-version change relevant
here is `autotune` → `extension` (a compiler-side feature, not kernels).

**Implication for the newer release:** swapping 0.1.6 → 0.3.2 does *not* surface a
kernel catalog. It would only get us newer compiler/dialect features. Not needed
for Phase 0/A; revisit only if Phase B attention needs a 0.3.x compiler feature.

## 2. What real kernels exist to pull into hipDNN? → **aiter's set (MFMA/gfx9)**
The production flyDSL kernels live in **aiter**, not flyDSL:
`third_party/aiter/aiter/ops/flydsl/kernels/`
- `moe_gemm_2stage.py` — MoE GEMM, 2-stage pipeline
- `mixed_moe_gemm_2stage.py` — mixed-dtype MoE GEMM
- `mfma_preshuffle_pipeline.py` — MFMA pipeline w/ preshuffle
- `mfma_epilogues.py` — MFMA epilogues
- `reduce.py` — reduction helper (MFMA-free; the one RDNA-friendly piece)

All are **low-level `flir` API** (not the high-level `@kernel`/`@jit` used in the
Phase 0 vadd POC) and all target **MFMA / gfx9**. **There is no attention/flash
kernel in the flyDSL set** — attention on AMD currently comes from AITER ASM /
CK / AOTriton, not flyDSL. So Phase B ("flyDSL attention vs rocKE vs AITER ASM")
means **writing** a flyDSL attention kernel, not pulling an existing one.

## 3. Do authors contribute kernels into flyDSL, or keep them elsewhere?
→ **They keep them elsewhere.** flyDSL upstream is compiler-only by design;
kernel authors (the aiter/MoE team) keep their kernels in their consumer repo
(`aiter/ops/flydsl/`). This matches the tool-vs-content split: flyDSL is the
compiler, each consumer owns its kernels. For hipDNN this means we treat flyDSL
as a build-time compiler and **own our attention kernel source in-tree** (the
hipDNN worktree), compiled build-time to HSACO and executed via the `hsaco` UKD
escape hatch. We are not blocked on, and should not expect, an upstream flyDSL
kernel library.

## Consequences for the hipDNN integration plan
- **Phase A** (any flyDSL kernel end-to-end through hipDNN): use the high-level
  API POC we already have working (`vadd.py`); emit HSACO at build time; load via
  `hsaco` descriptor. No dependency on any shipped kernel.
- **Phase B** (attention bake-off): budget for **authoring** a flyDSL attention
  kernel in the low-level `flir` API (that's what all real flyDSL kernels use),
  targeting gfx950 MFMA. No existing flyDSL attention to lift.
