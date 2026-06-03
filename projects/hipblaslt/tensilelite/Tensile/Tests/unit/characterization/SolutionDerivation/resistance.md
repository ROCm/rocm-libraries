# Resisting items — `Solution` derivation slice 3a

The small derivation statics are fully covered. Two classes of residue remain,
both reachable-in-principle but requiring inputs this slice does not stand up:
**(a)** cap-gated arms needing a non-gfx942 ISA, and **(b)** the deep reject
matrices of the predicates + the two giant `assign*` methods, which need a
dtype/ISA/MI **config sweep** (slice 3b). New file in the per-target dir per the
add-only rule.

## (a) Cap-gated arms — unreachable with the gfx942 fixture's isaInfoMap

| Code | Lines | Why unreachable here |
|---|---|---|
| `getMIOutputInfo` WMMA branches | 564-569 | `HasWMMA_V1/V2/V3` are false on gfx942 (an MFMA arch); the fixture's `isaInfoMap` only carries gfx942-family caps via this solution's `ISA`. Reaching them needs a WMMA ISA (e.g. gfx11xx/gfx12xx) state. |
| `isVgprForLocalReadPackingDoable` `HasEccHalf=False` arm | 1083 | gfx942 has `HasEccHalf=True`; the disabling branch needs an ISA without it. |

These are genuinely cap-gated; covering them requires seeding the statics with a
state whose `ISA` is a WMMA / non-EccHalf arch present in `isaInfoMap`. Deferred
to slice 3b (multi-ISA sweep).

## (b) Deep reject matrices — need a config sweep (slice 3b)

| Static | Covered this slice | Remaining |
|---|---|---|
| `isDirectToVgprDoable` (1103-1270) | real-state outcome (A/B) + the `not EnableMatrixInstruction` reject | ~24 further reject branches, each gated on a *DTV-passing* base config (LocalReadVectorWidth, LSU/TLU, conversion, MatrixInstBM/BN, WaveSeparateGlobalRead, VectorWidth/GRVW, SIA, InnerUnroll, UnrollMajorLDS, Sparse, PGR, TransposeLDS, …). Requires a state that passes all *prior* checks before each target reject. |
| `isDirectToLdsDoable` (1275-1399) | real-state outcome (A/B) + `UseSubtileImpl` short-circuit + `not EnableMatrixInstruction` | ~18 further branches (numBytesPerLoad 8/16/<4, MT-not-pow2, LRVW>MIInputPerThread, NumThreads%WavefrontSize, UnrollMajorLDS, WSGR variants, LRVW==2, D/Z-GEMM NumLoadsCoalesced, bpe>bpr, size-mismatch, conversion). Same "passing base + one mutation" need. |
| `assignProblemIndependentDerivedParameters` (576-919) | early-return guard + full happy re-run (gfx942 HSS MI config) + SIA=4 reject | The `EnableMatrixInstruction=False` (ThreadTile) path, the MX / DirectToLds / DirectToVgpr / tailLoopOpt branch thicket, and the other reject arms — each needs a distinct self-consistent config. |
| `assignDerivedParameters` (1419-2487) | preamble (WavefrontSize/MaxLDS/F32Xdl/CUOccupancy) + early guard + full happy re-run | The ~1000-line body's many cap-gated derivation/reject branches — the largest remaining block in `Solution.py`. |

**Why deferred, not forced:** each remaining reject needs a base `state` that
passes every earlier check and trips exactly the target — i.e. a curated config
matrix across dtype × ISA × MatrixInstruction × the relevant flags. That sweep
is a multi-day increment in its own right (slice 3b); this slice banks the
fully-tractable small statics + the happy-path derivation and pins the
real-state predicate outcomes, raising `Solution.py` 38.81% → 41.14% line.

## `setGlobalLoadTileDimClassic` / `depthUIteration` / `_deriveAndValidateMXScaleLayoutAndTransport`

Not driven directly in this slice (they are invoked transitively by the happy
`assign*` re-run, covering their happy paths). Their dedicated branch coverage
is part of the slice-3b sweep.

## Determinism technique (not a gap)

- Small statics use crafted minimal dicts (deterministic, fixture-free).
- Predicate/assign tests seed from a deep-copied real solution state and snapshot
  selected derived scalars + reject outcomes, never env-coupled values.
- The SIA=4 test branches on `rocisa.hasStinkyTofuBackend()` /
  `isSupportedByStinkyTofu` so it pins the correct outcome for whichever rocisa
  build the container has.
