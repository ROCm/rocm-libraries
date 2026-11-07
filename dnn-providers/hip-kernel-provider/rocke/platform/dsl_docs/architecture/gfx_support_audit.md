# rocKE — gfx architecture support audit (AICK-1582)

An inventory of which gfx targets rocKE supports, per component, distinguishing
**common code paths** (`core`, `helpers`, `instances/common`, `examples/common`,
`tests`) from **arch-specific lanes** (`instances/gfx*`, `examples/gfx*`). Derived
from the code; **device names only**, no performance data.

**Out of scope** (tracked elsewhere): fixing the gaps below (follow-up stories) and the
silent gfx950 device-detection fallback (AICK-1541 / AICK-1546).

## Targets in scope

| Target | Catalog (`arch_specs.json`) | ISA backend class | `instances/` lane | `examples/` lane | MMA family (wave) |
|---|---|---|---|---|---|
| gfx908 | ✓ | `Gfx9MfmaBackend` | ✗ | ✗ | MFMA (wave64) |
| gfx90a | ✓ | `Gfx9MfmaBackend` | ✗ | ✗ | MFMA (wave64) |
| gfx942 | ✓ | `Gfx9MfmaBackend` | ✓ | ✓ | MFMA (wave64) |
| gfx950 | ✓ | `Gfx950Backend` | ✓ | ✓ | MFMA (wave64) — **default/baseline** |
| gfx1151 | ✓ | `Gfx11RdnaBackend` | ✓ | ✓ | WMMA (wave32) |
| gfx1201 | ✓ | `Gfx12RdnaBackend` | ✓ | ✓ | WMMA (wave32) |
| gfx1250 | ✓ | `Gfx1250Backend` (extends `Gfx12RdnaBackend`) | ✓ | ✓ | WMMA (wave32) |
| gfx11-generic | ✓ | `Gfx11RdnaBackend` | n/a | n/a | WMMA (wave32) |
| gfx1100-gfx1103, gfx1150, gfx1152-gfx1154 | ✓ (points at gfx11-generic) | `Gfx11RdnaBackend` | n/a | n/a | WMMA (wave32) |
| gfx12-generic | ✓ | `Gfx12RdnaBackend` | n/a | n/a | WMMA (wave32) |
| gfx1200 | ✓ (points at gfx12-generic) | `Gfx12RdnaBackend` | n/a | n/a | WMMA (wave32) |

Source: `core/arch/data/arch_specs.json`, `core/isa/backend.py:602` (`BACKEND_REGISTRY`),
`instances/`, `examples/`.

## Per-component support

### Core — arch catalog (`core/arch/`)
Every target above is populated in `arch_specs.json` with wave size, LDS capacity,
`vmcnt` width, MMA-atom catalog, memory-capability bits, and `isa_triple`.
`known_arches()` / `arch_from_isa()` / `ArchTarget.from_gfx()` read this SSOT, and
`arch_specs.json` is the list of which MMA ops each target has.

### Core — ISA backends (`core/isa/backend.py`)
`backend_for(arch)` resolves the gfx string to a backend class; a registered-but-
unpopulated (forward-declared) target raises a `KeyError` with a "forward-declared ...
no arch_specs.json metadata" message. Every row registered today has its metadata.
Capability divergences:

| Capability | Gfx9Mfma (908/90a/942) | Gfx950 | Gfx11Rdna (1151, generic) | Gfx12Rdna (1201) | Gfx1250 |
|---|---|---|---|---|---|
| MMA emission | MFMA | MFMA | WMMA (`emit_wmma`) | WMMA | WMMA |
| `s_waitcnt` | legacy split | legacy split | gfx11 layout | gfx11 layout | split counters (no legacy) |
| buffer SRD word3 | CDNA | CDNA | RDNA | RDNA | RDNA |
| async-LDS counter | ✗ | ✗ | ✗ | ✗ | ✓ |

### Core — lowering (`core/lower_llvm.py`, `core/lower_hip.py`)
Substantially **arch-neutral**: CFG, loops, memory ops, and type mapping are target-
independent. All arch-specific codegen (MMA family, `s_waitcnt`, buffer SRD, transpose
LDS reads) is funnelled through the single `ISABackend` selected at lowering init. The
default arch when none is given is **gfx950** (`core/backend.py`).

### Building blocks — MMA atoms (`helpers/atoms.py`)
Availability is **data-driven** via `ArchTarget.mma.has_shape()`; there are no hardcoded
per-arch `if gfx…` atom guards in Python.

Which MMA instructions each target has is recorded in one place: the `mma` list of
its row in `core/arch/data/arch_specs.json` (for a member that points at a generic
target, the generic's row).

### Building blocks — common code (`helpers/`, `instances/common/`)
Mostly **arch-agnostic**: `distribution`, `tensor_view`, `pipeline`, `reduction`,
`schedule`, and `epilogues` operate in the tile/atom domain (epilogues use the atom's
own lane→output map, so they work for MFMA and WMMA alike). Most `instances/common`
builders import `ArchTarget` **only for spec validation** (`is_valid_spec` →
`has_shape`), not for codegen branching. The one genuinely arch-polymorphic dispatch is
`instances/common/gemm_universal.py::_mma_family()`, which selects MFMA vs WMMA by wave
size — one kernel body, both ISAs.

### Op-family coverage
| Family | gfx90a | gfx942 | gfx950 | gfx1151 | gfx1201 | gfx1250 |
|---|---|---|---|---|---|---|
| GEMM (`gemm_universal`, `mfma_gemm`) | ✓ | ✓ | ✓ | ✓ (WMMA) | ✓ (WMMA) | ✓ (WMMA) |
| Convolution (direct / implicit-gemm) | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Attention (`mfma_attention`) | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| MoE (fused / mega) | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| block-scale / MX GEMM | fp8: gfx942+ · MX: gfx950 | — | — | ✗ | ✗ | ✗ |

RDNA (gfx1151/1201/1250) currently has **GEMM only**; conv, attention, and MoE are
MFMA-authored and have no WMMA path.

### Tests (`tests/`)
| Target | Multiarch (CPU lowering) | Golden IR | Numeric (GPU) |
|---|---|---|---|
| gfx908 | ✓ | ✓ (one GEMM case) | ✗ (no numeric gate) |
| gfx90a | ✓ | ✓ | via CDNA gate (no independent numeric case) |
| gfx942 | ✓ | ✓ | ✓ |
| gfx950 | ✓ | ✓ (heaviest) | ✓ |
| gfx1151 | ✓ | ✓ | ✓ (WMMA subset) |
| gfx1201 | ✓ | ✓ | ✗ (no numeric gate) |
| gfx1250 | ✓ | ✓ (mostly GEMM) | ✗ (no numeric gate) |
| gfx11-generic | ✓ | ✓ | ✗ |
| other gfx11-generic / gfx12-generic members | ✓ | ✓ (one GEMM case each) | ✗ |

Source: `tests/instances/test_rocke_multiarch.py`, `tests/golden/rocke_representative_ir_sha256/`,
`tests/instances/test_rocke_numeric.py`.

## Common vs arch-lane summary
- **Common / arch-polymorphic:** `core`, `helpers`, `instances/common`, `examples/common`,
  and the multiarch CPU test suite carry all targets through the catalog + atom dispatch.
- **Arch lanes:** `instances/gfx*` and `examples/gfx*` hold tuning and proof-of-concept
  kernels. Note `instances/gfx1151/wmma_gemm.py` is a POC now superseded by
  `gemm_universal` (`build_universal_gemm(spec, arch="gfx1151")`).

## Known issues / workarounds
1. **gfx950 default + silent device-detection fallback** — out of scope here (AICK-1541 / AICK-1546).
2. **gfx908** -- catalog + backend present (compiler-checked facts), but no `instances/`/`examples/` lane and no numeric gate. Lowering-only today.
3. **gfx90a** — catalog + backend present, but no `instances/`/`examples/` lane and no independent numeric gate (folded into the CDNA gate). Lowering-only today.
4. **RDNA breadth** — gfx1151/1201/1250 support GEMM only; WMMA conv/attention/MoE are unimplemented.
5. **Numeric gaps** — gfx1201 and gfx1250 have no on-GPU numeric test gate; golden IR for gfx1250 is mostly GEMM.
6. **Arch registry and MMA traits key on DISJOINT namespaces** — a raw target string matches at most one, and there is no bridge between them. Same class as item 2, generalized: a target present in one registry and absent from the other.

   Measured (`arch_specs.json` keys vs the union of `supported_targets` over `load_mma_traits()`):

   | | targets |
   |---|---|
   | in both | `gfx1250`, `gfx908`, `gfx90a`, `gfx942`, `gfx950` |
   | arch registry only | `gfx11-generic`, `gfx1151`, `gfx12-generic`, `gfx1201` |
   | traits table only | `gfx11`, `gfx12` |

   The arch registry holds **specific targets**; the traits table holds a mix of specifics and **family umbrellas** (`gfx11`, `gfx12`). Overlap is partial in **both** directions, so a lookup miss in one namespace must never be reported as "this target is unsupported".

   **`target_family` does not bridge them.** It matches a traits key for exactly one target (`gfx950`), and only because that target's family string happens to be its own name — the other values (`gfx9_mfma`, `gfx11_rdna`, `gfx12_rdna`, `gfx12_cdna`) are lowering-path labels and match **zero** traits rows. Closing this needs a new ordered match-key set per target, resolved once and passed as an object; see `multi_arch_data_layout.md` → "Resolving a target against the MMA catalog".

   Related: `MmaTraits.supports()` takes a raw `str` and tests exact membership, which is the signature that makes the mismatch reachable.
7. **A matrix capability flag is not sufficient to emit a primitive** — a registered target can report a matrix capability TRUE while the catalog offers zero usable rows for it. Authority for "can I emit this" is the row carrying the layout parameters, not the flag.

## Proposed gap-closure priority
Ordering for follow-up stories (to be confirmed with the team):

1. **P1 — numeric validation** for targets that already have lanes but no on-GPU gate: gfx1201, gfx1250, and an independent gfx90a case.
2. **P2 — RDNA building-block breadth**: WMMA conv / attention / MoE, if RDNA is a delivery target.
3. **P3 -- gfx908**: a numeric gate on gfx908 hardware for its catalog row.
4. **P4 — golden IR breadth** for under-covered targets (gfx1250 non-GEMM families).
