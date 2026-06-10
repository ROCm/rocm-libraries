<!--
Copyright Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# rocisa module-builder boundary

This document records the design of the C++ rocisa module-builder foundation
(`tensile_writer/cpp/include/tensile_writer/rocisa_module_builder.hpp`,
bound as `_tensile_writer.subtile.rocisa_builder.ModuleBuilder`, re-exported as
`tensile_writer.subtile.module_builder.ModuleBuilder`) and, specifically, **how
writer-owned state crosses the C++/Python boundary**.

It is the foundation slice for moving the subtile emit loops
(`InstructionEmitter`, `SubtileGREmit`, `SubtileLREmit`, `SubtileScaleEmit`)
into C++ in later slices, without first re-implementing rocisa in C++.

## Why a Python-driven builder (not a C++ rocisa link)

`rocisa` is its own nanobind extension (`_rocisa`) exposing the Python
submodules `rocisa.code`, `rocisa.instruction`, and `rocisa.container`. There
were two ways to let C++ build a rocisa `Module`:

1. **Link `_tensile_writer` against the rocisa C++ library** and construct
   `rocisa::Module` directly. Rejected: it drags the whole ISA (and transitively
   HIP) build surface into the deliberately dependency-light migration wheel
   (`cpp_migration/CMakeLists.txt`), and it requires cross-extension nanobind
   type sharing so the returned `rocisa::Module` is recognised by `_rocisa`.

2. **Hold cached `nb::object` handles to the rocisa Python API and construct
   objects through it.** Chosen. No new build/link dependency; the rocisa
   dependency is runtime-only (an `import`, exactly like `Kernel.py`'s
   `from rocisa.code import Module`). The returned object *is* a real rocisa
   `Module` — the existing rocisa pass pipeline cannot tell it was assembled
   from C++.

The builder API speaks in backend-neutral terms ("items", "modules"), leaving
room for a StinkyTofu construction backend later. StinkyTofu is **not** used for
gfx950 subtile today (it emits through the rocisa string/Module path), so this
foundation targets rocisa.

## The boundary contract

**The builder owns no writer state.** This is the single invariant that keeps
register allocation and label minting authoritative in Python while the *shape*
of the emitted module moves to C++.

| Writer-owned state | Authority | How it crosses to the builder |
|---|---|---|
| VGPR / AGPR / SGPR indices (`writer.vgprPool.checkOut`, `agprPool`, `allocTmpSgpr`) | Python writer register pools | Resolved to plain `int` indices on the Python side, passed to `ModuleBuilder.vgpr(reg)` / `.sgpr(reg)`. The builder never checks out / checks in registers. |
| SGPR symbolic names (`sgpr("LoopCounterL")`) | Python writer / kernel | Passed as a `str` to `.sgpr(name)`. |
| Label names (`writer`-minted, e.g. `SkipTo...`, `LoopBeginL`) | Python writer label minting | Passed as a `str` to `.label(label_name, ...)`. The builder does not mint or uniquify labels. |
| `writer.states` scalars (`laneSGPRCount`, `unrollIdx`, `wavefront`, …) | Python writer | Read on the Python side and passed as `int` arguments to the relevant builder method. |
| Tail-loop helper vgprs (`_tail_vDiff`, `_tail_boundaryMask[]`) | Python `InstructionEmitter` (lifetime spans many leaves) | Their *indices* are passed in per call as `int`s; the builder holds no reference and does not manage their lifetime. |
| `kernel[...]` config | Python | Read on the Python side; only the derived scalars/decisions cross. |

Equivalently: **only `int`, `str`, `bool`, and already-built rocisa `nb::object`
items cross into the builder.** The builder assembles rocisa Items from those
primitives and returns rocisa objects. It never holds a handle to the writer,
its pools, its label manager, or the kernel dict.

### Consequences

- Register allocation stays in Python. A ported C++ emit loop computes operand
  *indices* (today already done by the data-only plans in `tile_info.hpp`),
  checks the registers out in Python, then calls the builder with those
  indices. Checkout/checkin ordering is therefore unchanged and observable.
- Labels remain globally unique because Python still mints them.
- The builder is reentrant and stateless across calls (one instance can be
  reused for a whole pass purely as a handle cache).

## API surface (foundation slice)

`ModuleBuilder` provides:

- **Factories:** `module(name)`, `text_block(text)`, `label(name, comment, alignment)`,
  `vgpr(reg, size)`, `sgpr(reg, size)`, `ds_modifiers(offset, na)` (`na`
  defaults to 1; pass `na=2` for dual-address DS instructions).
- **Generic instruction hook:** `instruction(class_name, args, kwargs)` builds
  `rocisa.instruction.<class_name>(*args, **kwargs)` — the open-ended path for
  any instruction a future ported leaf needs.
- **Mutation:** `add(module, item)`, `add_comment(module, text)`,
  `add_comment_align(module, text)`, `flatitems(module)`.
- **Representative typed leaves** (to demonstrate the mechanism and anchor the
  parity test): `barrier(comment)` (= `InstructionEmitter.emit_sync`),
  `wait_lr(comment)` (= `InstructionEmitter.emit_wait_lr`), and
  `single_item_module(item, name)`.

These two leaves were chosen first because they consume **no** writer state —
the simplest possible boundary — so the foundation can prove byte-identical
rocisa output before any register-carrying leaf is ported.

## GR / LR data-movement leaves (grc.109)

`hipblaslt_incremental_refactor-grc.109` ports the subtile GR/LR
*data-movement* emit leaves onto the builder. The rocisa construction now lives
in C++; the Python `SubtileGREmit` / `SubtileLREmit` functions are reduced to
thin boundary calls that get the data-only C++ plan, resolve writer register
state, and delegate to the builder:

| Builder method | Python leaf it replaces | Writer state resolved in Python |
|---|---|---|
| `single_buffer_load(tc, is_glc, is_slc, is_nt, offsetK, grBaseId, m0Offsets, soffset, voffs)` | `SubtileGREmit.emitSingleBufferLoad` | `soffset` (shared SGPR ref object or int 0) and per-load `voffs` VGPR indices (from `localSubtilesRegister` / `sharedVgprGROffset`) |
| `single_ds_read(tc, sId0, sId1, subIterK, dstVgpr, regsPerDsRead, offset, dstRegOffsets, addrVgprs)` | `SubtileLREmit.emitSingleDsRead` | destination tile base VGPR and per-read `sharedVgprLROffset` address indices |
| `gr_ptr_update(tc, inc)` | `SubtileGREmit._emitGRPtrUpdate_TLU0` | `inc` = `tileInfo.depthUBytes` |
| `gr_lds_buffer_swap(tc)` | `SubtileGREmit._emitGRLDSSwap_TLU0` | none (uses fixed `Srd`/`Swap` sgpr names) |
| `lr_lds_buffer_swap(tc, voffs, vswaps)` | `SubtileLREmit._emitLRLDSSwap_1x2` | `sharedVgprLROffset` / `sharedVgprLROffsetSwap` VGPR indices |

The `skip` predicate of `single_buffer_load` (loadRatioGR>1 duplicate-load
suppression) stays in Python: the leaf returns an empty `Module` and never calls
the builder for skipped subtiles. `add_comment0(module, comment)` was added to
support the `//`-style comment the swap leaves emit. `ds_modifiers` gained the
`na` parameter (the grc.107 review's `fk8` note) so dual-address DS leaves can
be expressed; the AB ds_read path uses the `na=1` default.

The GR/LR *offset-assignment* math (`graTileAssignment` / `lraTileAssignment`,
already C++-plan-driven) is out of scope for this slice and unchanged. The
MX-scale GR/LR data-movement leaves are ported in the follow-up `grc.111` slice
(see the section below).

## MX scale GR / LR data-movement leaves (grc.111)

`hipblaslt_incremental_refactor-grc.111` ports the MX scale-factor (`MXSA` /
`MXSB`) GR/LR *data-movement* emit leaves onto the builder, following the same
boundary pattern as the AB leaves. `SubtileScaleEmit.py` is reduced to a thin
boundary facade: the data-movement functions resolve writer register state and
delegate the rocisa construction to C++.

| Builder method | Python leaf it replaces | Writer state resolved in Python |
|---|---|---|
| `scale_gr_load(tc, is_glc, is_slc, is_nt, voff)` | `SubtileScaleEmit.globalReadDoScaleSubtile` | `sharedVgprGROffset[0]` VGPR index; `NonTemporal<tc>` flags |
| `scale_ds_read(tc, vdst, addrVgpr, dsOffset, scaleGroupIdx, k)` | `SubtileScaleEmit.emitSubtileScaleDsRead` (+ the `InstructionEmitter.emit_lr` scale inline) | destination tile VGPR, `sharedVgprLROffset[0]` address index, DS offset; `k<0` omits the K index from the comment (PGR=0 path) |
| `scale_gr_ptr_update(tc, inc)` | `SubtileScaleEmit.emitScaleGRPtrUpdate` | `inc` = `lrSubtileSize * lrGlobalSubtileGrid[1]` |
| `gr_lds_buffer_swap(tc)` (reused) | `SubtileScaleEmit.emitScaleGRLDSSwap` | none (byte-identical to the AB leaf with `tc` = `MXSA`/`MXSB`) |
| `lr_lds_buffer_swap(tc, voffs, vswaps)` (reused) | `SubtileScaleEmit.emitScaleLRLDSSwap` | `sharedVgprLROffset` / `sharedVgprLROffsetSwap` VGPR indices |

The MX scale GR/LR LDS-swap leaves are byte-identical to the AB swap leaves, so
they reuse `gr_lds_buffer_swap` / `lr_lds_buffer_swap` with the scale component
tag rather than adding new methods. The `MXBlockA`/`MXBlockB` guard and the
duplicate-load suppression stay in Python (the leaf returns an empty `Module`).

The scale *offset-assignment* emission (`graTileAssignmentScaleSwizzled` /
`lraTileAssignmentScaleSwizzled` / `globalReadScaleSwizzledDTLInitCommonSgpr`)
stays in Python because it allocates from the writer's register pools; it
already sources its scalar math from the C++ `MXScaleTileInfoQuery`
offset-assign plans (grc.105) and is unchanged by this slice.

## Verification

`Tensile/Tests/unit/test_subtileRocisaModuleBuilderCpp.py` is the smoke parity
test: it pins rocisa to gfx950 and asserts that modules/items built by the C++
`ModuleBuilder` render to **exactly** the same assembly string as the
equivalent objects built directly with the rocisa Python API (the construction
the Python emit path performs today). It also exercises the generic
`instruction(...)` hook and the `vgpr`/`sgpr`/`ds_modifiers`/`flatitems`
helpers against a `DSLoadB32` built both ways.

## Non-goals for the foundation slice (grc.107)

- The foundation slice itself moved no subtile emit loop to C++; the AB GR/LR
  data-movement leaves were ported in the follow-up `grc.109` slice and the MX
  scale GR/LR data-movement leaves in `grc.111` (see the sections above). The
  GR/LR/scale *offset-assignment* loops remain Python (rocisa construction
  driven by C++ data plans).
- No parallel production path / env-var switch (`TENSILE_WRITER_CPP` and
  dual-path patterns are explicitly forbidden). The builder is additive
  infrastructure; nothing selects it in production yet.
- No StinkyTofu backend (not used for gfx950 subtile).
