# ROCKE Refactoring Backlog (Prioritized)

Consolidated from the parallel architecture-review buckets (gemm-family +
helper/transform/launch reviews) and the perf-drift investigation. Findings are
de-duplicated and grouped by THEME, then ranked impact/effort within each theme.

**Status of local suites (gfx950 / MI355X):**
- `tests/test_rocke.py` — **245 passed** (2.8s)
- `tests/instances/test_rocke_multiarch.py` — **26 passed** (0.24s)

Both green. None of the items below are regressions; they are maintainability /
clarity / DRY / perf-enablement opportunities. Effort: **S** ~hours, **M** ~half-day,
**L** ~1-2 days. Items marked **[HIGH]** are the highest-leverage.

---

## Theme 1 — MmaOp-contract migration (legacy ISA-named MMA -> unified contract)

The canonical pattern is in `gemm_universal.py`: `_resolve_mma_op`
(`arch.mma.op_for_shape`) + `_emit_mma` (`b.mma(op,...)`) + `op.a_layout()/
b_layout()/c_layout()` for all lane/fragment coords. Several builders predate the
contract and still hand-dispatch `b.mfma_f32_*` / hardcode lane math. Migrating
them deletes the legacy `_emit_mfma`/`_mfma_atom_widths` shims and makes
MFMA-only paths WMMA-capable on gfx1151 for free.

| Rank | File / where | Change | Impact | Effort |
|---|---|---|---|---|
| 1 **[HIGH]** | `instances/common/gemm_universal.py` `_emit_mfma_acc_scatter` (consumed by `_emit_epilogue_default` + `_emit_epilogue_cshuffle`) | MFMA accumulator->output scatter is still hand-coded with magic-constant lane math (`row=(i//4)*8+(lane/32)*4+(i%4)` for 32x32; the 16x16 base) — but the former four-block duplication (default-32/16, cshuffle-32/16) has already been **consolidated** into one shared `_emit_mfma_acc_scatter` helper called by both epilogues. The WMMA path in the same file already uses `op.c_layout().coord(b,lane,i)`. Remaining win: drive the MFMA scatter from `op.c_layout()` too so the magic-constant math goes away. (Guard CDNA byte-identity.) | maintainability | M |
| 2 | `helpers/mfma_gemm_inner.py` `mfma_k_loop` -> `helpers/atoms.py` `MfmaAtom.emit` (372-401) consumed by `mfma_gemm.py` `build_mfma_gemm` (235-323) | `build_mfma_gemm` is 100% legacy: `atom.emit()` hardcodes `b.mfma_f32_16x16x16/16x16x32/32x32x8_f16` dispatch. Resolve op once via `target.mma.op_for_shape(...)` and emit via `b.mma(op,...)`; use `op.c_frag_len` for zero-acc. Unblocks WMMA on gfx1151 and is a step toward retiring `gemm_universal`'s `_emit_mfma`/`_mfma_atom_widths` (still consumed by `moe_gemm_fused`, `moe_fused_mega_fp8`, and `conv_implicit_gemm`). | maintainability | M |
| 3 | `instances/common/streamk_gemm.py` `_process_macro_tile` (323-407), `spec.atom` (149-159) | Same legacy path: `spec.atom` hardcodes `MfmaAtom.f16_16x16x16()/f16_32x32x8()`, emits via `mfma_k_loop`/`store_acc_to_global` + `atom.lane_to_output` hand math. Pinned to legacy 16x16x16/32x32x8 (cannot use K-packed 16x16x32 / 32x32x16). **Inherits the fix from item 2 for free** once `mfma_gemm_inner` is contract-driven; otherwise migrate `spec.atom` to `op_for_shape`. | maintainability | M |
| 4 | `instances/common/gemm_universal.py` `emit_mfma_phase` CDNA lane mapping | CDNA branch hardcodes `m_in_atom=lane%warp_tile_m`, `k_blk=lane/warp_tile_m`, `n_in_atom=lane%warp_tile_n` (the former dead (16,16)-vs-(32,32) if/else has already been collapsed to one decode). WMMA branch (`_emit_wmma_phase`) already sources coords from `op.a_layout()/b_layout()`. Source MFMA coords the same way and unify the two phase emitters. (Verify CDNA byte-identity.) | maintainability | M |

---

## Theme 2 — CK tensor-transform adoption (hand-rolled stride math -> TensorDescriptor DAG)

`transforms.py` (`TensorDescriptor` + unmerge/merge/embed/pad/replicate) is the
intended way to express layout/offset math; `conv_implicit_gemm` is the exemplar.
These kernels hand-compute strides/offsets inline.

| Rank | File / where | Change | Impact | Effort |
|---|---|---|---|---|
| 5 **[HIGH]** | `instances/common/gemm_universal.py` `emit_load_phase` DTLA branch (891-1000) | DirectToLDS path hand-computes `chunk_idx->(row,col)` and `off_elems=batch_off+(block_m_off+row)*K+(k_off+col)` inline (954-988), duplicated near-verbatim for A, B, and prefetch-parity offsets. Non-DTLA branch right below uses `make_tile_window` over a (1,K,1) descriptor. Express DTLA chunk->element as `TensorDescriptor` unmerge + the existing global view so one DAG is shared by A/B and both branches. | maintainability | L |
| 6 | `helpers/mfma_gemm_inner.py` `load_b_col_strided_scalars` (217-252) | B-load computes `addr=(k_base+j)*N+n_col` by hand per K-element with explicit mul/add + scalar-load loop. Re-express as a (K,N) row-major `TensorDescriptor` with lane->(n_col,k_base) coord (mirror `gemm_universal` A/B views with strides=(1,K,1)). Lets a future LDS-transpose (`TransposeLdsReader` per docstring) slot in as a transform op rather than a rewrite. | clarity | M |

---

## Theme 3 — Launch-semantics consolidation (grid/launch boilerplate, persistent, workspace)

`runtime/launcher.py`, `helpers/grid.py`, `helpers/persistent.py`,
`helpers/spec.py:ceil_div_grid`, `WorkspacePool`.

| Rank | File / where | Change | Impact | Effort |
|---|---|---|---|---|
| 7 **[HIGH]** | `instances/common/grouped_gemm.py` `GroupedGemmLauncher.__call__` (225-261) + `build_grouped_gemm_single_launch` | Launcher hand-builds `LaunchConfig` with inline ceil-div grid per group, duplicating `helpers/spec.py:ceil_div_grid`. Use `ceil_div_grid`. Bigger win (partially landed): `build_grouped_gemm_single_launch` now emits a real single-launch kernel for the **uniform-shape** case — a batched universal-GEMM body with `grid.z = num_groups` that decodes `block_id_z` as the group index and offsets the A/B/C base pointers, run via `GroupedGemmSingleLaunchRunner`. The remaining gap is the **fully-variable-shape** single launch (per-group `(M[g], N[g], K[g])` decoded from a descriptor table + runtime-K loop + out-of-range tile guard); that path — and a `helpers/persistent.persistent_tile_for_each` + `WorkspacePool` variant — is the unimplemented launch-overhead win. | perf | L |

---

## Theme 4 — Helper reuse / DRY

Repeated spec/grid/epilogue code that should be a shared helper.

| Rank | File / where | Change | Impact | Effort |
|---|---|---|---|---|
| 8 | `instances/common/batched_gemm.py` `to_universal_spec`; same in `grouped_gemm.py`, `flatmm.py` | **Block-size half landed:** the shared `helpers/spec.py:derive_block_size(tile, wave_size)` + `WarpTileBlockSizeMixin` now exist and are the single source of truth — `UniversalGemmSpec`, `BatchedGemmSpec`, `GroupedGemmSpec`, and `FlatMMSpec` all subclass the mixin instead of re-deriving `block_size` in `__post_init__` (`gemm_multi_d`/`gemm_multi_abd` embed a `UniversalGemmSpec` directly, so they never duplicated it). Remaining: the near-identical `to_*_spec()` / `kernel_name()` delegation across the wrapper specs is still copy-pasted and could share a base. | maintainability | S |
| 9 | `instances/common/gemm_multi_d.py` `_MultiDEpilogue` | **Done:** the former `_TiledMultiDEpilogue` / `_VectorizedMultiDEpilogue` pair is now a single `_MultiDEpilogue(FusedEpilogue)` driven by a `_load_kind` flag (`"tiled"` vs `"vector"`, sourced from `GemmMultiDSpec.d_load_kind`); the `from_ops` classifier, `_residual_kinds/_dtypes`, `off_base` hoist, and add/mul combine are shared. Remaining nicety: the fadd/fmul-over-`vec_extract` combine loop is still a candidate `helpers/fuse.py` primitive. | maintainability | S |
| 10 | `instances/common/gemm_universal.py` batched offsets (701-754); same pattern in `mfma_gemm.py` (271-272), `streamk_gemm.py` (348-349) | Flatten `(bx,by)->wgid`, ceil-div M/N tiles, SGPR-pin `block_m_off/block_n_off` is GEMM grid boilerplate in 3 builders. Extract a `helpers/grid.py` companion (alongside `chiplet_aware_super_tile_dynamic`) returning SGPR-pinned `(block_m_off, block_n_off)`. Also: **fix stale `batched_gemm.py` docstring** (27-44) claiming no `to_sgpr_u32` wrap exists — code at 709-712 already wraps. | maintainability | M |

> Tile-distribution adoption (`helpers/distribution.py` + `helpers/geometry.py`):
> no standalone item surfaced beyond the lane-math cases already folded into
> Theme 1 (items 1 & 4) — those manual lane->element maps are the
> tile-distribution adoption targets and are tracked there.

---

## PERF section

### Unified-attention drift (gfx950 / MI355X): ~25-40% slower than README
- **Status: ENVIRONMENT / BASELINE — not a code regression.** Do not "fix" kernels.
- **Drift is real & repeatable** (not box noise): with warmup>=10, iters>=30 on
  torch's stream (same HIP-event timer as README), CK latency is rock-stable
  run-to-run (decode ck-auto 58-61us, std ~1us across 8 repeats).
- **Triton reproduces the README on this device** (tri-auto decode 53us vs README
  57.5us; prefill 28us vs 30.4us), but **CK is systematically slower than
  README's CK** (decode ck-auto 58 vs 45.7us; ck-3d 61 vs 45.1us; prefill ck-3d
  56.9 vs 43.8us). Every auto lane now loses (0.80-0.91x) where README claims
  1.17-1.35x wins.
- **Isolated to the 3D split-KV (segment+reduce) path** that decode/auto route to.
  The forced **2D kernel reproduces the README exactly** (prefill ck-2d 23.2us vs
  22.0us), and the **3D kernel body/ISA is byte-intact** with every README-credited
  optimization still present. Conclusion: the slowdown is in the 3D
  segment/reduce path's environment/baseline (driver/runtime/clocks on this MI355X
  box), not in kernel source. **No refactor action; track as a baseline note.**

### Qwen example drift
- Same class of environment/baseline drift on this gfx950 box; no source
  regression identified. Measure with warmup + multiple iters + medians (box is
  perf-noisy: Triton swung 62->90us run-to-run). Treat reported deltas as
  env-dependent, not as code defects.

**Perf verdict:** drift is real and repeatable but environmental/baseline (3D
split-KV path + noisy box), **not a code regression**. The only perf-coded
backlog item (grouped-GEMM single-launch / persistent path): the
uniform-shape single launch has landed; the remaining launch-overhead win is the
fully-variable-shape / persistent path, still unimplemented.
