# Plan: switch rocKE AOT producer from FMHA to UnifiedAttention2DTiledSpec

## Goal
Replace the explicit FMHA kernel family (`FmhaMfmaSpec` / `build_fmha_fwd_mfma`) in the
rocKE AOT producer + kpack packaging + rocke-client runtime dispatch with the
`UnifiedAttention2DTiledSpec` path. For this pass, pin only the semantic/shape knobs and
leave every performance knob at its default. End-to-end tests must still launch kernels
via hipDNN through hip-kernel-provider.

## LOCKED SCOPE (user-confirmed)
- Arches: **gfx942 + gfx950 only**. Drop gfx1151 (no tiled-2D spec). Defer gfx1250 (fp8-only, follow-up).
- No fp8: `kv_storage_dtype=None` on every instance ⇒ the gfx1250 divergent default and the
  `use_register_pv`/`use_register_p` name mismatch are **non-issues** this pass.
- Selection: **keep dense selection keys** (adapter maps dense SDPA → seqlen_q=seqlen_k=S; no
  CompileSpec/parseCompileSpec/satisfies change). Unified kernel stays an impl detail behind `sdpa_fwd`.
- Shapes: **mirror the current FMHA set** — fp16, d64 + d128, MHA (hq4/hkv4) + GQA 4:1 (hq8/hkv2),
  mask_mode none — adapted to the unified `compile_spec` field names.

## Grounding (current architecture — verified)

Producer is a **family-agnostic, two-leg CMake-driven pipeline**:
- Leg 1 `aot/tools/rocke_aot_build.py`: reads `kernels/<arch>/<family>/aot_list.json`, validates
  each instance vs JSON schemas, dynamically loads the family **handler** module by `--handler`
  path, and per instance calls `handler.parse_instance_fields -> build_kernel -> compile_kernel(backend="python")`
  (comgr → HSACO, host-only, no GPU), then `handler.emit_sidecar` → `<name>.sidecar.json`.
- Leg 2 `aot/tools/rocke_kpack_pack.py`: verifies HSACO sha256, packs into `rocke_client_<arch>.kpack`
  (zstd-3) under toc_key `rocke/<op>/<family>/<name>`, emits `rocke.aot.bundle/v1` manifest carrying
  `selection`/`launch`/`args_signature` verbatim. Installed to `<plugin>/arch_content/rocke/<arch>/`.

The **handler contract** (`aot/python/rocke_client_aot/instance_schema.py`): module-level `OP`
and `FAMILY` constants + three callables: `parse_instance_fields(instance, source) -> (fields, spec, reason)`,
`build_kernel(spec, *, arch) -> KernelDef`, `emit_sidecar(instance, spec, artifact, hsaco_filename) -> dict`.
`instance_name`/`enrich_args_signature` are FMHA-internal helpers, not framework-required.

C++ runtime dispatch is **mostly generic/schema-driven**. `family`/`cache_key` are opaque
(never used for selection). Coupling points that are FMHA-shape-specific:
1. `AotInstance::CompileSpec` field set + `AotCatalog.cpp::parseCompileSpec` + `SelectionConstraints.cpp::satisfies`
   (selection keys: dtype, layout, seqlen_q/k, num_query_heads/kv_heads, head_size, mask_mode, batch, attrs).
2. `SdpaGraphAdapter.cpp::buildSdpaLaunchInputs` — hard-coded **dense** launch ABI arg names/derivations
   (Q/K/V/O ptrs, `scale_log2`, `seqlen_q/k`, `stride_{q,k,v,o}_{token,head}`).
3. `SdpaGraphAdapter.cpp::sdpaGridSymbols` — fixed symbol table (batch, seqlen_q/k, num_*_heads, head_size, block_size_q/k).

## The crux: ABI + grid mismatch (biggest risk / largest work item)

The unified 2D tiled kernel is **paged-KV**, not dense. Its declared params
(`kernels/gfx950/attention_tiled_2d.py:1155+`):
`output, query, key, value, sinks_ptr, block_tables_ptr, seq_lens_ptr, alibi_slopes_ptr,
qq_bias_ptr, cu_q_ptr, scale, k_scale, v_scale, out_scale, softcap, num_seqs,
block_table_stride, qq_bias_stride_0, ...`

Grid: `(num_kv_heads, total_q // block_q + num_seqs, 1)`; block `(wave_size * num_warps, 1, 1)`.
(FMHA today: grid `(ceil_div(seqlen_q, block_q), num_query_heads, batch)`, block `(wave_size,1,1)`.)

To launch this for a **dense** hipDNN SDPA problem, the adapter/plan must **synthesize paged-KV
launch inputs**: device buffers `block_tables` (identity block map), `seq_lens` (=seqlen_k per seq),
`cu_q` (cumulative Q offsets `[0, S, 2S, ...]`), plus scalars `k_scale=v_scale=out_scale=1`,
`softcap=0`, `num_seqs=batch`, `block_table_stride`, and null `sinks/alibi/qq_bias`. This is **new
device-buffer machinery** in `RockeClientPlan::execute` — not just an arg rename. It is the largest
and riskiest chunk.

## Arch coverage change (decision needed)

| arch    | FMHA family (today) | tiled-2D spec exists? | notes |
|---------|---------------------|-----------------------|-------|
| gfx942  | yes                 | yes                   | rejects fp8 KV; narrow 16x16x16 |
| gfx950  | yes                 | yes (default arch)    | wide-K; full knob surface |
| gfx1151 | **yes**             | **NO**                | only legacy FMHA / WMMA fmha adapter |
| gfx1250 | no                  | yes                   | **requires** fp8e4m3 KV + fixed shape (d64, block32, GQA-8, bf16, no softcap/alibi) |

Switching families **drops gfx1151** (no tiled-2D spec) and **gains gfx1250** (fp8-only, one shape).
No single instance covers all arches: gfx942 rejects fp8, gfx1250 requires it.

## Knob call-outs (the explicit ask)

`UnifiedAttention2DTiledSpec` is **three structurally different dataclasses**, one per arch package,
dispatched by `_tiled_2d_impl(arch)`. Field counts: gfx1250 ≈ 17, gfx942 ≈ 38, gfx950 ≈ 44.

**COMMON to all three (safe to pin by name):**
- Required semantic (no default): `head_size, block_size, num_query_heads, num_kv_heads, dtype,
  use_sinks, sliding_window, has_softcap`
- Optional w/ default: `use_alibi=False, use_qq_bias=False, num_seqs=0, num_warps=1,
  waves_per_eu=None, tile_size=None, block_m_per_warp=16`
- `kv_storage_dtype`: **DIVERGENT DEFAULT** — `None` on gfx942/gfx950, `'fp8e4m3'` on gfx1250.

**Name mismatch:** register-P is `use_register_pv` on gfx942/gfx950 but `use_register_p` on gfx1250.

**ARCH-SPECIFIC (leave at default this pass — do NOT pin):**
- gfx950-only: `use_v_double_buffer, kv_ring_depth, use_staggered_iter_wait, use_sched_barrier,
  sched_barrier_mask, use_q_reread, use_q_direct_reg, use_softmax_mfma_interleave(+mode/groups),
  use_mask_phase_split, use_mfma_32x32` (native).
- gfx942-only: `use_mfma_32x32x8` (K=8 atom), `use_conflict_free_v[_store][_split/_ck_vlds],
  use_k_sliced_ring/ldsseq, use_iglp_opt, use_qk_pv_sched_group_barrier, use_q_direct_global,
  kv_cache_policy, use_global_load_lds_k, use_q_major_grid`.
- Shared-but-heuristic (gfx950 default path): `use_transposed_qk_32x32` + transposed sub-flags,
  `use_register_pv, use_fp8_mfma_qk, use_i64_kv_addr, use_early_v_schedule, use_fast_paged_kv_desc`.

**Pin strategy for AOT:** pin the 8 semantic fields + `dtype` + `kv_storage_dtype`; leave
`num_warps=1, waves_per_eu=None, tile_size=None(→block_size), block_m_per_warp=16` and every
`use_*` perf flag at default. This yields the baseline correct QK→softmax→PV kernel on each arch.
Do **not** route through `_tiled_spec_from_problem` (it auto-selects arch-specific perf knobs);
build the arch spec directly from pinned fields so the AOT output stays deterministic and
knob-free, per the "defaults only" scope.

## Phased implementation plan

### Phase 1 — Producer: unified handler + schemas + instance lists (Python, no GPU)
1. New handler `kernels/common/attention_tiled_2d_aot.py` (mirror `fmha_mfma_aot.py`):
   - `OP="sdpa_fwd"`, `FAMILY="attention_tiled_2d"`, `ABI_VERSION="hipkg-attention-unified/v1"`,
     `ALGORITHM="unified_attention_2d_tiled"`.
   - `parse_instance_fields`: validate the unified `compile_spec` field set (head_size, block_size,
     num_query_heads, num_kv_heads, dtype, sliding_window, use_sinks, has_softcap, kv_storage_dtype,
     canonical_layout, + optional use_alibi/use_qq_bias/num_seqs).
   - `build_kernel`: import the arch spec class via `_tiled_2d_impl(arch)`, construct
     `UnifiedAttention2DTiledSpec(**pinned_fields)` (defaults for all knobs), call
     `build_unified_attention_2d_tiled(spec, arch=arch)`.
   - `emit_sidecar`: emit `selection` (dtype/layout/head_size/num_*_heads/block_size/sliding_window/
     mask/attrs), `launch` (grid_formula `x=num_kv_heads`, `y=total_q//block_q + num_seqs`, `z=1`;
     block `[wave_size*num_warps,1,1]`; tile_sizes), and `args_signature` for the paged-KV param list
     (pointer + scalar ABI). New signature source analogous to `fmha_fwd_mfma_signature` — add a
     `unified_attention_2d_tiled_signature(spec, arch)` probe or extract from the built `KernelDef`.
2. New schema overlay `aot/schemas/attention_tiled_2d/{instance,sidecar}.schema.json`
   (allOf `$ref ../{instance,sidecar}.schema.json`): new `$id`, name pattern, `op` const `sdpa_fwd`,
   `family` const `attention_tiled_2d`, unified `compile_spec`, sidecar cache_key regex + symbol
   pattern (`^rocke_unified_attention_2d_tiled_` or actual kernel_name prefix).
3. Per-arch instance lists `kernels/<arch>/attention_tiled_2d/aot_list.json` for **gfx942, gfx950,
   gfx1250** (see arch table). Start each with a small, representative shape set; gfx1250 must use
   the mandated fixed shape + fp8.
4. Remove the FMHA producer inputs: `kernels/common/fmha_mfma_aot.py`, `aot/schemas/fmha_fwd_mfma/`,
   `kernels/<arch>/fmha_fwd_mfma/aot_list.json` (clean cutover — no dual family).

### Phase 2 — Build system re-point (CMake)
- `aot/CMakeLists.txt`: replace `_ROCKE_FMHA_HANDLER`/`_ROCKE_FMHA_SCHEMA_DIR` with the unified
  handler + schema dir; replace the three `rocke_client_add_aot_instances` calls (drop gfx1151, add
  gfx1250) with `ARCH_DIR .../<arch>/attention_tiled_2d`; update NAME slugs and the numeric-test
  artifact-dir foreach (`sdpa_fwd_fmha_mfma_${arch}` → unified name).
- `aot/tools/rocke_aot_build.py:105,107`: drop the `parsed.fmha_spec` compat fallback + error text.

### Phase 3 — Runtime dispatcher (C++) — the crux
- **Selection keys:** the unified selection uses block_size/num_seqs/sliding_window rather than dense
  seqlen_q/seqlen_k. Decide whether to keep the existing dense selection keys (adapter maps a dense
  SDPA to seqlen_q=seqlen_k=S and matches on head_size/heads/dtype/layout/mask) or extend
  `CompileSpec`/`parseCompileSpec`/`satisfies` with unified keys. Recommended: keep dense selection
  keys (minimal change) since the incoming hipDNN problem is dense; the unified kernel is an
  implementation detail behind the same `sdpa_fwd` op.
- **Launch inputs (`SdpaGraphAdapter::buildSdpaLaunchInputs` + `RockeClientPlan::execute`):** emit the
  paged-KV arg set; allocate/fill device buffers `block_tables` (identity), `seq_lens` (=S), `cu_q`
  (`[0,S,2S,...]`); bind scalars `scale, k_scale=1, v_scale=1, out_scale=1, softcap=0, num_seqs=batch,
  block_table_stride`; null `sinks/alibi/qq_bias`. This is the new device-buffer machinery.
- **Grid symbols (`sdpaGridSymbols`):** add `total_q`, `block_q`, `num_seqs`, `num_kv_heads` (and any
  others the grid_formula references) so `LaunchAbi::evalGrid` can evaluate the unified grid.
- `args_signature` parser/packer (`LaunchAbi::bindArgs/packArgs`) already handles arbitrary
  pointer/scalar signatures — no change beyond matching the new arg names.

### Phase 4 — Tests
- Host pytest `aot/tests/test_sdpa_aot_instance.py` + `test_kpack_pack.py`: swap HANDLER/SCHEMA_DIR/
  EXPECTED_BASENAME/EXPECTED_COMPILE_SPEC/EXPECTED_INSTANCES/cache_key/ABI/symbol constants + the
  `is_valid_spec`/`build_fmha_fwd_mfma` monkeypatches to the unified handler + `build_unified_attention_2d_tiled`.
- Python numeric verifier `aot/tests/sdpa_aot_numeric.py`: pack the paged-KV args (build block_tables/
  seq_lens/cu_q on host), read the unified grid_formula, keep the fp32 softmax reference.
- C++ e2e `TestRockeClientApplicability.cpp`: fixtures track the shipped unified instance shapes;
  the flow (is_supported_ext → build → execute → fp32 parity) is unchanged if selection keys stay dense.
- Add gfx1250 fp8 coverage (numeric only where a gfx1250 device is present).

### Phase 5 — Cleanup (gated on Phase 1–4 passing a smoke test)
- Remove all lingering `fmha_fwd_mfma` strings in comments/docs under `aot/`;
  reconcile `dsl_docs` file_index if it references the AOT family.

## Verification
- Host: `python tests/run_all.py` gate is not affected (kernels unchanged); run the AOT pytest suite
  (`rocke_client_aot_pytest`) after the handler/schema swap.
- Build: `cmake --build` regenerates HSACO + kpack per arch (gfx942/gfx950/gfx1250) with no GPU.
- GPU (where available): `rocke_client_sdpa_aot_numeric_<arch>` and the C++ e2e integration test —
  proves an actual kernel launch through hipDNN via the AOT kpack.

## Open decisions (need user input)
1. **gfx1151**: drop it from AOT (no tiled-2D spec) — OR keep the FMHA family for gfx1151 only
   (dual-family producer) — OR add a gfx1151 WMMA tiled-2D spec (large, out of scope this pass)?
2. **gfx1250**: include now (fp8-only, one fixed shape) or defer? It needs its own fp8 instance list.
3. **Selection keys**: keep dense keys (recommended, minimal C++ change) vs extend CompileSpec with
   unified block_size/num_seqs keys?
4. **Instance shape set**: which (dtype, head_size, block_size, GQA ratio, sliding_window) tuples to
   ship per arch for this first pass?
