# gfx1250 Universal Attention Port Plan

This page captures the implementation plan for porting ROCKE universal
attention to **gfx1250**, a **CDNA multi-chip (gfx1250-class) device** that
adopts the GFX12 programming model. It is a planning document, not a statement of
shipped support. As of writing there is **no `gfx1250` reference anywhere in the
Python tree**.

**gfx1250 is not part of the `gfx1201` / RDNA4 family.** It is a distinct CDNA
architecture. The only thing gfx1250 shares
with `gfx1201` is the surface fact "wave32 + WMMA". Everything that matters for
codegen — the WMMA matrix ISA shapes and fragment ABI, split wait counters, the
unified 320 KB WGP cache/LDS pool, 57-bit virtual-address buffer SRD packing,
split/named/cluster barriers, and required shader-end padding — is gfx1250-specific
and must **not** be aliased to the RDNA4 `gfx12_rdna` backend. Add a distinct
gfx1250 target instead of aliasing it to another arch, and prefer capability
queries (`target.wave_size`, `target.matrix.supports_shape(...)`,
`target.memory.has_tdm`, `target.wait_model`) over `if gfx == ...` checks.

## Goal & Phasing

**Scope: 2D path (prefill) only for now.** The 3D split-KV decode path is
deferred; its reference shapes are recorded below for when it is picked up.

| Phase | Goal | Success gate |
|---|---|---|
| **1** | Functional correctness | Dense WMMA FMHA fwd + scalar unified **2D** verify numerically on gfx1250 |
| **2** | Performance parity with gfx950 | **2D (prefill)** tiled WMMA attention matches gfx950 latency on prefill `seq_len = 64, 128` and the `aiter_ua_2_shapes.json` trace cohort |
| **3** | Beat gfx950 + reach roofline | gfx1250-optimized **2D** tiles (wide WMMA, async/TDM, 320 KB LDS) ≥ gfx950 and ≥ target % of roofline |

"Universal attention" spans two stacks today:

- **AITER Unified Attention** — `instances/common/attention_unified.py`
  (`build_unified_attention_2d/3d/reduce`), with MFMA-tiled forks in
  `instances/gfx950/` and `instances/gfx942/`. 2D ≈ paged prefill, 3D ≈ split-KV
  decode + reduce. **This port targets the 2D (prefill) path first.**
- **FMHA family** — `instances/common/fmha_mfma.py` (`build_fmha_fwd_mfma`,
  already wave-size aware) plus variants.

**Prefill shapes (now):** `seq_len = 64, 128` plus the `aiter_ua_2_shapes.json`
d64 trace cohort (all 2D-path). **3D decode shapes (deferred):** documented in
`examples/gfx950/qwen3_30b_a3b/README.md` — Qwen3-30B-A3B GQA-8 (nhead_q=32,
nhead_k=4), `head_dim=64`, `block_size=16`, bf16, `kv_len ∈ {512, 1024, 2048,
4096}`.

## Context: gfx1250 programming model

The gfx1250-class features below are **not implemented in ROCKE
core today**. This suggested capability table is the design target for the
gfx1250 `ArchTarget`:

```text
wave_size = 32                 matrix_path = wmma         has_mfma = false
has_wmma = true                has_swmma = true           has_tdm = true
has_async_global_lds = true    has_ds_load_tr = true      max_lds_bytes = 320*1024
wgp_cache_lds_shared = true    virtual_address_bits = 57  waitcnt_model = split_gfx1250
barrier_model = split_named_cluster                       requires_shader_end_padding = true
```

WMMA shapes: **16x16x32 F16/BF16** (primary), 16x16x64/128 FP8/BF8,
16x16x128 block-scaled F8/F6/F4, sparse SWMMAC, and IU8 — distinct from RDNA4
`gfx1201`'s 16x16x16. Drive instruction selection by capability query so
gfx1250-only instructions do not leak into portable code.

Key early decisions:

- **gfx1250 identity & WMMA ABI** — confirm via a WMMA
  probe the matrix shapes/fragment lane maps actually
  available; do **not** assume RDNA4 16x16x16 layouts. Catalog only the
  WMMA atoms validated on the bring-up board.
- **Low-precision form caveats** (current bring-up silicon):
  - Some low-precision (FP8/FP4) WMMA forms may run at reduced rate; set Phase 3
    roofline targets against rates measured on the bring-up board, not vendor
    peak numbers.
  - Block-scaled `WMMA..._F4` and block16 convert ops are **not catalogued**; do
    not emit them.
  - Keep IU4 out of gfx1250-portable code paths.
  - Gate any documented hardware-validation workarounds in backend validation
    rather than assuming a fully clean ISA on the bring-up board.
  - For the **2D fp16/bf16 prefill path in current scope**, the FP8/FP4
    limitations mostly affect the deferred FP8-KV / Phase 3 work, not Phase 1–2
    bf16/fp16 correctness and parity.
- **Compile path** — Phase 1 uses the **HIP path**
  (`compile_kernel_via_hipcc(kernel, arch="gfx1250")`) so the on-device clang
  handles gfx1250-specific lowering (split waits, 57-bit SRD, shader padding) that
  ROCKE core does not yet emit. Hot kernels migrate to the **LLVM-direct path**
  (`compile_kernel(..., arch="gfx1250")`) in Phase 2/3 as core gains those
  features.

## Target folders

```text
rocke/instances/gfx1250/
├── __init__.py
├── wmma_attention_fwd.py          # dense FMHA fwd (Phase 1)
└── attention_tiled_2d.py          # WMMA paged prefill (Phase 2)
# attention_tiled_3d.py (split-KV decode + reduce) — deferred, out of current scope
rocke/examples/gfx1250/
└── attention/
    ├── wmma_attention_fwd_verify.py   # numpy/torch ref (Phase 1)
    ├── parity_unified_attention.py    # 2d vs Triton+ref (Phase 2)
    ├── benchmark_prefill2d_live.py    # --shapes aiter_ua_2_shapes.json (Phase 2/3)
    ├── benchmark_prefill2d_traces.py  # trace cohort bench (Phase 2/3)
    └── _profile_one.py                # single-shape profiler (Phase 2/3)
# Shape source reused from gfx950: examples/gfx950/attention/aiter_ua_2_shapes.json
# (d64 prefill2d trace cohort). Qwen3 decode shapes inform the 3D path.
```

## Shared interface contracts (freeze first)

These let the four legs run in parallel without colliding:

1. **Arch token** `"gfx1250"`, resolvable via `ArchTarget.from_gfx("gfx1250")`.
2. **Builder signatures** mirror existing: `build_*(spec, arch="gfx1250") -> KernelDef`;
   specs are frozen dataclasses exposing `kernel_name()`.
3. **Compile entry**: Phase 1 → `compile_kernel_via_hipcc`; Phase 2+ → `compile_kernel`.
4. **Verify contract**: Phase 1 functional kernels compare to a numpy/torch fp32
   reference (tolerance gates fp16 `2e-2`, bf16 `4e-2`, matching gfx942 / gfx1151
   examples). Phase 2 perf parity reuses the gfx950 harnesses' contract: compare
   to AITER `ref_paged_attn` and require bit-accuracy vs Triton (`max_abs` within
   fp16/bf16 ULP) on the `aiter_ua_2_shapes.json` cohort.
5. **Run contract**: use the `remote-gpu-build-run` skill — preflight an idle
   GPU, build with host `hipcc`, run with `HIP_VISIBLE_DEVICES`, wrap in `timeout`.

---

## Parallel Legs

The work is split into four legs (core, helpers, instances, examples) so it can
be implemented by parallel agents. Leg A (core) is the critical-path
prerequisite.

### Leg A — Core (arch metadata + backend + lowering)

Owns: `core/arch/data/arch_specs.json`, `core/arch/target.py`,
`core/isa/backend.py`, `core/lower_llvm.py`, `core/lower_hip.py`, `core/ir.py`,
`test/test_rocke_multiarch.py`.

**Phase 1 (functional)**

- **A1.** Probe the real gfx1250 WMMA ABI: write `examples/gfx1250/wmma_probe.py`
  (structurally modeled on `examples/gfx1201/wmma_probe.py`, but **not assuming
  its 16x16x16 RDNA4 layout**), build/run on the remote host to discover the
  available gfx1250 WMMA shapes (expect 16x16x32 F16/BF16 first) and their A/B/acc
  fragment lane maps.
- **A2.** Add a **distinct** `gfx1250` row to `arch_specs.json`: `family: "cdna"`,
  a new `target_family` (e.g. `"gfx12_cdna"` — naming TBD), `wave_size:
  32`, gfx1250 capability flags, and a WMMA `mma` catalog built from A1's probe
  results (new `op_id`s, e.g. `wmma_gfx1250_f32_16x16x32_*`). Catalog only the
  **validated** atoms (no `WMMA..._F4` / block16; keep IU4 out of portable code).
  This may require extending the `arch_specs.json` schema with the gfx1250
  capability fields (`has_tdm`, `has_async_global_lds`, `virtual_address_bits`,
  `waitcnt_model`, `barrier_model`, `requires_shader_end_padding`,
  `wgp_cache_lds_shared`). Set `lds_capacity_bytes` to a validated value (start
  conservative; raise toward 320 KB in Phase 3).
- **A3.** Register a **new** backend: `Gfx1250Backend` (gfx1250 CDNA), **not**
  `Gfx12RdnaBackend`. It may borrow WMMA call emission as a starting point but
  must own gfx1250-specific buffer SRD (57-bit), wait model, and (later) shader
  padding. Add new `_FragInfo` lane maps in `target.py` for the probed gfx1250 WMMA
  op_ids, and the matching IR builder methods + `_MMA_C_FRAG_LEN` rows in `ir.py`.
- **A4.** Close HIP-path gaps used by attention: arch-aware buffer SRD word3 in
  `lower_hip.py` `_op_tile_buffer_rsrc` (currently hardcoded CDNA `0x00027000`),
  and add HIP-path WMMA op handlers + `__builtin_amdgcn_wmma_*` builtins for the
  gfx1250 shapes the attention body uses.
- **A5.** Fix stale `known_arches()` expectation in `test_rocke_multiarch.py`
  (still asserts `{gfx942, gfx950, gfx1151}`); add gfx1250 metadata assertions
  (CDNA family, wave32, gfx1250 capability flags).

Gate: `ArchTarget.from_gfx("gfx1250")` works; probe matches reference; a trivial
WMMA GEMM compiles + runs on the host.

**Phase 2 (perf parity)**

- **A6.** LLVM-direct path: wire `Gfx1250Backend.emit_wmma` to the correct gfx1250
  WMMA intrinsics (`llvm.amdgcn.wmma.*` for the probed shapes) and assemble via
  comgr; set the gfx1250 datalayout/triple.
- **A7.** Split wait counters: implement the gfx1250 `waitcnt_model = split_gfx1250`
  (`S_WAIT_LOADCNT/STORECNT/DSCNT/KMCNT/ASYNCCNT/TENSORCNT`) as the LLVM-direct
  emission path, replacing the monolithic `s_waitcnt` for gfx1250. Flat-memory
  waits must cover both the load/store and DScnt sides.
- **A7b.** gfx1250 buffer SRD: implement 57-bit virtual-address V# packing with
  byte-based `num_records` (and the documented OOB workaround) instead of the
  RDNA/CDNA word3 constants.

**Phase 3 (beat + roofline)**

- **A8.** Expand the gfx1250 WMMA catalog for performance: validated 320 KB
  `lds_capacity_bytes`, FP8/BF8 and block-scale WMMA atoms (16x16x64/128) with
  `_FragInfo` lane maps, and SWMMAC sparse forms if used. On the current bring-up
  silicon some of these low-precision forms may be reduced-rate and
  `WMMA..._F4`/block16 are not catalogued — gate them by capability query so a
  future silicon row can enable the faster/extra forms.
- **A9.** Async memory + TDM: add async global→LDS load/store and Tensor Data
  Mover (`TENSOR_LOAD_TO_LDS` / `TENSOR_STORE_FROM_LDS`) ops with their own
  `TENSORcnt` / `ASYNCcnt`, gated behind the capability flags.
- **A10.** Shader-end padding (`S_CODE_END` / 64-DWORD tail) — required by the
  gfx1250 code-object loader (a packaging requirement, not an optimization), plus
  split/named/cluster barrier lowering (`S_BARRIER_SIGNAL` / `S_BARRIER_WAIT`).

### Leg B — Helpers (make the shared body wave32-clean)

Owns: `helpers/atoms.py`, `helpers/mfma_attention.py`,
`instances/common/_fmha_warp_body.py`, `helpers/scan.py`, `helpers/reduction.py`,
`helpers/pipeline.py`.

**Phase 1**

- **B1.** gfx1250 `WmmaAtom` factory in `atoms.py` (today only gfx11 f16/bf16);
  delegate layout to `ArchTarget.mma.by_op_id(...)` for the probed gfx1250 op_ids
  (16x16x32 first). Drive atom selection by capability query, not `gfx ==` checks.
- **B2.** Parameterize wave size: remove `WARP_SIZE = 64` hardcode in
  `_fmha_warp_body.py` and the `kb.block_size(64)` literals (`fmha_varlen.py`,
  `fmha_head_grouping.py`, `fmha_fwd_fp8.py`); derive from
  `ArchTarget.from_gfx(arch).wave_size`.
- **B3.** wave32 reductions/scan: make `scan.py` / `reduction.py` defaults
  arch-derived so softmax row-reductions are correct on wave32 (32-lane masks,
  `v_mbcnt_lo` only).
- **B4.** Arch-aware attention body: ensure
  `mfma_attention.py::mfma_attention_fwd_inner_body` selects WMMA atoms + fragment
  maps for gfx1250 (it already branches on wave_size for gfx1151).

Gate: softmax / QK·PV inner body produces correct numerics on a 1-block wave32 case.

**Phase 2**

- **B5.** WMMA pipelining helpers for wave32 (double-buffer LDS staging, K-loop
  unroll) tuned for gfx1250.
- **B6.** ds_load_tr / transpose-load helpers (gfx1250 exposes wave32 LDS transpose
  loads such as `DS_LOAD_TR16_B128`; `has_ds_load_tr = true` in the capability
  table) for efficient WMMA operand staging.

**Phase 3**

- **B7.** Async-copy + TDM staging helpers and WMMA co-execution hazard spacing
  (V_NOP scheduling), to hit roofline overlap.

### Leg C — Instances (the attention kernels)

Owns: `instances/gfx1250/*`, dispatch seams in
`instances/common/attention_unified.py` and `attention_arch.py`,
`instances/__init__.py`, `instances/SUPPORT_MATRIX.md`.

**Phase 1 (functional)**

- **C1.** Dense WMMA FMHA fwd: use `instances/gfx1151/wmma_fmha_fwd.py` as a
  *structural* wave32-WMMA scaffold for `instances/gfx1250/wmma_attention_fwd.py`,
  but retarget the atom to the probed gfx1250 WMMA op_id (16x16x32) and gfx1250
  fragment maps — not the gfx1151/gfx1201 16x16x16 layout. Fastest route to first
  correct attention output.
- **C2.** Scalar Unified Attention (2D) on gfx1250: confirm
  `build_unified_attention_2d` (the non-tiled scalar kernel) compiles + verifies
  with `arch="gfx1250"` — already arch-polymorphic, gives 2D prefill functional
  coverage without a tiled rewrite. (3D `build_unified_attention_3d/reduce` is
  deferred with the rest of the 3D path.)

Gate: dense FMHA fwd + scalar unified 2D pass numeric verify on gfx1250.

**Phase 2 (perf parity vs gfx950)**

- **C3.** WMMA tiled 2D (prefill): new `instances/gfx1250/attention_tiled_2d.py`.
  Largest lift — the gfx950 kernel is MFMA wave64 + `ds_read_tr`; gfx1250 needs a
  WMMA wave32 rewrite of QK→softmax→PV with LDS staging. Use
  `UnifiedAttention2DTiledSpec` as the shape contract, and target the feature set
  the `aiter_ua_2_shapes.json` cohort exercises: head_size 64, block_size 32,
  GQA-8, attention sinks, sliding-window, bf16 and fp8-KV (sync-dequant), and
  64-bit paged-KV addressing for production-scale caches (>2 GiB). Also cover the
  small prefill `seq_len = 64, 128` shapes used as the initial parity target.
- **C5.** Wire the 2D dispatch seam: extend `_tiled_2d_impl` in
  `attention_unified.py` to route gfx1250, add gfx1250 to
  `validate_tiled_attention_arch` / gate sets in `attention_arch.py`. Re-export
  from `instances/__init__.py`; update `SUPPORT_MATRIX.md`.
- **C4 (deferred).** WMMA tiled 3D (decode): `instances/gfx1250/attention_tiled_3d.py`
  (split-KV segment + reduce) and the `_tiled_3d_impl` dispatch seam — out of
  current scope; pick up after the 2D path reaches parity, using the Qwen3
  decode shapes from `examples/gfx950/qwen3_30b_a3b/README.md`.

**Phase 3 (beat gfx950 + roofline)**

- **C6.** gfx1250-optimized 2D tiles: larger LDS tiles (320 KB), wider WMMA-K
  (16x16x32) if available, async/TDM K/V staging, register-P fast path (mirror
  `attention_tiled_2d_fastkv_regp.py`), FP8 KV prefill.

### Leg D — Examples & Verification

Owns: `examples/gfx1250/**`, plus reuse of
`examples/common/fmha_fwd_verify_hip.py` and `parity_extended_kernels.py`.

**Phase 1**

- **D1.** Verify harness: clone
  `examples/gfx1151/attention/wmma_fmha_fwd_verify.py` →
  `examples/gfx1250/attention/wmma_attention_fwd_verify.py` (numpy ref, tol `2e-2`).
- **D2.** Scalar UA (2D) verify: drive `build_unified_attention_2d` with
  `--arch gfx1250` at prefill `seq_len = 64, 128` against a torch fp32 reference
  (clone the gfx942 `parity_unified_attention.py` torch path — no Triton
  dependency).

Gate: both exit 0 with `bad=0` / within tol on the host.

**Phase 2** — drive perf parity from the existing gfx950 attention harnesses and
the captured trace cohort; do not invent new shape sets or timers.

- **D3.** Shape source (2D prefill): start with `seq_len = 64, 128`, then the
  `examples/gfx950/attention/aiter_ua_2_shapes.json` cohort (JSONL dump of real
  AITER `unified_attention` calls — the **d64 prefill2d trace cohort**:
  `head_size 64`, `block_size 32`, GQA-8 (64 query / 8 KV heads), attention
  sinks, sliding-window `(127,0)` or full, bf16 or bf16-Q + fp8-KV, chunked
  prefill across 1..512 sequences; all route to the **2D path**). The d128
  reference scenarios from `parity_unified_attention.py::default_scenarios()` are
  an optional extra prefill set.
- **D4.** Model the gfx1250 examples directly on the gfx950 harnesses (reference
  implementations, mirror their structure under `examples/gfx1250/attention/`),
  using the **2D lane** only:
  - `parity_unified_attention.py` — the `2d` apples-to-apples lane vs AITER
    Triton + `ref_paged_attn`, HIP-event timer on torch's stream,
    `--attempts/--warmup`, JSON report.
  - `benchmark_prefill2d_live.py` — consumes `--shapes aiter_ua_2_shapes.json`
    (forced-2D Triton vs CK variants, per-shape correctness vs Triton,
    production-scale `cap_blocks` 65536/131072 to stay HBM-bound), plus
    `benchmark_prefill2d_traces.py` and `_profile_one.py`.
  - Emit CSVs mirroring `prefill2d_bf16_triton_ckdsl_perf.csv` /
    `prefill2d_fp8_triton_ckdsl_perf.csv`.
  Phase 2 success gate: gfx1250 within parity of gfx950 (and bit-accurate vs
  Triton, `max_abs` within fp16/bf16 ULP) on prefill `seq_len = 64, 128` and the
  `aiter_ua_2_shapes.json` 2D cohort.

**Phase 3**

- **D5.** Roofline tooling: add achieved-vs-peak (HBM BW + WMMA FLOPs) reporting
  and a sweep driver, run at production-scale `cap_blocks` (HBM-bound regime).
  Phase 3 gate = gfx1250 ≥ gfx950 and ≥ target % of roofline on the 2D prefill
  `aiter_ua_2_shapes.json` cohort.
- **D6.** NaN / correctness root-cause workbench: keep a small, repeatable GPU
  loop for the causal NaN and test one hypothesis at a time. Include **layout
  mismatch** as a first-class hypothesis alongside WMMA hazard spacing:
  - Verify QK and PV fragment lane maps for the gfx1250 `16x16x32` atom using
    one-hot / asymmetric inputs, not just random GEMM.
  - Validate the accumulator-layout -> P LDS -> A-operand-layout transform for
    both 16-column sub-tiles of `BLOCK_K=32`.
  - Validate the PV B-operand layout (`V[k, d_col]`) and the row/column mapping
    used by the epilogue for `D=64/128`.
  - Check GQA head mapping and Q/K/V/O strides independently from the matrix
    layout.
  - Compare debug dumps for masked QK scores, softmax denominator `L`, P values
    read back from LDS, and PV accumulators. A layout mismatch should be visible
    as a deterministic transpose/permute error; a scheduling hazard should be
    shape- and scheduling-sensitive.

---

## Sequencing & dependency graph

```text
A1 probe ─► A2 arch row ─► A3 backend ─┬─► (unblocks B, C, D on-device work)
                                       │
B1/B2/B3 wave32 helpers ───────────────┤
                                       ▼
C1 dense FMHA + C2 scalar UA-2D ──► D1/D2 verify       = PHASE 1 DONE
                                       │
A6/A7 LLVM path ─► C3 tiled-2D WMMA ─► C5 2D dispatch ─► D3/D4 2D-cohort parity = PHASE 2 DONE
                                       │
A8/A9/A10 gfx1250 features ─► B5/B7 ─► C6 optimized tiles ─► D5 roofline   = PHASE 3 DONE
```

- After A1–A3 land, legs B, C1/C2, and D1/D2 run concurrently.
- Leg A is the long pole for Phases 2–3 (LLVM path, gfx1250 features); B and C2 can
  start against stubs using the HIP path.

## Risks

- **gfx1250 identity & WMMA ABI** — confirm the available WMMA shapes / fragment
  lane maps via A1 before committing the arch row. Do not assume the RDNA4
  `gfx1201` 16x16x16 layout. Catalog only validated atoms (no `WMMA..._F4`/block16,
  keep IU4 out of portable code) and gate by capability query so a future silicon
  row can diverge later.
- **Do not alias to `gfx12_rdna`** — gfx1250 is CDNA gfx1250; reusing
  `Gfx12RdnaBackend`, its buffer SRD, or its wait model would be incorrect. Keep
  the design capability-driven, not `if gfx == ...`.
- **gfx1250 core features are net-new** — split wait counters, 57-bit buffer SRD,
  320 KB unified LDS, async/TDM, split barriers, and shader padding are not in
  core today. The HIP-path-first strategy defers these for Phase 1 functional
  correctness, but Phases 2–3 require landing them in Leg A.
- **Tiled 2D WMMA rewrite (C3)** is the largest engineering item; the gfx950 MFMA
  tiled kernel cannot be mechanically retargeted (wave64→wave32, MFMA→WMMA,
  different fragment layout, gfx1250 LDS/wait model). The 3D decode path (C4) is
  deferred until the 2D path reaches parity.

---

# Phase 2 — Closing the gfx950 ↔ gfx1250 gap

Phase 1 delivered correct v1 WMMA kernels (tiled-2D prefill, split-KV 3D decode).
A cross-arch comparison of the **same rocke decode path** on the qwen3-30B-A3B
shapes (GQA-8, d64, block16; one harness, identical shapes/ABI/reference) shows
the v1 gfx1250 kernels trail the mature gfx950 kernels. This phase closes that gap
by **mirroring each gfx950 optimization and adapting it to gfx1250 wave32 / WMMA**.

## Measured baseline (decode 3D, segm+reduce, num_segments=16, µs/launch)

| regime | shape | gfx950 (MFMA w64) | gfx1250 (WMMA w32) | ratio |
|---|---|--:|--:|--:|
| tiny (ns=2) | kv1024 fp8 | 12.2 | 91.2 | 7.5× |
| tiny (ns=2) | kv4096 fp8 | 22.6 | 97.8 | 4.3× |
| batched (ns=256) | kv1024 fp8 | 66.5 | 99.9 | 1.5× |
| batched (ns=256) | kv4096 bf16 | 211.1 | 318.7 | 1.5× |

Tiny shapes are launch/issue-bound (overhead-dominated); the batched regime
(~1.5×) is the meaningful throughput signal and the gap to close. Prefill (2D)
baseline: `prefill_big` (1 seq, q=2048, full) 659 µs, the heavy target.

## P0 — correctness prerequisite: gfx1250 dependent-WMMA hazard

The batched decode at `(num_seqs≥256, kv_len=4096, num_segments=16, fp8)`
**intermittently produces a garbage (>1e20) value in the P·V accumulator**
(`segm_output`), seed/timing dependent (~1/3 of runs). Root cause: the gfx1250
**dependent-WMMA co-execution hazard** in the long P·V accumulator chain at high
occupancy (24k+ workgroups). Findings:

- Not a workspace-coverage gap — all `segm_{output,max,expsum}` slots are written
  (NaN-sentinel probe shows 0 unwritten). The huge value is *written* by the
  segment kernel's PV WMMA.
- The dense-kernel `hipcc` crutch is **not available**: hipcc/clang crashes
  (frontend exit 70) on the 3D segment HIP source.
- Fixed `v_nop` spacing was historically insufficient on the dense kernel.

**This is the gating item.** Options, in order of preference:
1. **Proper backend WMMA scheduler** (Leg A): emit the correct
   inter-WMMA `s_nop`/wait so dependent WMMAs don't co-execute. Benefits every
   gfx1250 WMMA kernel (attention + GEMM + MoE). Largest, correct fix.
2. Targeted `v_nop`/wait spacing after the PV (and accumulating QK) WMMA, swept
   empirically per shape — interim, may not be robust.
3. Restructure to shorten the dependent PV chain (e.g. two interleaved
   accumulators) so consecutive PV WMMAs are independent.

Until P0 lands, gate the affected high-occupancy fp8 decode config or fall back.

## P1–P3 — mirror gfx950 optimizations onto gfx1250 WMMA

| # | gfx950 optimization | gfx1250 (gfx1250) adaptation | status |
|---|---|---|---|
| O1 | half-local **register-P** (skip P→LDS→frag round-trip) | `compute_pv_from_probs` (register-resident probs) | in progress |
| O2 | native **fp8 MFMA** QK/PV (no dequant) | use the **K=64 fp8 WMMA** atom (`wmma_gfx1250_f32_16x16x64_fp8_fp8`) for QK/PV directly on fp8 — skips the per-tile fp8→bf16 dequant and doubles K throughput | new atom wired in `ir.py`/backend |
| O3 | async DMA **double-buffered** K/V staging (global→LDS, overlap with MFMA) | gfx1250 async global→LDS / TDM (`has_async_global_lds`, `has_tdm`) + double-buffer the K/V LDS tiles; split-wait drains already in place | needs Leg A async + ping-pong |
| O4 | `ds_read_tr` transposed LDS reads (conflict-free frag loads) | enable gfx1250 `ds_load_tr`; replace the scalar `vec_insert` fragment builds in `compute_pv` with transposed LDS reads | needs Leg A `ds_read_tr` |
| O5 | **wave/occupancy tuning** (`waves_per_eu`, multi-warp BLOCK_M=32) | multi-wave workgroups (BLOCK_M=32, 2–4 waves) to amortize KV reads + hide latency; re-sweep `waves_per_eu` (no-op at 1 wave) | after O1/O3 |
| O6 | **320 KB unified LDS** residency | stage larger KV windows / more double-buffer depth in gfx1250's 320 KB LDS to cut O(n²) global KV re-reads (prefill) | Leg A LDS budget |
| O7 | larger `BLOCK_Q` (amortize KV re-reads across more query rows) | raise BLOCK_Q (BLOCK_M=32 ⇒ multi-WMMA-tile accumulators) — biggest prefill win | after O5 |
| O8 | fast paged-KV descriptor / fewer block-table loads | hoist per-tile block lookup; for block_size 32 reuse the single cached block (already done) | partial |

**Ordering:** O0 (WMMA hazard) → O1 (register-P) + O2 (fp8 WMMA, skip dequant) →
O3/O4 (async + transposed LDS, needs Leg A gfx1250 features) → O5/O6/O7 (occupancy,
LDS residency, bigger tiles) → re-bench vs gfx950 each step. Validate correctness
(numpy ref) and latency on GPU-3 after every optimization; keep the cross-arch
harness as the regression gate.

> Note: the shared gfx1250 attention kernels (`_wmma_attention_common.py`,
> `attention_tiled_2d.py`) are under active P2 refactoring (O1 register-P);
> coordinate edits to avoid clobbering.
