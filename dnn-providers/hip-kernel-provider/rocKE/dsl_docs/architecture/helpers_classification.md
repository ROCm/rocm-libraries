# rocKE `helpers/` — Classification

The `helpers/` layer sits between `core` (the IR builder + IR types) and
`instances/` (concrete kernels). Helpers are reusable kernel-authoring
primitives.

They split into two fundamentally different populations:

- **Dual-engine emitters** (28) — have a byte-identical C99 port in
  `Cpp/helpers/`; they take the IR builder and emit ops. These are on the
  **byte-identity contract**.
- **Python-only** (16) — host-side tooling or a separate fusion subsystem that
  never emits kernel SSA, so they legitimately (or not-yet) have no C++ port.

---

## The core interface (what "emitter" means)

- **Python:** a helper receives `b: IRBuilder` (`core/ir.py:254`) and calls
  `b.const_i32 / add / fmul / global_load / global_store / scf_for / scf_if /
  mma / vec_extract / param`. Emitting an op returns a `Value` handle;
  side-effect ops (store/sync/ret) return nothing. Regions are scoped with
  `with b.scf_for(...)`.
- **C++:** mirror via opaque `rocke_ir_builder_t*` with method-for-method
  `rocke_b_*` calls (`ir.h`, ~209 of them). Python's `with` context managers
  become explicit `rocke_b_region_enter` / `rocke_b_region_leave`. Errors are
  **sticky-failing** (first error → all later calls no-op) instead of `raise`.
  Each `Cpp/helpers/<name>.cpp` reproduces its Python sibling's `rocke_b_*` call
  sequence in the same order — that is what the byte-identity gate enforces.

---

## Group 1 — Dual-engine emitters (on the byte-identity path)

### (a) Matrix-core / compute intrinsics

| Helper | Purpose |
|---|---|
| `atoms` | `MfmaAtom` dataclass + `.emit()` dispatch to MFMA/WMMA intrinsics (the compute-unit primitive) |
| `mfma_gemm_inner` | Universal MFMA-tiled K-loop body for all GEMM-shaped kernels |
| `mfma_attention` | MFMA-tiled FMHA forward inner body (QK · softmax · PV) |

### (b) Memory / data-movement

| Helper | Purpose |
|---|---|
| `loads` | Coalesced (sync) + async (DMA) global→LDS tile loaders |
| `tensor_view` / `transforms` | CK-tile coordinate-transform DAG (pad/merge/unmerge/embed) → address arithmetic |
| `distribution` | CK-tile `tile_distribution_encoding` load/store/shuffle/wmma |
| `io` | Dtype-dispatched scalar/vector load/store |
| `layouts` | LDS padding + XOR-swizzle bank-conflict avoidance |
| `gather_scatter` | MoE indirect addressing |
| `preshuffle` | Preshuffled-B per-lane offset |
| `grid` | Chiplet/XCD workgroup-ID remapping |

### (c) Math / numeric primitives

| Helper | Purpose |
|---|---|
| `activations` | sigmoid/tanh via `exp2` |
| `quant` | f32 → i8/fp8e4m3/bf8e5m2 |
| `i4_dequant` | INT4 packed unpack + dequant |
| `mx_scale` | MX (E8M0) microscaling decode/apply |
| `rotary` | RoPE rotary position embedding |
| `qk_scale` | Sage attention per-block Q/K scale |

### (d) Scheduling / pipelining

| Helper | Purpose |
|---|---|
| `schedule` | `sched_group_barrier` masks + `s_setprio` bookends |
| `persistent` | Persistent-kernel atomic-counter tile loop |
| `streamk` | StreamK tile partition (Atomic / Reduction) |
| `epilogues` | Accumulator → GMEM stores + LDS cshuffle |
| `sweep` / `scan` / `reduction` | LDS iteration / prefix-scan / tree-reduce |

### (e) Spec / config (value-only ports, no IR emission)

| Helper | Purpose |
|---|---|
| `spec` | Validation, signature building, kernel-name (no `IRBuilder`; field-for-field value fidelity) |
| `geometry` | `WarpGrid` block/warp/lane decomposition — **ported into the C++ `epilogues` header as `rocke_warp_grid_t`**, not a gap |
| `attention` | Config dataclasses + heuristic selectors (mostly returns plain config) |

---

## Group 2 — Python-only

### Host-side tooling (correctly never ported — consume IR, don't emit)

| Helper | Purpose |
|---|---|
| `compile` | `KernelDef → LLVM IR → HSACO` driver (comgr) |
| `autotune` | Runtime tuning search + JSON cache + HIP-event timing |
| `manifest` | `ck.dsl.example.manifest/v1` artifact JSON I/O |
| `split_k` | Dispatch-time heuristic (verified: 0 `core.ir` refs) |

### Fusion subsystem (a separate graph compiler — all 6 verified with 0 `core.ir` refs)

A layered planner that operates on a higher-level `FusionGraph` IR, **not** kernel
SSA:

```
fusion_ir (FusionGraph DAG)
  → fusion_legalize        (dtype/shape/op legality)
  → fusion_scheduler       (group into gemm_epilogue / elementwise regions)
  → fusion_lowering        (registry → calls existing instance builders)
  → fusion_memory          (workspace liveness + slot coloring)
  → fusion_validation      (sweep vs torch eager)
```

Relationship to `fuse` (which **is** ported): `fuse` is the lower
pattern-match → epilogue-op layer that plugs into a GEMM body (hence on the emit
path); the `fusion_*` family is the planner layer *above* it that orchestrates
existing builders.

### Not-yet-ported emitters (DO import `core.ir` and emit SSA — real porting candidates)

| Helper | Purpose | Why likely unported |
|---|---|---|
| `codebook` | Codebook dequant i4/i8 → fp8/bf8 | Narrow use (Sage int variants) |
| `rng` | Philox4x32-10 PRNG for FMHA dropout | Limited to attention dropout |
| `sparse_iter` | Block-sparse / VSA K-iterators | Newer feature |
| `mfma_attention_bwd` | MFMA attention backward bodies | Self-described "minimum-viable" |
| `pipeline` | Software-pipeline ping-pong scaffolding | — |

These are the **only** helpers that emit IR but lack a C++ port — so any kernel
relying on them cannot use the `cpp` backend and silently falls back to Python.

---

## C++-side notes

- **4 CPP-only files are split-outs, not C++-exclusive features:**
  - `attention_ext.cpp` — overflow from `attention.py` (softcap, MFMA dispatch, wave64 reduce)
  - `spec_ext.cpp` — now an empty dedup placeholder
  - `fused_moe_e2e_orchestrator.cpp` + `fused_moe_e2e_spec.cpp` — split of
    `instances/common/fused_moe_e2e.py`, filed under `helpers/` by naming; launch
    stubs return `ROCKE_ERR_NOTIMPL`.

- **Header taxonomy** (`Cpp/include/rocke/`):
  - `helper_rocke.helpers.<name>.h` — primary port per Python helper module
  - `helper_rocke.instances.<name>.h` — port of an `instances/` module
  - `helper_helper_rocke.*` / `helper_helpers.*` / `helper_instance_*` —
    overflow / companion headers for Python modules too large for a single C
    translation unit (e.g. the second batch of `attention.py`)

---

## Coverage summary

| Population | Count | C++ port | On byte-identity gate |
|---|---|---|---|
| Dual-engine emitters | 28 | yes | yes |
| Host-side tooling | 4 | no (by design) | no |
| Fusion subsystem | 6 | no (by design) | no |
| Not-yet-ported emitters | 5 | **no (gap)** | no |

The actionable gap is the 5 not-yet-ported emitters; everything else is either
covered or intentionally Python-only.
