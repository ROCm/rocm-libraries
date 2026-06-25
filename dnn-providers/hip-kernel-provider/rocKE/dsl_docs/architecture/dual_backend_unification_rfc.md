# RFC: Dual-Backend Unification, Differential Parity, and Full-Stack Hardening of `rocke`

| | |
|---|---|
| **Status** | Proposed (enactable plan) |
| **Scope** | `Python/rocke` (Python), `python/Cpp` (C → C++20), `dnn-providers/rocke-provider` (C++ hipDNN plugin) |
| **Type** | Architecture + Test Infrastructure + Migration |
| **Supersedes / relates to** | `architecture/multi_arch_data_layout.md`, `hipdnn_provider/plan.md`, the existing `tests/parity` harness in `Cpp` |
| **Owner** | rocke team |
| **Target horizon** | Multi-phase; each phase independently shippable. Differential soak ≈ 2–4 weeks before any default flip. |

---

## 0. Executive summary (TL;DR)

We maintain the same kernel-generation engine **twice** — once in Python (`rocke`, the research/authoring surface) and once in C99 (`Cpp`, the Python-free engine the hipDNN provider links). Equivalence is held together today by a sampled `sha256` byte-identity harness. That works, but the dual implementation is a growing tax: the repo's own history records repeated **SSA-numbering drift** and **GCC argument-evaluation-order** bugs that exist *only because there are two implementations*.

This RFC proposes a single, coherent program that:

1. **Reorganizes** `Cpp` from 220 flat dot-named files (`helper_rocke.helpers.atoms.c`) into a real subfolder tree mirroring the Python package.
2. **Builds a comprehensive, repeatable test spine** — an IR verifier, a round-trippable IR serialization, IR-level diff, `.ll` byte-identity, fuzzing under sanitizers, golden/regression gates, and **detailed on-GPU numeric tests across every instance family** — before any risky refactor.
3. **Migrates** `Cpp` from C99 to **C++20**, incrementally and parity-gated, using namespaces (which also structurally fix the duplicate-symbol problem) and exceptions (which map 1:1 to Python's `raise`).
4. **Introduces a Python "veneer" / dual-backend flag**: the `IRBuilder` and instance layer can switch between the **native Python** backend and the **C++** backend (via pybind11) — `python` | `cpp` | `both`. `both` runs *both* and asserts equality on everything.
5. **Closes all gaps and drift** between Python and C++ so every family Python supports reaches full C++ parity (IR + `.ll` + numeric).
6. **Completes the dispatcher** beyond today's fp16-RCR-GEMM-only scope to all families, fixing the known `rdna_wmma` arch-gating bug, kept in lockstep via the dispatch-parity harness.
7. **Hardens the hipDNN provider end-to-end** so nothing fails E2E across supported ops/dtypes/layouts.
8. **Runs both backends behind the flag, differentially checked, for a multi-week soak** until parity is statistically proven, *then* flips the runtime default to C++ — keeping both instance layers mirrored and maintainable, with the differential harness as the permanent guarantee.

The chosen architecture is explicitly **dual-backend-behind-a-flag with continuous differential testing**, not a one-shot replacement. We never delete the Python implementation; we prove the C++ one equal over real calendar time first.

---

## 1. Motivation

### 1.1 The dual-maintenance tax
- `rocke` (Python): ~core 10.5k + helpers 21k + instances 54k LOC. The authoring surface; fast to iterate; the source of truth for correctness.
- `Cpp` (C99): 220 `src/*.c` + 146 headers, ~157k LOC. A faithful, byte-identical port whose only reason to exist is **no Python at runtime** in the provider.
- `rocke-provider` (C++17): the hipDNN plugin that links `librocke_core.a` (C-JIT path) and/or ships pre-built HSACO/`.ll`.

Every new op family, atom, or knob must be written **2–3 times** and kept byte-identical. The byte-identity bar means even cosmetic Python refactors can break the C port.

### 1.2 Evidence of drift as a recurring defect class
From the project's own findings:
- Pure **SSA value-id numbering drift** between Python and C on the WMMA/RDNA path and `chiplet_swizzle` (structural divergence: runtime `mul/sdiv` vs constant-folded literal, plus workgroup `id.x/.y` read-order swap).
- Two **GCC right-to-left argument-evaluation** SSA-drift bugs in shared epilogue load paths (undefined-order reliance — exactly the class sanitizers + sequencing catch).
- A **dispatcher selection bug**: the `rdna_wmma` candidate is not arch-family-gated and reports `supported=True` on gfx950, out-ranking the intended CDNA candidate.

These are not one-offs; they are the *expected* failure mode of two hand-synced implementations. The remedy is (a) make divergence cheap to detect (the test spine) and (b) reduce the surface where divergence can occur (single lowerer of record + a switchable single engine).

### 1.3 Why now
We need to simultaneously (1) expand the dispatcher to more families, (2) close C-port gaps, and (3) guarantee nothing fails E2E on hipDNN. Doing any of those safely on top of a dual, drift-prone, flat-file, warning-suppressed C99 base is reckless. The foundation must come first.

---

## 2. Goals and non-goals

### 2.1 Goals
- G1. A **clean, foldered** `Cpp` source tree mirroring the Python package; no dot-names.
- G2. A **comprehensive differential test harness** covering: IR well-formedness, IR equivalence, `.ll` byte-identity, fuzzing, golden regression, and **numeric correctness on real GPUs for every instance family**.
- G3. **Every example is repeatable**: deterministic, single-command, golden-recorded, CI-smoked.
- G4. A **C++20** engine that compiles cleanly with `-Wall -Wextra -Werror` and passes ASan/UBSan.
- G5. A **Python veneer / dual-backend flag** (`python` | `cpp` | `both`) so the same authoring API can drive either engine, and `both` differentially checks every build.
- G6. **Zero gaps / zero drift**: every family Python supports has full C++ parity (IR + `.ll` + numeric).
- G7. A **complete dispatcher** across all families with arch-family correctness, kept in C++/Python lockstep.
- G8. **hipDNN E2E green**: every supported op/dtype/layout passes the provider's end-to-end numeric gates.
- G9. Both instance layers **mirrored and maintainable**, with differential testing as a permanent CI guarantee.

### 2.2 Non-goals (for this RFC)
- N1. Deleting the Python implementation. (Explicitly retained as authoring + differential oracle.)
- N2. Making the IR an *input* format for human authoring (MLIR-as-input remains a non-goal; serialization is a machine artifact).
- N3. Changing GPU kernel numerics/algorithms. Output `.ll`/HSACO must stay byte-stable except where a fix is explicitly blessed via golden update.
- N4. Rewriting the optimization runbook or adding new kernels for performance. (Perf work proceeds on the stabilized base afterward.)

---

## 3. Current state (baseline)

| Component | Language | Role | Notes |
|---|---|---|---|
| `rocke` | Python | authoring + reference lowering | `core/ir.py` (`IRBuilder`), `core/lower_llvm.py` (4099 LOC), `runtime/comgr.py`, `dispatch/`, `heuristics/`, `instances/` |
| `Cpp` | C99 | Python-free engine | 220 `src/*.c`, 146 headers; gfx950 byte-baseline; `tests/parity` (63 emit pairs) |
| `rocke-provider` | C++17 | hipDNN plugin | GEMM/Attention/Conv engines; `runtime/` header-only core; Fast / JIT / C-JIT paths; `dispatch_parity` harness |

**Known C-port gaps** (to be closed in WS5): fused-MoE e2e host runtime (`ROCKE_ERR_NOTIMPL`), RDNA WMMA op wiring (`lower_llvm_mma.c:282`), buffer-addrspace tensor views, smoothquant/pooling distribution helpers, conv `LdsLayout`/`WarpGrid` accessors, attention I64-KV/fp8 side paths, stale `img2col` "stubbed load" comment (verify real).

**Known build fragility**: `cc src/*.c` fails (duplicate `rocke_conv_problem_*` / `gemm_multi` helpers across TUs; undefined refs from norm/pooling/reduce TUs). Only the CMake static-archive path links. Warnings are suppressed (`-Wno-comment`, `-Wno-unused-function`, `-Wno-format-truncation`).

**Dispatcher scope**: Python `dispatch/` implements **fp16 RCR GEMM only** (4 candidates). C++ `Dispatcher` is its twin (identity = `cache_key`). Parity harness compares the structural tuple `(block_m, block_n, block_k, pipeline, epilogue)`.

---

## 4. Target architecture

### 4.1 The dual-backend, flag-switchable engine

```
                         ┌────────────────────────────────────────────┐
   Python authoring API  │  rocke  (specs, IRBuilder DSL, instances)  │
   (unchanged surface)   └───────────────┬───────────────┬────────────┘
                                          │               │
                              ROCKE_BACKEND selects      │
                                          │               │
                     ┌────────────────────▼──┐   ┌────────▼─────────────────┐
                     │  PYTHON backend         │   │  CPP backend (pybind11)  │
                     │  native build + lower   │   │  → Cpp (C++20) core  │
                     └────────────┬────────────┘   └───────────┬──────────────┘
                                  │   serialize IR / emit .ll   │
                                  └──────────────┬──────────────┘
                                                 │
                            ┌────────────────────▼────────────────────┐
                            │  DIFFERENTIAL SPINE  (mode = both)        │
                            │  verify · IR-diff · .ll sha · numeric     │
                            └────────────────────┬────────────────────┘
                                                 │  (C++ canonical at runtime)
                            ┌────────────────────▼────────────────────┐
                            │  rocke-provider  (hipDNN, C++)           │
                            │  Fast / JIT / C-JIT → comgr → HIP launch  │
                            └──────────────────────────────────────────┘
```

- **The veneer**: Python's public authoring API does not change. Underneath, the `IRBuilder` (and the instance builders) dispatch to either the native Python implementation or the C++ engine via pybind11 bindings.
- **The IR is the interchange artifact** (seam (a) from the design discussion): a round-trippable serialization with **explicit SSA ids**, so the C++ side prints what was assigned rather than re-deriving (kills the numbering-drift class).
- **`both` mode is the differential workhorse**: every build runs through both backends and asserts equality at IR, `.ll`, and (where a GPU is present) numeric levels.
- **Long-term**: both instance layers stay mirrored; the C++ backend becomes the runtime default after the soak; the differential harness stays on permanently as the maintenance guarantee.

### 4.2 The backend-selection contract

| Selector | Values | Meaning |
|---|---|---|
| Env `ROCKE_BACKEND` | `python` | native Python build + lower |
| | `cpp` (current default) | drive the C++ engine via pybind |
| | `both` | run both, assert equality, return the Python result (or C++ in runtime profile) |
| API `compile_kernel(..., backend=...)` | same three | per-call override |
| Provider env `ROCKE_C_JIT` | `1` | provider uses the C++ engine at runtime (existing flag, now the validated path) |

Equality semantics in `both`:
- **IR level**: canonicalized serialized IR must be identical (string-equal after canonicalization).
- **`.ll` level**: `sha256` identical, *or* identical rejection (both reject the spec with matching reason/exit).
- **Numeric level** (GPU lanes only): cross-backend output **bit-identical** (same HSACO ⇒ same bytes), and each backend vs a CPU/torch reference within the per-dtype tolerance table (§7.3).

---

## 5. Workstreams

Each workstream lists **objective**, **tasks**, **acceptance criteria (AC)**, and **parallelism**. Sequencing/dependencies are in §10.

### WS0 — Repository reorganization (foldering) [parallel-safe, do first]

**Objective.** Replace flat dot-names with a subfolder tree mirroring the Python package, *without* changing behavior or `.ll` output.

**Decision (verified against the live Python package, 2026-06-19): mirror `rocke/` EXACTLY, not a family grouping.** The Python package is `instances/common/` + per-arch dirs (`gfx942/gfx950/gfx1151/gfx1201/`) — it does *not* group by family — and the lowerers live under `core/` (`core/lower_llvm.py`, `core/ir.py`, …). A faithful 1:1 file correspondence is worth more than family aesthetics because the entire dual-backend program depends on Python↔C lockstep (and the future C++ namespaces mirror these paths). **Headers stay flat** in `include/rocke/` for WS0 (they are not dot-named; all 142 includes use the `rocke/` prefix and resolve via `-I include`, so moving `.c` files breaks nothing — header foldering, if desired, is deferred to WS3 with namespaces).

**Target layout** (achieved):

| Current (flat dot-name) | Target |
|---|---|
| `arena.c`, `strbuf.c` | `src/support/` (C-only primitives; no Python equiv) |
| `ir_*.c` | `src/core/ir/` (↔ `core/ir.py`, split) |
| `passes.c` | `src/core/passes.c` |
| `isa_backend.c` | `src/core/isa/backend.c` |
| `arch_target_*.c`, `helper_rocke.core.arch.c` | `src/core/arch/` |
| `lower_llvm_*.c` | `src/core/lower_llvm/` (↔ `core/lower_llvm.py`, split) |
| `lower_hip_*.c` | `src/core/lower_hip/` |
| `lower_cktile.c` | `src/core/lower_cktile.c` |
| `helper_rocke.helpers.X.c`, `helper_helpers.asm.c` | `src/helpers/X.c` (↔ `helpers/X.py`) |
| `instance_<name>.c` | `src/instances/common/<name>.c` (↔ `instances/common/`) |
| `helper_rocke.instances.common.<name>.c` | `src/instances/common/<name>.c` (collision-disambiguated → `<name>_helpers.c` when an `instance_<name>.c` exists) |
| `instance_<arch>_<name>.c`, `helper_rocke.instances.<arch>.<name>.c` | `src/instances/<arch>/<name>.c` |

**Tasks (DONE).**
- T0.1 ✅ Script-generated old→new path map (220 files): 0 unmapped, 0 collisions after disambiguating the 2 `instance_`/`helper_` same-base pairs (`gemm_multi_d`, `img2col`) with a `_helpers` suffix.
- T0.2 ✅ Moved every file with plain `mv` (git is handled out-of-band by the owner). No content edits except build-glob references.
- T0.3 ✅ CMake `file(GLOB …)` → `file(GLOB_RECURSE … CONFIGURE_DEPENDS src/*.c)`; updated `run_parity.sh` / `run_gemm_parity.sh` from `src/*.c` to `$(find "$CKC/src" -name '*.c')`.
- T0.4 ✅ No `#include` changes needed (all includes are `rocke/`-prefixed; 0 relative includes).
- T0.5 ✅ Single-emission discipline + archive-link unchanged (duplicate-symbol fix lands in WS3 via namespaces).

**AC — MET.** `librocke_core.a` builds via the recursive glob from `/tmp`; the entire `run_parity.sh` (4/4) and `run_gemm_parity.sh` (7/7 incl. identical rejections) suites produce **byte-identical `.ll`** (sha unchanged) before and after. (Archive size shrank ~7 KB purely from shorter `ar` member-name metadata; compiled code identical.)

**Parallelism.** Done by the main session (sequential/mechanical; not fan-out-friendly). Landed before WS3 so the C++ migration targets the final tree.

---

### WS1 — IR verifier + serialization + round-trip [parallel-safe]

**Objective.** Make the IR a first-class, checkable, round-trippable artifact on *both* sides.

**Tasks.**
- T1.1 **IR verifier** (an LLVM-`verify`-style pass), implemented in Python first (reference), then C++. Checks:
  - SSA dominance: every operand defined before use within its region.
  - Type consistency: operand/result types match each opcode's contract.
  - Arity/result counts per opcode; no dangling/foreign value refs.
  - Region well-formedness: `scf.for`/`scf.if` terminators present & correct; loop-carried values typed.
  - Attr maps: required keys present per opcode; enum values legal.
  - Vector widths, `smem` shapes, addrspace correctness; wave-size invariants.
- T1.2 **IR serialization** `ck.dsl.ir/v1`: canonical, parseable text capturing types, values **with explicit SSA ids**, ops (opcode, operands-by-id, results, sorted attrs), regions, params, kernel metadata. Versioned header.
- T1.3 **Parsers**: Python parser (text → `KernelDef`) and C++ parser (text → `rocke_kernel_def_t`).
- T1.4 **Round-trip idempotence**: `build → serialize → parse → serialize` is byte-identical, on both sides.
- T1.5 **Canonicalizer** for IR diff (stable value-id normalization, attr sorting) so semantic equality is detectable independent of incidental id gaps.

**AC.** Verifier verdicts identical Python↔C++ on the whole corpus; round-trip idempotent on both sides; serializer/parser fuzz-clean.

**Parallelism.** Verifier and serializer are separable (2 agents). This WS is *also the first half of the IR-seam collapse* — the serializer is reused as the runtime interchange.

---

### WS2 — The full differential test harness [the spine; highest priority after WS0/WS1 scaffolding]

**Objective.** One test runner that proves equivalence on everything, with detailed numeric coverage and repeatable examples. Layers L1–L6 (detailed in §7).

**Tasks.**
- T2.1 **Matrix enumerator**: family × instance × config (full enumeration, not sampled) × dtype × shape-corpus × arch × backend. Drives every layer.
- T2.2 **L1 verifier-parity**, **L2 IR-diff**, **L3 `.ll` sha-identity** lanes (CPU-only; run on every PR). L3 extends the current 63-pair harness to full enumeration + all arches + identical-rejection accounting.
- T2.3 **L4 fuzzer**: random *valid* spec generator (constrained by `is_valid_spec`) → assert L1+L2+L3 agreement, no crash, under **ASan/UBSan**. Seed-logged for reproducibility.
- T2.4 **L5 golden/regression gate**: lock current known-good `.ll`/HSACO digests (extend `_golden/`); changes must match or be explicitly blessed.
- T2.5 **L6 numeric on-GPU** (per §7): reference impls + tolerance table + cross-backend bit-identity + backend-vs-reference. Arch runners gfx942/gfx950/RDNA (multi-arch GPU CI cluster).
- T2.6 **Parity dashboard**: JSON + human report, per-family/arch/backend status over time; consumed by the soak (WS8).
- T2.7 **Example repeatability framework** (§7.5): every file under `examples/` becomes deterministic (fixed seeds), single-command, golden-recorded, registered, CI-smoked.

**AC.** All six layers runnable from one entrypoint; CPU lanes green in CI; GPU lanes green on each arch runner; dashboard emits per-family parity status; every example reproduces its recorded golden.

**Parallelism.** Strongly parallelizable by layer and by family (many agents). This is the gate that unblocks all subsequent risky work.

---

### WS3 — C99 → C++20 migration [gated by WS0+WS2]

**Objective.** Single, modern, hardened engine; namespaces fix duplicate symbols; exceptions mirror Python `raise`. **Output stays byte-identical** at every step.

**Hard rules.**
- R3.1 **Migrate module-by-module behind the L3 `.ll` gate.** Never big-bang.
- R3.2 **Two-step per module**: (1) compile-as-C++ with *zero* semantic change → verify byte-identical; (2) idiomatic refactor (containers/exceptions/namespaces) → re-verify byte-identical.
- R3.3 **Keep the `vsnprintf`-based numeric formatting core.** `std::format`/`std::to_chars` format floats/ints differently and would break byte-identity. Wrap, don't replace.
- R3.4 **Preserve a stable `extern "C"` ABI** for the provider during transition (provider is C++17; it can consume C++ later, but the ABI stays put until WS7 updates it).

**Tasks (in order).**
- T3.1 Primitives: `arena` (→ keep, optionally `pmr`), `vec.h` (→ `std::vector`/`pmr`), `strbuf` (→ `std::string` wrapper *around the kept vsnprintf core*).
- T3.2 Error model: sticky `builder->status` → **exceptions** (`ckc::ValueError`/`TypeError`/`KeyError`/`NotImplError` mapping to Python's). `extern "C"` shims translate exceptions → status codes for legacy callers.
- T3.3 Namespaces mirroring the WS0 folders (`ckc::core::ir`, `ckc::helpers`, `ckc::instances::gemm`, …) + `inline` header helpers → **eliminates the duplicate-symbol problem**; `cc`-equivalent whole-source build now links.
- T3.4 IR core, then `passes`, then lowerers (`lower/llvm`, `lower/hip`, `lower/cktile`), then instances family-by-family.
- T3.5 Turn on `-Wall -Wextra -Werror`; remove the `-Wno-*` suppressions; fix root causes. ASan/UBSan in the test build (catches the GCC arg-eval class).
- T3.6 Audit & fix all multi-side-effect-in-one-expression sites (sequencing) flagged by UBSan/grep-lint.

**AC.** Whole engine builds as C++20 with warnings-as-errors, clean under ASan/UBSan; **every L3 sha unchanged** vs the C99 baseline; `cc`-equivalent whole-program link succeeds (duplicate-symbol gone).

**Parallelism.** After primitives + error model + namespaces land (serial core), instances migrate family-by-family in parallel (many agents), each gated independently.

---

### WS4 — Python veneer / dual-backend flag (pybind11) [gated by WS1+WS2; overlaps WS3]

**Objective.** Same Python authoring API drives either engine; `both` differentially checks.

**Tasks.**
- T4.1 pybind11 module exposing the C++ engine: builder ops, lowering entry points, serialization/parse, verifier.
- T4.2 `IRBuilder` backend dispatch: native Python vs forward-to-C++. The 225-method surface is bound; generate bindings where mechanical.
- T4.3 `ROCKE_BACKEND` env + `backend=` API param wiring through `compile_kernel`, the instance builders, and `sweep`/`run_manifest`.
- T4.4 `both` mode: run both backends, assert equality (IR + `.ll`), surface a precise diff on mismatch; choose return per profile (Python in dev, C++ in runtime).
- T4.5 Wheel/packaging: ship the pybind extension; CI builds it for the supported toolchains.

**AC.** A representative kernel compiles identically via `python`, `cpp`, and `both`; `both` raises a precise, actionable diff on an injected mismatch; provider C-JIT path consumes the same C++ core.

**Parallelism.** 1–2 agents; depends on WS1 (serialization) and a usable C++ build (early WS3).

---

### WS5 — Close gaps & drift [continuous; driven by WS2 findings]

**Objective.** Every family Python supports reaches full C++ parity (IR + `.ll` + numeric). Each known gap is a task with the parity test as its acceptance criterion.

**Tasks (known gaps).**
- T5.1 **Fused-MoE e2e host runtime** (largest gap): port the NOTIMPL host orchestration (`fused_moe_e2e_*`, packed/permuted weights, HIP launch chain).
- T5.2 **RDNA WMMA op wiring** in the LLVM backend (`lower_llvm_mma.c:282`) + RDNA datalayout/triple (currently all backends share gfx950's). Unblocks `matmul_nbits`, `wmma_gemm*`, `wmma_fmha_fwd` end-to-end on gfx1151/gfx1201.
- T5.3 **Buffer-addrspace tensor views** (`tensor_view` f16 path intentionally unported).
- T5.4 **smoothquant / pooling distribution helpers** (`make_naive_tensor_view_packed`, `make_lds_view`).
- T5.5 **conv `LdsLayout`/`WarpGrid` accessors** + `validate_for_async` (currently a no-op).
- T5.6 **Attention I64-KV-addr / fp8-dequant side paths** (stubbed-to-link).
- T5.7 Verify the **`img2col`** "stubbed load" comment is stale vs real emission; reconcile.
- T5.8 **Drift tickets** for every divergence WS2 surfaces (e.g. `chiplet_swizzle` constant-folding + workgroup id read-order; any residual SSA-id drift after the serialization seam).

**AC.** Each gap closes with L1–L3 parity + L6 numeric within tolerance on the relevant arch; the gap inventory reaches zero for Python-supported families.

**Parallelism.** Highly parallel (one agent per gap/family).

---

### WS6 — Complete the dispatcher [gated by WS2; can overlap WS5]

**Objective.** Selection across all families, arch-correct, Python/C++ in lockstep.

**Tasks.**
- T6.1 Extend `dispatch/` beyond fp16 RCR GEMM: GEMM (all dtypes/layouts), conv, attention, MoE, norm — candidate registries, support predicates (generalize `gemm_config_supported`), spec selection, priorities, grid/block.
- T6.2 **Fix the `rdna_wmma` arch-gating bug**: add an arch-family gate to the candidate's support predicate so it cannot report `supported=True` on CDNA (gfx950).
- T6.3 Mirror every candidate in the C++ `Dispatcher`; extend `dispatch_parity` to **all families** (compare on structural tuples, not minted names).
- T6.4 Extend the ML heuristic models/feature extractors per family (GEMM 72 / FMHA 68 / conv 101 exist; add MoE/norm; keep feature parity with `ml_heuristic.hpp`).

**AC.** For every family, Python and C++ dispatchers select the same kernel (structural tuple) across the shape corpus; `dispatch_parity` is green for all families; the `rdna_wmma` regression test fails on the old code and passes on the fix.

**Parallelism.** One agent per family registry.

---

### WS7 — hipDNN end-to-end hardening [gated by WS5+WS6]

**Objective.** Nothing fails E2E on hipDNN across supported ops/dtypes/layouts.

**Tasks.**
- T7.1 Expand `integration_tests/EndToEnd*` to all supported ops × dtypes × layouts; add the missing SDPA cases (B>1, BHSD-transpose, paged-KV graph passthrough) and general GEMM B-layout detection (vs RCR-assumed).
- T7.2 Harden the runtime: kernarg byte-packing bounds (per `args_signature`), comgr/HIP error checking, module/handle lifetime (no leaks), LightGBM C-API error paths.
- T7.3 Fuzz the provider param parsers (`Rocke*ParamParser`).
- T7.4 Make `ROCKE_C_JIT=1` (C++ engine) the **validated** provider path; numeric E2E gates (`max_abs_diff` thresholds) per op.
- T7.5 Provider-side golden: pin selected kernel + output digest per E2E case.

**AC.** All E2E demos green on each arch runner; numeric gates within tolerance; sanitizer-clean provider build; no leaks under a launch-loop stress test.

**Parallelism.** Per-engine (GEMM/Attention/Conv) agents.

---

### WS11 — Reconcile drift with the merge-target `rocke-prototype` [pre-merge]

**Objective.** This branch (the `rocke-c-interface` branch) will PR into the `rocke-prototype` branch (a separate worktree of this repo). The prototype has **8 commits / ~16.5k Python lines not here** (merge-base `42a064df` = "dispatcher prototype #8237"). Most are **codegen-affecting**, so the C++ backend — built to match *this* branch's Python — will **drift** from the merged Python unless reconciled. Ensure the target's changes are **captured here** and **mirrored in the C++ backend** so the merge introduces **zero codegen drift**.

**The target delta** (`git diff 42a064df..60283014863 -- rocke`):

| Change | Files | C++ impact |
|---|---|---|
| #8624 vector-sizes-as-args (implicit gemm) + #8355 **conv 3d** | `conv_implicit_gemm.py` +470 | mirror in C++ conv |
| #8348 **fp32/bf16 MFMA** extension | `helpers/atoms.py` +80, `mfma_gemm_inner.py` | new atoms in C++ atoms/mma + catalog |
| #8320 gfx950 compv4 **schedule hints** | `gemm_universal.py` +76, `helpers/schedule.py` +178, `loads.py` +108 | mirror in C++ |
| #8313 **fix backend drift llvm22** + #8293 ISA backend cracks | `core/lower_llvm.py`, `isa/backend.py`, `arch/target.py`, `arch_specs.json`, `ir.py` | reconcile C++ lowerer/isa |
| #8486 verification improvements | verifier paths | reconcile with WS1 `verify.py` |
| #8609 **gfx1250** (RDNA4) new arch | `instances/gfx1250/*` ~6k + examples + tests | **new** ckc gfx1250 backend + instances (large) |
| attention/moe deltas | `attention_unified.py` +349, `moe_*`, `fused_moe_e2e.py` +124 | mirror in C++ |

**Method.** Point the differential harness's **Python side at the target tree** (`PYTHONPATH=<target-worktree>/dnn-providers/hip-kernel-provider/rocKE/Python`) against the current C++ engine → `run_diff` enumerates the **exact per-family C++ drift** = the authoritative work-list. Mirror each codegen change in C++; gate `run_diff` GREEN vs target-Python.

**Tasks.** T11.1 core backend fixes (#8313/#8293) → C++ `lower_llvm`/`isa`. T11.2 conv (#8624/#8355) → C++ conv. T11.3 atoms fp32/bf16 (#8348) → C++ atoms/mma/catalog. T11.4 gemm/schedule (#8320). T11.5 attention/moe deltas. T11.6 **gfx1250** new arch (assess: new ckc backend + instances; prioritize after existing-family drift). T11.7 **capture-here** — ensure this branch's Python reflects target changes; **flag conflicts** with WS1/WS6 additions (`core/__init__.py`, `dispatch/gemm/*` both diverged from the #8237 base).

**AC.** `run_diff` (C++ vs target-Python) GREEN for all existing families; gfx1250 C++ parity tracked; merge introduces no codegen drift. Coordinate with the owner-driven git merge. Feeds WS8 (soak the reconciled state).

---

### WS10 — Complete idiomatic C++20 migration [continues WS3; no shortcuts]

**Objective.** WS3 Phase 1 only made the engine *compile* as C++20 (C-style code through the C++ frontend). WS10 is the **actual idiomatic conversion** — and explicitly includes the items that look "cosmetic" but are required for the long-term mirrored dual-backend to stay maintainable (1:1 Python↔C++ correspondence, clean namespaces for future provider direct-linking). **Nothing is skipped as cosmetic.**

**Hard rules (same as WS3).** Two-step per module (compile-clean → then refactor); **byte-identity gated after every module** (`run_diff --mode ll` 62 GREEN/0 DRIFT, `--mode ir --canonical` 61 GREEN/1, the L5 golden, ctest); **keep the `vsnprintf` numeric-formatting core** (std::format/to_chars format floats differently → would break byte-identity).

**Tasks.**
- **T10.1 Exceptions.** Replace the sticky-error status model (`builder->status` / `ROCKE_ERR_*` returns) with C++ exceptions (`ckc::ValueError`/`TypeError`/`KeyError`/`NotImplError`/`OOM`) mirroring Python's `raise`. Preserve a **stable `extern "C"` ABI**: public C wrappers `try/catch`→status code for legacy callers (provider, emitters, bindings); internal C++ throws. This is the single biggest drift-reducer (makes the port a literal Python translation).
- **T10.2 std containers.** `vec.h` `ROCKE_VEC` macro → `std::vector` (pmr/arena allocator to keep arena lifetime + determinism); `strbuf` → `std::string` wrapper **around the kept `vsnprintf` core**; `arena` kept (or `pmr::monotonic_buffer_resource`).
- **T10.3 Namespaces.** Wrap all internal symbols in `namespace ckc { ... }` with sub-namespaces mirroring the folders/Python (`ckc::core::ir`, `ckc::helpers`, `ckc::instances::gemm`, …); `extern "C"` only on the public ABI. (Duplicate-symbol turned out moot, but do it for structure + future provider direct-link — **not** skipped.)
- **T10.4 Rename + header foldering (cosmetic but REQUIRED).** `src/**/*.c` → `*.cpp` (all 222), drop the `LANGUAGE CXX` override for a native CXX glob; update the harness (`run_diff` `*.c`→`*.cpp`, the `run_*parity.sh`). Complete the **header foldering deferred from WS0**: `include/rocke/` flat (incl. dot-named `helper_rocke.helpers.spec.h`) → subfolders mirroring `src` (`helpers/spec.h`, …); update every `#include`. Result: **full 1:1 Python↔C++ tree + namespace correspondence.**
- **T10.5 Idiom + full `-Werror`.** Finish removing every `-Wno-*`; RAII for owned resources, `nullptr`, const-correctness, `enum class` where safe — without changing `.ll`.

**AC.** Idiomatic C++20 engine; `-Werror` + sanitizer clean; `.ll`/IR byte-identical to baseline at every step (golden holds); 1:1 Python↔C++ file/namespace correspondence; provider + `bindings/` + emitters still build. **Boundary:** WS10 is the engine's internal conversion; the `ROCKE_BACKEND=python|cpp|both` veneer wiring into Python `ir.py`/instances stays under WS4.

**Parallelism.** Largely sequential (exceptions + namespaces touch everything); instances can migrate in parallel batches once the core lands. Blocked by WS3 (Phase 1 + hardening/golden); feeds WS8 (the idiomatic engine must be soaked).

---

### WS9 — `rocke.dispatch` full family completeness [extends WS6]

**Objective.** Drive the dispatcher from the **2 implemented GEMM families** (fp16 RCR, bf16 RCR `UniversalGemm`) to **full family coverage**, Python + C++ in lockstep, parity-verified.

**Current state (post-WS6).** `rocke/dispatch/gemm/{fp16_rcr,bf16_rcr,common,support}.py` implement fp16 & bf16 RCR. The `dispatch_parity` harness covers both (`--dtype fp16|bf16`): fp16 compares against the **real shipped `kernels/gfx950/` HSACO bundle**; bf16 has no shipped bundle, so `dispatch_parity/gen_bf16_bundle.py` **synthesizes a manifest-only bundle** from the Python bf16 candidates and the real `rocke::Dispatcher` selects over it (CPU-only, no kernel materialized) — manifests minted from the same `UniversalGemmSpec` the Python side selects, so any reported divergence is a genuine **selection-logic** difference. `conv/SDPA/fp8/other layouts` are **scaffolded** in `rocke/dispatch/families/{conv,attention,moe,norm}.py` but not yet implemented.

**Tasks.**
- T9.1 Implement each scaffolded family (`families/conv.py`, `attention.py`, `moe.py`, `norm.py`) + remaining GEMM dtypes (**fp8**) and layouts (RCC/CRR/…): candidate registry, support predicate (generalize `gemm/support.py:gemm_config_supported`), `dispatch_<family>` selection, grid/block, sweep space.
- T9.2 Mirror every new candidate in the C++ `rocke::Dispatcher`; keep the shared `cache_key` identity.
- T9.3 Extend `dispatch_parity` per family (structural-tuple compare). Reuse the **manifest-synthesis pattern** (`gen_bf16_bundle.py`) for families with no shipped HSACO bundle; compare against the real bundle where one ships.
- T9.4 Extend the ML heuristic feature extractors per family (GEMM 72 / FMHA 68 / conv 101 exist; add MoE/norm), keeping byte-parity with `ml_heuristic.hpp`.

**AC.** `dispatch_parity` GREEN for every implemented family; Python↔C++ selection identical across the shape corpus; the manifest-synthesis path proves selection-logic parity even before kernels ship.

**Parallelism.** One agent per family (fan-out); each gated by its `dispatch_parity` entry. Blocked by WS6 (framework + `rdna_wmma` fix); feeds WS7 (provider engines) and WS8 (soak).

---

### WS20 — Personal-ID / hardcoded-path scrub + guard [pre-merge]

**Premise.** Committed code/scripts/docs must carry **no personal identifiers or machine-specific paths**.

- **Sweep targets** (committed source/scripts/CI/tests/docs, *not* build dirs): usernames/NTIDs, emails (e.g. `@amd.com`), internal hostnames (login hosts, node names), personal absolute paths (absolute `workspace`/`home` checkout paths, `~/work`, `/tmp/claude*`, personal venvs, `~/.ssh/<key>`).
- **Replace with:** repo-relative paths (from `__file__`/script dir), **env vars** (`$USER`/`$TMPDIR`/`$HIPDNN_ROOT` etc.), or documented placeholders (`<login-host>`/`<user>`/`<repo>`); a gitignored-local-config pattern (e.g. `~/.rocke_env`) for remote-cluster scripts. Doc branch refs → keep if legitimately documenting an upstream branch, else `<target-branch>`.
- **Guard:** the deny-list check `Cpp/ci/tiers/check_no_personal_ids.sh` rejects the personal-id/hostname/personal-path patterns; it is wired into the static CI tier and the pre-commit config. The deny-list of personal tokens lives at the top of that script and is easy to extend.

**AC.** Zero personal usernames/emails/internal-hostnames + zero personal absolute paths in committed source/scripts/CI/tests; docs genericized/placeholdered; deny-list guard live; functionality preserved via env/relative. **Sequencing:** after WS10 (engine quiesces), coordinate the deny-list with WS19; sweep the final state once. Blocks the PR/merge.

---

### WS19 — Build/config/artifact hygiene: make staleness loud-failing [defense-in-depth]

**Premise.** *Every* observed false failure was a stale/wrong artifact used silently — the stale `Cpp/build` archive (`ROCKE_LIB` defaulted to it → "Conv C-JIT broken"), the cluster sync shipping stale `src` (→ phantom `-Werror=switch`), missing `-D__HIP_PLATFORM_AMD__`/`HIPDNN_BUILD_DIR` (→ "demos don't compile"), stale memory gap-lists. The engine was fine; the *periphery* drifted. Goal: staleness must **fail loud, not silently**.

**Key constraint:** stamp **artifacts, not the emitted `.ll`** — a hash comment in the `.ll` would break the byte-identity differential gate (Python doesn't emit it). Stamps live in the archive/HSACO/manifest metadata; consumers validate those.

- **L1 — Versioning / freshness stamps (the core guard).** Single `ROCKE_ENGINE_VERSION` (semver) + a build-id = content-hash of the engine source. Embed in artifacts: `rocke_build_id()`/`rocke_engine_version()` in `librocke_core.a`, a `.note` on the comgr HSACO, `engine_version`+`source_hash` in `kernels/<arch>/manifest.json`, engine ver in the `ck.dsl.ir/v1` header. **Consumers validate + fail loud on mismatch** (provider C-JIT checks the archive's source-hash vs current; runtime checks manifest compat).
- **L2 — Commit guards.** `.gitignore` all engine build products (`build*/`, `*.a/.o/.so`, `CMakeCache`, `_deps/`); pre-commit hook **rejects** committing them (so `Cpp/build` never re-commits); remove the checked-in stale build dirs. Distinguish the *intentionally-shipped* `kernels/<arch>` bundles (committed, but L1-stamped + regeneratable).
- **L3 — Single canonical build entrypoint.** One script/top-CMake that builds engine+provider+demos with all correct flags/paths baked in (`HIPDNN_ROOT`/`HIPDNN_BUILD_DIR`/`-D__HIP_PLATFORM_AMD__`/the full `_deps` include set/`ROCKE_LIB`→**fresh** archive). The provider builds the engine fresh, not from a stale checked-in `.a`. No hand-rolled `g++`.
- **L4 — CI gates (blocking).** Build-from-clean; the differential harness (vs Python + vs target); "no committed build artifacts"; L1 stamp validation; the golden gate.
- **L5 — Sync integrity.** Replace rsync-with-gitignore-filter (shipped stale copies) with content-hash-verified sync / `git archive` snapshots; remote build verifies source-hash before building.
- **L6 — Documentation.** One canonical BUILD/hygiene doc; docs carry "verified against `<commit>`"; extend `verify_dsl_docs.py` to actually *run* the documented build recipes so docs can't go stale-wrong.

**AC.** A stale/mismatched artifact cannot be used silently (loud version-mismatch); build products can't be committed; one canonical build path; CI enforces clean-build + freshness + differential + golden; remote sync integrity-checked. **Non-engine layers (L2/L3/L6) can start now; L1 stamps coordinate with WS10/WS12 and must not alter emitted `.ll`.**

---

### WS18 — Adversarial code review & hardening [dynamic workflows; feeds WS8]

**Premise.** A comprehensive review + hardening of **all** code this program produced/touched — broader than WS3-hardening (which was `-Werror`+sanitizers on the C engine alone). Run as **dynamic workflows**: fan out finders per (dimension × area) → **adversarially verify** each finding (independent refuters, default-refuted, majority vote — kills noise) → synthesize confirmed issues → fix → re-verify. Every fix harness-gated (`run_diff`/golden/numeric unchanged, tests green).

**Areas (all):** the C++ engine (`core/ir`, lowerers, instances, helpers), Python `rocke` (core, helpers, instances, dispatch, heuristics, `backend.py`, `ir_serialize`/`verify`), the provider (runtime, engines, plans, dispatcher, comgr, ml_heuristic), the `rocke_engine` bindings, the differential harness itself, the GPU CI cluster scripts.

**Dimensions (one workflow wave each, fanned across areas, adversarially verified):** (1) correctness/logic (esp. the recent conv rename/3d, sched-suppression, datalayout, atoms, agpr_alloc); (2) memory/resource/lifetime safety (arena ownership, leaks, UAF, the RAII conversion); (3) error-handling + edge cases + OOM + boundary shapes; (4) **security/input-validation** (comgr `.ll` feed, kernarg packing bounds, the IR parser, provider ParamParsers, C-JIT); (5) API/ABI correctness (`extern "C"` boundary, exceptions-across-ABI shims, version gating); (6) concurrency/reentrancy (launcher FIFO, multi-stream, runtime cache, harness parallelism); (7) byte-identity + determinism invariants (the `vsnprintf` core, arena-order determinism); (8) perf/quality (non-blocking).

**Cadence.** An **early partial pass** can run now on the stable verified components (harness, dispatch, serializer/verifier, reconciled engine, bindings); a **final comprehensive pass** after WS10/WS16/WS17 stabilize. **AC:** adversarially-confirmed issue inventory with fixes landed + harness-green; security surfaces fuzzed; no unhandled OOM/edge/ABI hazards; perf findings logged.

---

### WS17 — Boilerplate reduction [after WS10/WS16; byte-identity gated]

**Principle.** Cut **mechanical** duplication without collapsing the deliberate **1:1 Python↔C instance correspondence** (load-bearing for parity) or hurting readability/over-abstracting. Every change **byte-identity gated** (`run_diff` ll/ir + L5 golden unchanged) + tests green. Refactors *stable* code once, so it sequences after the engine settles. Complements WS10 (which already removes void*/sticky-error/manual `vec`+`strbuf` boilerplate via C++ idioms — not re-done here).

- **W1 Parity-emitter consolidation.** The 63 `tests/parity/*_emit.{c,py}` pairs are near-identical (`main` + `make_cfg` + ll/ir/verify dispatch). Replace with a **shared table-driven driver** (one C + one Python), each family providing only its config table + build entry. Biggest single cut. Gate: `run_diff` identical.
- **W2 C instance-builder skeleton.** Factor the repeated `_spec_default/_finalize/_kernel_name/_is_valid_spec/build_*/_lower_to_llvm` scaffolding into macros/helpers (or templates post-C++), only where it doesn't damage the mirror.
- **W3 Binding-glue reduction.** Macro/template-generate WS16's per-family binding + `spec_to_dict` (a registration macro + spec-field descriptors) instead of ~145 hand copies.
- **W4 Script/registry DRY.** Delete redundant `run_*.sh` (superseded by `run_diff.py`); unify dispatch registries + `lever_spaces`; shared tolerance/reference tables; shared sbatch templates.
- **W5 Duplication audit.** Similarity scan across `src/instances`+helpers+Python instances; cut safe copy-paste, report intentional parity anchors.

**AC.** Measurable duplication reduction, **zero output change** (golden holds), tests green, 1:1 structure + readability preserved, no over-abstraction. Feeds WS13 (docs reflect simpler structure) and the WS8 soak.

---

### WS16 — Extend the cpp binding to the full instance catalog [after WS10; feeds WS12/WS8]

**Objective.** The `rocke_engine` pybind binding + `backend.py` dispatch currently cover **universal GEMM only**. Until they cover **every** instance family, `ROCKE_BACKEND=cpp|both` and the IR-seam cpp-lowerer path are GEMM-only. WS16 extends the binding to the **whole catalog** (~145 builders).

**Scope (every family with a C builder).** GEMM (universal ✓, batched, grouped, multi_d/abd, streamk, mfma, flatmm, block_scale, mx, matmul_nbits, batched_contraction, wsp3); conv (implicit_gemm, direct_grouped, img2col, deep_fused_conv_pool); attention (unified, all fmha_*, sage, sparse, tiled_2d/3d per arch); MoE (fused_moe, fused_moe_e2e, moe_fused_mega(_fp8), moe_gemm_fused, moe_sorting, moe_smoothquant, topk_softmax); norm (layernorm2d, rmsnorm2d, add_rmsnorm2d_*); elementwise/reduce/pooling/transpose/permute/smoothquant; RDNA wmma_*; per-arch variants.

**Per family.** (1) `rocke_engine.<fam>_lower_llvm`/`_serialize_ir`/`_verify`/`_is_valid` over the extern-`C` entry points; (2) a `<fam>_spec_to_dict` converter (each family's spec dataclass↔C struct — the bulk of the work); (3) a `lower_<fam>(spec,arch,backend)` dispatch entry mirroring `lower_universal_gemm`. The python/cpp/both + `BackendMismatch` diff machinery is family-agnostic and reused as-is.

**Validation per family.** binding-cpp `.ll` == the C engine's own `.ll` (faithful exposure; compare vs the family's parity emitter) AND == Python where they match. (Post-reconciliation the C engine matches *target* Python for the reconciled families, so `both` vs *current* Python legitimately diverges there until the merge.)

**AC.** `ROCKE_BACKEND=cpp` lowers every family; `both` runs for every family (GREEN where C-engine==Python). **Fan-out by family-group.** Sequenced after WS10 (stable idiomatic engine + extern-`C` ABI); feeds WS12 (cpp lowerer path) and the WS8 soak (full-catalog differential).

---

### WS14 / WS15 — Stratified, multi-arch CI [sequence after WS13]

**Premise.** The library spans no-GPU logic, comgr compilation, GPU numeric execution, integration layers, and performance — so CI must be **stratified** and **multi-arch**. Today: Tier-0 strong (`run_diff` ll/ir is pure text-gen; dispatch/verify/serialize units), Tier-2 partial (gfx950-only numeric), Tiers 1/4/5 + all multi-arch numeric missing.

**WS14 — tiers + instance suite.**

| Tier | Scope | Gate |
|---|---|---|
| T0 no-GPU | imports, units, IR build, lowering-text shape, transform-DAG, helper contracts, arch metadata, dispatcher reg+rejection, docs-link | per-PR |
| T1 compile-only | comgr HSACO per arch, **no launch**; structural `.ll` checks (triple/datalayout/intrinsic-counts/addr-shape/vectorization/waitcnt/barrier/metadata) or IR checksum (golden) | per-PR (targeted) |
| T2 numeric smoke | small shapes/instance, numpy/torch/hipDNN-CPU refs, tight tol, short | nightly/arch |
| T3 integration | examples, manifest_runner, torch_backend, hipDNN provider, pipelines, runtime cache | nightly |
| T4 model slices | small deterministic E2E (attention / MoE / norm+matmul / quant-linear / conv-fusion) → operator+dispatcher coverage | nightly |
| T5 perf | scheduled/manual cohorts, median+spread, **non-blocking** alerts | scheduled |

- **Instance suite:** `python/test/hipkg_instances/` — `test_{gemm,conv,attention,norm_reduce_small_ops,quant_moe_sparse}_family.py` + `lever_spaces/{gemm_universal,conv_implicit_gemm,attention_unified}.py`. Each: spec-validation + invalid combos; signature/grid; compile-only smoke per arch; numeric smoke where GPU; **lever-space sampling** (not full combinatorial); dispatcher reg + rejection explanations.
- **Compile-only CPU strategy:** CPU refs for *semantics* (numpy/torch/hipDNN-CPU/tiny KernelDef interpreter for scalar ops); LLVM-IR *structural* validation or IR checksum for compiler evidence; HIP-debug lowering as a readable cross-check only (**never** run AMDGPU IR on CPU as correctness).
- **Filters:** op / arch / dtype / family / smoke|full / numeric|perf / integration-target. **Split** must-pass-per-PR (T0 + targeted T1) vs nightly GPU/perf.

**WS15 — multi-arch execution.** Run the tiers across **gfx942/gfx950/gfx1151/gfx1201/gfx1250** on the multi-arch GPU CI SLURM cluster (per-arch nodes). SLURM job scripts per tier per arch (T0/T1 on ROCm-no-GPU nodes; T2-T4 on per-arch GPU nodes; T5 cohorts); HSACO cache across runs (stable key) to cut GPU-node dependency; stratified scheduling (T0/T1 per-PR, T2-T4 nightly/arch, T5 scheduled); aggregate multi-arch dashboard.

**Success.** A helper-only PR gets high confidence from T0 + targeted T1; GPU CI used only where genuinely required; perf regressions carry enough metadata to separate source-regression from environment drift. The WS8 soak runs over this multi-arch surface.

---

### WS13 — Documentation & comment cleanup [functionality-first; sequence last]

**Premise.** The WS/wave/RFC nomenclature in this document is **internal program management** — it must NOT appear in any user-facing doc (READMEs, the rest of `dsl_docs`) or inline code comment. Those describe **functionality and usage**. Conversely, the real capabilities built across this program must be **documented as functionality**. This RFC is the *sole* permitted home of the WS language (and may be relocated out of the shipped `dsl_docs` tree). **Standing rule for all contributors/agents:** never embed `WSn`/`Wave X`/`RFC §…`/task numbers in shipped docs or comments — write what the code does and why.

**Wave 1 — Scrub.** Remove every WS/Wave/RFC-cross-ref/task-number reference from user-facing docs + inline comments across `rocke`, `Cpp`, `rocke-provider`; reframe each as a functionality/why comment. (Agents have added many, e.g. `RFC WS1.T1.2`, `WS2 differential`, `(WS11)`.)

**Wave 2 — Cover** (document new functionality in READMEs + `dsl_docs` + docstrings):
- `ck.dsl.ir/v1` serialization (`serialize`/`parse`/`canonicalize`) → `dsl_docs/ir_lowering` + reframe `ir_serialization_format.md` as a clean format reference.
- The IR verifier (`verify`/`verify_or_raise`).
- The differential test harness (`run_diff.py` modes `ll`/`ir`/`verify` + `--canonical`/`--pyroot`/`--golden`, `numeric.py`, the golden anchor) → a tests README or `dsl_docs/development/testing.md`.
- Dispatch family coverage (gemm fp16/bf16, norm, conv, attention, moe) → `dispatch/README.md`.
- The `Cpp` C++ engine README (what it is, C++20/CMake build, the parity harness, the bindings).
- The pybind `rocke_engine` bindings README; numeric-harness usage.

**Wave 3 — Reconcile + consistency.** Update stale docs for the reorganized C tree (`reference/file_index`); reconcile `limitations.md` (MLIR-as-input stays a non-goal; `ck.dsl.ir/v1` is a *machine* interchange, not human authoring); refresh `reference/` indexes (`api_index`, `op_vocabulary`) for the new public surface; style/consistency pass; verify in-doc examples run (tie to example-repeatability).

**AC.** Zero WS/wave/RFC-internal nomenclature in any user-facing doc or inline comment; all new functionality documented (what/how) in READMEs + `dsl_docs` + docstrings; docs reflect the final reorganized C++ tree.

**Sequencing.** **Last.** Runs after the code stabilizes (WS10/WS11/WS12) so docs match final functionality and the comment-scrub is done once, not repeatedly against in-flight code.

---

### WS12 — IR-seam collapse: shrink the dual-builder drift tax [resolves D3]

**Premise (proven empirically).** The dual *builders* drift silently — the 42-family arg-eval-order finding (WS5) showed it's structural, recurring, and invisible to `.ll`. The harness makes it *survivable*, not *cheap*. WS12 shrinks the tax at the root: make **`ck.dsl.ir/v1` the runtime interchange** so there is **one lowerer of record** (C++) consuming serialized IR. For baked/shipped instances this **eliminates the C builder** — no C builder, no arg-eval drift. The instance **builders stay dual and per-instance selectable** (Python authoring + C++ runtime-JIT): the collapse is at the *lowerer*, not the builders. Byte-identity is then required only for the **flex set** (instances that need a C builder for runtime arbitrary-shape JIT), and there the gate is **IR-level** (cheaper, upstream of `.ll` text). Foundation already built: WS1 serializer + `rocke_ir_parse`, WS4 `ROCKE_BACKEND` frontend flag, the differential harness.

**Wave A — IR-artifact lowering pipeline.** End-to-end `Python KernelDef → serialize ck.dsl.ir/v1 → C rocke_ir_parse → C lower → comgr → HSACO`. Cross-engine equivalence test: **Python-built IR lowered by C == Python's own lower** (byte-identical) across families (extends WS1's C round-trip to the cross-engine direction). Deliver a `lower-from-IR` entry + test.

**Wave B — provider/runtime IR-artifact path.** Add IR-artifact serving to `rocke-provider` (ship serialized IR per baked instance; provider `parse → lower → comgr`, no C builder, no Python) alongside Fast(HSACO)/JIT(`.ll`)/C-JIT. Offline `KernelLibraryGen` emits IR artifacts from Python.

**Wave C — flex-set designation + builder retirement.** Classify instances: **IR-served** (baked) vs **C-builder-required** (runtime arbitrary-shape JIT). For IR-served, freeze/remove the C builder off the critical path; the differential IR-diff now gates **only the flex set**. Per-instance frontend flag (ties to WS4 `ROCKE_BACKEND`).

**Wave D — harden the IR contract.** Mandatory IR verifier on parse (reject malformed at the boundary); versioned format gating + migration policy; **fuzz the IR parser** (malformed/adversarial); **golden IR artifacts** per instance (the IR becomes a blessed, checked-in artifact); IR-diff + IR-verify as permanent CI gates for the flex set.

**AC.** Baked instances served via Python-IR → single C lowerer (no C builder); flex set explicitly enumerated + IR-diff-gated; provider IR-artifact path works (Python-free); IR contract hardened (verify-on-parse + parser fuzz + golden). Resolves D3.

**Dependencies.** Builds on WS1/WS4 + the harness; informed by WS10 (the single lowerer should be the idiomatic C++ one) and the WS11 flex decisions. Feeds WS8 (soak the collapsed pipeline).

---

### WS8 — Differential soak & default flip [final]

**Objective.** Prove parity over real calendar time before changing the runtime default.

**Tasks.**
- T8.1 Run `both` mode in CI on every PR + a **nightly full-matrix sweep** (all families/configs/arches/backends), feeding the WS2 dashboard.
- T8.2 Triage any mismatch to a WS5 drift ticket; reset the soak clock on a real divergence.
- T8.3 **Exit criterion**: zero mismatches across the full matrix for a sustained window (proposal: **14 consecutive green nightly runs**, ≈2–4 weeks accounting for fixes).
- T8.4 Flip the provider/runtime default to the **C++ backend**; keep Python as authoring + permanent differential oracle (`both` stays on in CI).

**AC.** 14-green-nightly window achieved; default flipped; both layers remain mirrored with differential CI permanently enabled.

---

## 6. Why this ordering (collapse strategy)

We adopt the **IR-seam collapse (a)** mechanically — the serializer from WS1 *is* the interchange — while delivering the **veneer/dual-backend (toward seam b)** the owner prefers via WS4. Concretely:
- The serializer makes the IR the artifact ⇒ a single lowerer of record is *possible*, and the numbering-drift class dies (explicit SSA ids).
- The veneer lets Python drive the C++ engine ⇒ the *complete* collapse (one engine, Python as binding) is reachable later without a rewrite, because the bindings + differential harness already exist.
- Keeping both instance layers mirrored (owner's requirement) is sustainable *because* the differential harness makes drift cheap to detect — that is the trade we are explicitly buying.

---

## 7. Test strategy (deep dive)

### 7.1 Layered model
| Layer | What it proves | Where it runs | Gate |
|---|---|---|---|
| L1 verifier-parity | IR well-formedness; verifier agreement | CPU / per-PR | block |
| L2 IR-diff | builder equivalence (upstream of text) | CPU / per-PR | block |
| L3 `.ll` sha-identity | lowerer equivalence (incl. identical rejection) | CPU / per-PR | block |
| L4 fuzz + sanitizers | no crash, agreement on random valid specs, UB-free | CPU / nightly + per-PR smoke | block on crash/UB |
| L5 golden/regression | no unintended output change | CPU / per-PR | block unless blessed |
| L6 numeric on-GPU | correctness vs reference + cross-backend bit-identity | GPU arch runners / nightly + targeted per-PR | block on arch lanes |

### 7.2 Numeric reference implementations (L6)
Per family, a trusted CPU/torch reference with **explicit dtype + accumulation** matching:
- GEMM / batched / grouped / streamk / multi-d: `torch.matmul` in f32 accumulate, cast to output dtype.
- Conv (implicit-gemm / direct / img2col): `torch.nn.functional.conv2d` (grouped where relevant).
- Attention (all FMHA variants, unified, paged, splitkv, varlen, appendkv, bwd): `F.scaled_dot_product_attention` + hand-rolled references for paged/varlen/append/bwd specifics.
- MoE (fused, e2e, mega, sorting, topk): composed torch reference (gate → topk → expert GEMMs → combine), block-scale aware.
- Norm / elementwise / reduce / pooling / transpose / permute: direct torch equivalents.
- Quantized (fp8 block-scale, mx, matmul_nbits i4, smoothquant): numpy/torch dequant references matching the OCP/FNUZ/block-scale semantics.

### 7.3 Tolerance table (calibrate against existing parity harnesses; starting points)
| Output dtype (f32 acc) | backend vs reference | cross-backend |
|---|---|---|
| f32 | rtol 1e-5, atol 1e-6 | **bit-identical (0)** |
| f16 | rtol 1e-2, atol 1e-2 | bit-identical |
| bf16 | rtol 1.5e-2, atol 1e-2 | bit-identical |
| fp8 e4m3 / bf8 | rtol 3e-2 (block-scale aware) | bit-identical |
| int8 / i4 | exact (0) | bit-identical |

Cross-backend is **bit-identical** because both backends emit the same HSACO; any non-zero cross-backend delta is a drift bug, not a tolerance question. (Final values are calibrated per-family against the current Python parity baselines, not invented.)

### 7.4 The matrix
`family × instance × config(full enumeration) × dtype × shape-corpus × arch × backend`. The shape corpus per family includes: aligned, padded/tail, prime/odd, skinny-M (decode), large, and the edge shapes the existing stress harnesses already exercise. Arches: gfx942, gfx950 (CDNA); gfx1151, gfx1201 (RDNA). Backends: `python`, `cpp` (and `both` as the comparator).

### 7.5 Example repeatability (G3)
- Every file under `examples/` gets: fixed RNG seeds; a single documented command; a recorded golden (output digest and/or numeric reference); registration in `examples/REGISTRY.md`; a CI smoke lane.
- A `examples/run_all.py --check` driver runs each example and asserts its golden, so "the examples are repeatable" is *enforced*, not aspirational.

---

## 8. Instance family coverage matrix

Priority: **P0** = ship-critical / on the provider's E2E path; **P1** = broad use; **P2** = specialized. Complexity drives test depth and migration risk.

| Family | Members | Complexity | C-port status (baseline) | Priority |
|---|---|---|---|---|
| GEMM core | universal, batched, grouped, multi_d, multi_abd, streamk, mfma_gemm, flatmm | high | ported; parity-tested | **P0** |
| GEMM quantized | block_scale, mx_gemm, matmul_nbits (i4), batched_contraction | high | matmul_nbits RDNA-only (needs WS5.2) | P1 |
| GEMM experimental | wsp3 (warp-specialized) | very high | partial | P2 |
| Conv | conv_implicit_gemm, conv_direct_grouped (4c/16c), img2col, deep_fused_conv_pool | very high | implicit-gemm/deep-fused ported; img2col verify; LdsLayout gaps | **P0** |
| Attention | attention_unified (2d/3d/reduce), fmha_mfma/varlen/appendkv/paged_prefill/splitkv_decode/head_grouping/bwd/fwd_fp8, tiled_2d/3d per arch | very high | broad; I64-KV/fp8 side paths stubbed | **P0** |
| Attention specialized | sage_attention, sparse_attention | high | partial | P2 |
| MoE | fused_moe, fused_moe_e2e, moe_fused_mega(_fp8), moe_gemm_fused, moe_sorting, moe_smoothquant, topk_softmax | very high | **e2e host runtime NOTIMPL (WS5.1)** | P1 |
| Norm / quant | layernorm2d, rmsnorm2d, add_rmsnorm2d_rdquant, smoothquant | medium | smoothquant distribution gaps | P1 |
| Elementwise / misc | reduce, elementwise, transpose(_bc/batched), permute_nd, pooling | medium | pooling distribution gaps | P1 |
| RDNA WMMA | wmma_gemm, wmma_gemm_int8/iu8, wmma_fmha_fwd | high | **WMMA op unwired in LLVM backend (WS5.2)** | P1 |

The "very high" complexity families (conv, attention, MoE) get the deepest numeric corpus and are the explicit focus of "especially the more complex ones."

---

## 9. CI/CD gates

| Lane | Trigger | Contents | Blocking |
|---|---|---|---|
| `parity-cpu` | every PR | L1 + L2 + L3 (full enumeration), build clean-link, warnings-as-errors | yes |
| `fuzz-smoke` | every PR | L4 short budget under ASan/UBSan | yes (crash/UB) |
| `golden` | every PR | L5 digests | yes (unless blessed) |
| `numeric-gfx942/gfx950/rdna` | nightly + label-triggered | L6 on arch runners | yes on arch lanes |
| `dispatch-parity` | every PR | C++↔Python selection across all families | yes |
| `e2e-hipdnn` | nightly | provider EndToEnd, all ops/dtypes/layouts, leak/stress | yes |
| `examples` | every PR (smoke), nightly (full) | `examples/run_all.py --check` | yes |
| `soak-nightly` | nightly | `both` full matrix → dashboard | tracked (WS8 exit) |

GPU runners: the multi-arch GPU CI cluster (gfx942/gfx950/RDNA nodes) provides the arch lanes; jobs scheduled to avoid cross-thermal-state measurement (numeric correctness is thermal-independent; any perf assertions use same-session ratios only).

---

## 10. Rollout, sequencing, and parallel-agent map

```
Phase 0  (foundation, parallel)         WS0 reorg  |  WS1 verifier+serialize  |  hardening: warnings+sanitizers scaffolding
   │
Phase 1  (the spine)                     WS2 full harness (L1–L6) + example repeatability   ── establishes BASELINE parity, surfaces current drift
   │
Phase 2  (engine + veneer, overlap)      WS3 C++ migration (core serial → instances parallel)  ||  WS4 pybind dual-backend flag
   │
Phase 3  (completeness, parallel)        WS5 close gaps/drift  ||  WS6 complete dispatcher
   │
Phase 4  (product)                       WS7 hipDNN E2E hardening
   │
Phase 5  (proof)                         WS8 differential soak → default flip
```

**Dependency rules.**
- WS0 before WS3 (migrate into the final tree once).
- WS1 + WS2 before any risky refactor (WS3/WS5/WS6) — the spine must exist to gate them.
- WS4 needs WS1 (serialization) + an early WS3 C++ build.
- WS7 needs WS5 (gaps closed) + WS6 (dispatcher) for the families it exercises.
- WS8 is last; its clock resets on any real divergence.

**Parallel-agent suitability.** WS2 (by layer/family), WS3 instances (by family), WS5 (by gap), WS6 (by family), WS7 (by engine) are all fan-out friendly. The serial cores are: WS1 serializer design, WS3 primitives+error-model+namespaces, WS4 binding scaffold. Recommend orchestrating each fan-out phase as a batch of parallel agents with the parity harness as the shared acceptance gate.

---

## 11. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Numeric text-formatting drift during C++ migration | high | Keep the `vsnprintf` core (R3.3); L3 gate every module; two-step migration (R3.2) |
| SSA-id drift recurs | high | IR serialization carries explicit ids (seam a); L2 IR-diff catches it upstream of `.ll` |
| GCC arg-eval-order UB | medium | UBSan in test build (T3.5); sequencing audit (T3.6); warnings-as-errors |
| Dual instance maintenance burden long-term | medium | Accepted trade per owner; differential harness makes it safe; seam (b) reachable later via existing bindings |
| GPU runner availability for L6 | medium | GPU CI cluster scheduling; nightly cadence; CPU lanes carry per-PR load |
| Scope is very large | high | Strict phasing; each WS independently shippable; fan-out with a shared gate |
| pybind surface (225 methods) churn | medium | Generate bindings mechanically; bind the stable IR/lower entry points, not every internal |
| Provider ABI break during WS3 | medium | `extern "C"` shims (R3.4) keep the ABI stable until WS7 |
| Golden churn hides real regressions | medium | Blessing requires explicit review; L6 numeric independently validates blessed changes |

---

## 12. Success / exit criteria

- **G1** flat dot-names gone; subfolder tree; history preserved; `.ll` unchanged.
- **G2/G3** all six test layers green; every example reproduces its golden via one command.
- **G4** C++20, `-Werror`, ASan/UBSan clean; whole-program link (no duplicate symbols).
- **G5** `python`/`cpp`/`both` all functional; `both` raises actionable diffs.
- **G6** gap inventory zero for Python-supported families; numeric parity within tolerance on all arches.
- **G7** dispatcher complete across families; `dispatch_parity` green; `rdna_wmma` bug fixed with a regression test.
- **G8** hipDNN E2E green across ops/dtypes/layouts; sanitizer/leak clean.
- **G9** 14 consecutive green nightly soak runs; runtime default flipped to C++; both layers mirrored with permanent differential CI.

---

## 13. Decisions & open questions

**Resolved (2026-06-19):**
- **D1 (was Q1) — IR serialization format: a separate explicit machine format `ck.dsl.ir/v1`.** `print_ir` stays human-only; a dedicated versioned, round-trippable format is the interchange artifact. Two formats to maintain, but a clean machine contract independent of the human view. (WS1.T1.2/T1.3.)
- **D2 (was Q3) — Soak exit bar: 14 consecutive green nightly full-matrix runs (run-count based), regardless of calendar time.** (WS8.T8.3.)

**Resolved (2026-06-19):**
- **D3 (was Q2) — Collapse the LOWERER via the IR seam; keep instance BUILDERS dual as a flex frontend.** `ck.dsl.ir/v1` becomes the runtime interchange: one lowerer of record (C++) consumes serialized IR. Baked/shipped instances go Python-build-IR → serialize → single-C-lowerer (no C builder → the arg-eval-class drift vanishes for them). Instance builders stay **dual and selectable per-instance** (Python authoring + C++ runtime-JIT) — NOT collapsed to one. Byte-identity is required only for the **flex set** (instances needing a C builder for runtime arbitrary-shape JIT), and there the gate is IR-level, not `.ll`. This shrinks the drift tax without losing either frontend. Tracked as **WS12** (the §WS12 waves A–D).

**Open:**
- Q4. RDNA datalayout/triple: land the per-arch backend in WS5.2, or carve a dedicated WS given it unblocks all WMMA families?
- Q5. Heuristic models: retrain per family on fresh sweeps, or port the existing trained models and validate? (Affects WS6.4 cost.)

---

## Appendix A — Folder reorg map (canonical)

The authoritative old→new path table (220 rows) is generated from the dot-names by the WS0 script. Pattern summary (faithful 1:1 with the Python package — no family grouping):
- `helper_rocke.<pkg>.<mod>.c` → `src/<pkg-as-dirs>/<mod>.c` (dot-to-slash; e.g. `helper_rocke.helpers.atoms.c` → `src/helpers/atoms.c`; `helper_rocke.instances.common.fuse.c` → `src/instances/common/fuse.c`).
- `instance_<name>.c` → `src/instances/common/<name>.c`; `instance_<arch>_<name>.c` → `src/instances/<arch>/<name>.c` (arch ∈ gfx942/gfx950/gfx1151/gfx1201). C part-files (the port split single Python modules) cluster by name prefix within `common/`.
- Same-base `instance_X.c` + `helper_…instances.common.X.c` collisions (only `gemm_multi_d`, `img2col`) → the helper-origin TU gets `<name>_helpers.c`.
- `lower_llvm_<bucket>.c` → `src/core/lower_llvm/<bucket>.c`; `lower_hip_*` → `src/core/lower_hip/`; `lower_cktile.c` → `src/core/lower_cktile.c`.
- `ir_*.c` → `src/core/ir/`; `arch_target_*` + `helper_rocke.core.arch.c` → `src/core/arch/`; `isa_backend.c` → `src/core/isa/backend.c`; `passes.c` → `src/core/passes.c`; `arena.c`/`strbuf.c` → `src/support/`.
- Headers in `include/rocke/` stay FLAT in WS0 (not dot-named; all includes `rocke/`-prefixed). Header foldering deferred to WS3 with namespaces.

## Appendix B — Definition of Done (per instance)

An instance is "done" when, for every config in its enumeration and every relevant arch:
1. L1 verifier passes on both backends with identical verdicts.
2. L2 serialized IR is canonical-identical Python↔C++.
3. L3 `.ll` is `sha256`-identical (or identical rejection).
4. L5 golden recorded.
5. L6 numeric: each backend within tolerance vs reference, cross-backend bit-identical, on each supported arch.
6. (If on the provider path) WS7 E2E case green.

## Appendix C — Glossary
- **Seam (a) / IR seam**: the IR is the durable interchange artifact; a single lowerer of record consumes it.
- **Seam (b) / authoring seam**: one C++ engine; Python is a pybind veneer.
- **Veneer**: the unchanged Python authoring API forwarding to a selectable backend.
- **Differential mode (`both`)**: run both backends, assert equality on IR/`.ll`/numeric.
- **Golden / blessed**: a recorded known-good digest; a change is "blessed" when intentionally updated under review.
