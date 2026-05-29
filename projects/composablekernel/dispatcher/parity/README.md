<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Tile Engine ↔ Dispatcher parity

This directory proves the **dispatcher** reproduces **Tile Engine** for GEMM:
the same config produces the same kernel, the same registry key offline and at
runtime, the same numbers, and (within tolerance) the same throughput.

It is split so that everything provable *without* a GPU is provable here, and the
GPU-only half is staged and ready to run unchanged on a GPU node.

```
Tile Engine config JSON
        │
        ▼
 te_to_dispatcher.py ───────────────► dispatcher config objects  (a)
        │                                   │
        │                                   ├── identifier.py        (Python encode_identifier)
        │                                   └── cpp_identifier_oracle (C++ KernelKey::encode_identifier)
        │                                            │
        │                                   check_identifier_parity.py  (b)  ── g++ only, no GPU
        │
        ├── drive_codegen.py ──► unified_gemm_codegen.py ──► gemm_<id>.hpp   (c)
        │                                                         │
        │                                   harness.cpp + build_harness.sh   (d)  ── hipcc; run needs GPU
        │
        └── check_parity.py ───────────────────────────────────────────────► (e)(f)
              stage 1 identifier  (always)        — CPU
              stage 2 numerical   (GPU-gated)     — dispatcher verify [+ TE verify]
              stage 3 performance (GPU-gated)     — dispatcher TFLOP/s vs TE TFLOP/s
```

## What runs where

| Stage | Needs | This box (CPU-only) |
|------|-------|---------------------|
| Translate (a) | python3 | ✅ runs |
| Identifier parity (b) | python3 + g++ | ✅ runs (283968/283968 match on full config) |
| Drive codegen (c) | python3 | ✅ runs (emits the header) |
| Build harness (d) | hipcc + CK include tree | ✅ **builds** (running needs a GPU) |
| Numerical parity (f) | GPU (+ TE build for cross-check) | ⛔ GPU-gated → SKIPPED here |
| Performance parity (f) | GPU + TE build | ⛔ GPU-gated → SKIPPED here |

No `cmake` and no GPU on this box, so the orchestrator gates 2–3 to **SKIPPED**
(not FAILED) and exits 0 when the identifier stage passes.

## The two kinds of name (important)

* **Registry identifier** — `encode_identifier()`'s canonical key, used for
  dispatch lookup. Scheduler `default` → `auto` here. This is what stage 1 proves
  matches between Python (codegen-side) and C++ (runtime-side).
* **Kernel/file name** — built from the *raw* TE trait strings
  (`compv3`/`intrawave`/`default`), used to name the generated header
  `gemm_<name>.hpp` and the TE executable `benchmark_gemm_universal_<name>`.

They coincide for the `fp16_rcr…intrawave` example but diverge whenever a TE
string maps to a different canonical form (e.g. `default`→`auto`).
`check_parity.py:te_kernel_name()` builds the file-name form;
`identifier.py:encode_identifier()` builds the registry form.

## Quick start

```bash
# 1. Identifier parity over every config in a TE JSON (CPU-only, fast).
python check_identifier_parity.py configs/single_fp16_rcr.json --verbose

# 2. Generate one kernel header for a single config.
python drive_codegen.py configs/single_fp16_rcr.json --index 0

# 3. Build the single-kernel harness against that header (hipcc).
./build_harness.sh        # auto-picks the lone generated gemm_*.hpp

# 4. Full orchestration. On this box stages 2-3 SKIP; on a GPU node they run.
python check_parity.py configs/single_fp16_rcr.json                 # CPU: stage 1 only
python check_parity.py configs/single_fp16_rcr.json --dry-run       # print full plan
```

### On a GPU node

```bash
# Dispatcher-only numerical + performance:
python check_parity.py configs/single_fp16_rcr.json \
    --sizes 512x512x512,1024x1024x1024,2048x2048x2048 --arch gfx942

# Dispatcher vs Tile Engine (numerical first, then performance within tolerance):
python check_parity.py configs/single_fp16_rcr.json \
    --te-build-dir /path/to/tile_engine/build \
    --perf-tol 0.10
```

`--te-build-dir` is searched recursively for `benchmark_gemm_universal_<name>`.
The TE benchmark writes `latency(ms),tflops,bandwidth` to a CSV (only when it
verifies), which the orchestrator parses for both the numerical pass signal and
the performance baseline.

## Parity definitions

* **Identifier**: Python and C++ `encode_identifier()` agree byte-for-byte for
  every translated config ⇒ the offline registry key equals the runtime key, so
  dispatch lookups cannot silently miss.
* **Numerical**: the dispatcher harness `PASSED` against its CPU fp32 reference
  (tol `1e-2·√K`); with `--te-build-dir`, the TE benchmark must also verify for
  the same `MxNxK`. Either tool emitting `SKIPPED`/unsupported is a skip, not a
  failure. Numerical is adjudicated **before** performance.
* **Performance**: `|disp_TFLOPs − te_TFLOPs| / te_TFLOPs ≤ --perf-tol`
  (default 10%). The dispatcher harness reports GFLOP/s; the orchestrator
  converts to TFLOP/s to match TE's units.

## Files

| File | Deliverable | Role |
|------|-------------|------|
| `te_to_dispatcher.py` | (a) | TE JSON → dispatcher config dicts (`_te`/`signature`/`algorithm`) |
| `identifier.py` | (b) | Python `encode_identifier()` oracle |
| `cpp_identifier_oracle.cpp` | (b) | C++ `KernelKey::encode_identifier()` oracle (batched stdin) |
| `check_identifier_parity.py` | (b) | diff the two oracles over every config |
| `drive_codegen.py` | (c) | drive `unified_gemm_codegen.py` for ONE config |
| `harness.cpp` | (d) | single-kernel runner via `CK_TILE_SINGLE_KERNEL_INCLUDE` |
| `build_harness.sh` | (d) | hipcc build of the harness against a generated header |
| `check_parity.py` | (e)(f) | the 3-stage orchestrator above |
| `configs/single_fp16_rcr.json` | — | example single fp16 rcr config |
