# CK DSL fused-MoE parity & benchmark harness

This folder hosts the cross-backend parity + benchmark scripts for the
CK DSL `FusedMoeForward` pipeline. It is the canonical performance
harness for the CK DSL MoE work.

Two scripts live here:

| Script | Role |
|---|---|
| [`fused_moe_e2e_perf.py`](fused_moe_e2e_perf.py) | End-to-end fused-MoE forward perf comparison: CK DSL vs torch eager vs Triton vs CK Tile C++. The four-way perf harness. |
| [`tune_gate_up_silu.py`](tune_gate_up_silu.py) | Tile-shape and activation-barrier sweep for the experimental fused gate / up / SiLU GEMM. Picks the best fused-MoE GEMM variant for a given decode / small-batch shape. |

Both scripts share input generation, scenario definitions, and the
torch-eager reference from `fused_moe_e2e_perf.py`.

---

## End-to-end perf: `fused_moe_e2e_perf.py`

Drives `ck_dsl.instances.FusedMoeForward` and three reference
implementations on a set of MoE forward shapes, reports per-backend
latency (ms), correctness vs the torch reference (max abs / mean abs
/ rel error), and the resulting speedup ratios.

### Backends

1. **CK DSL** — `ck_dsl.instances.FusedMoeForward`. Composes
   `topk_softmax` (router) + `MoeSortingLauncher` (sort 3-chain) +
   `FusedMoeLauncher` (gather / silu_mul / topk_reduce streaming
   kernels) + `GroupedGemmLauncher` (per-expert gate / up / down
   GEMMs) into one pipeline driven by chained
   `ck_dsl.runtime.launcher.launch_kernel` calls. In static-offset
   mode the forward is HIP-graph-capturable; the harness uses graph
   replay where possible to keep the comparison focused on kernel
   work rather than Python launch overhead.
2. **Torch eager** — vectorised per-expert mask + scatter (faster
   than a naive per-token-per-topk Python loop, but still pure
   torch ops with no fusion). Gold-standard correctness oracle.
3. **Triton** — purpose-written single-kernel fused-MoE that mirrors
   the CK DSL pipeline's algorithmic shape (per `(token, k)` program:
   gate / up / SiLU mul / down GEMMs against per-expert weights,
   atomic-add into an `f32 Y` accumulator). This is **not** a tuned
   production Triton kernel — it's a fair-baseline reference. AITER's
   tuned `e2e_moe` Triton kernel currently crashes with a
   memory-access fault on MI355X (gfx950); the script falls back to
   this purpose-written kernel until that is fixed upstream.
4. **CK Tile C++** — `build/bin/tile_example_fused_moe` invoked via
   `subprocess` with matching `-t / -e / -k / -h / -i / -prec_*`
   arguments. The C++ binary uses its own random inputs (no public
   hook to feed external tensors), so the row reports a perf-only
   number; correctness is validated against torch eager on the
   Python side only.

### Scenarios

Five default scenarios (`default_scenarios()` in the script). All use
`f16` activations unless overridden by `--dtype bf16`.

| Scenario | tokens | experts | topk | hidden | intermediate | shape class |
|---|---:|---:|---:|---:|---:|---|
| `decode_T1_E8_K2_H4096_I7168` | 1 | 8 | 2 | 4096 | 7168 | inference decode (one token, large experts) |
| `decode_T8_E8_K2_H4096_I7168` | 8 | 8 | 2 | 4096 | 7168 | small-batch decode |
| `batch32_E8_K2_H4096_I7168` | 32 | 8 | 2 | 4096 | 7168 | mid-batch |
| `prefill_T128_E8_K2_H4096_I7168` | 128 | 8 | 2 | 4096 | 7168 | prefill (many tokens) |
| `small_T32_E4_K2_H128_I256` | 32 | 4 | 2 | 128 | 256 | small validation shape (matches parity tests) |

Pass `--scenario NAME` (repeatable) to restrict to a subset. Pass
`--dtype bf16` to switch every scenario to bf16.

Scenario constraints driven by the CK DSL pipeline (and matching what
the standard MoE workloads look like in production):

- `hidden` and `intermediate` must be multiples of the GEMM tile dims
  (default `tile_n=128`, `tile_k=64`) and the streaming kernel
  `block_size` (64 by default).
- `experts ≤ sort_block_size` (= 64) for the single-block scan kernel.
- `topk ≤ experts`.

### Running

```bash
cd <composablekernel-checkout>
export AITER_PATH=<aiter-checkout>
PYTHONPATH="python:${AITER_PATH}" python \
  python/ck_dsl/examples/moe/fused_moe_e2e_perf.py \
  --attempts 10 --warmup 3 \
  --report /tmp/moe_perf.json
```

Flags:

| Flag | Default | Notes |
|---|---|---|
| `--scenario NAME` (repeatable) | all | restrict to the named scenarios |
| `--attempts N` | `10` | timed iterations per backend; reported number is `elapsed_ms / N` from a single HIP-event pair recorded on torch's current stream |
| `--warmup N` | `3` | untimed warmup iterations |
| `--dtype {f16, bf16}` | `f16` | activation dtype for all backends |
| `--skip-ck-dsl` | off | skip the CK DSL backend |
| `--skip-torch` | off | skip the torch eager reference |
| `--skip-aiter` | off | skip the Triton baseline (the flag name is historical — the actual Triton call is the purpose-written fallback documented above) |
| `--skip-cktile` | off | skip the CK Tile C++ binary |
| `--report PATH` | none | dump every measurement to JSON |

The script returns a non-zero exit code if any scenario sees a CK DSL
correctness regression (`max_abs > 0.5`). AITER / CK Tile failures are
treated as environment issues and do not gate.

### Methodology

Every row in the tables below is the **mean per-launch wall time over
10 timed iterations** after 3 untimed warmup launches, measured with
HIP events recorded on torch's current stream. Both backends use
the same timer and the same stream, so the numbers are directly
comparable. Concretely, `time_callable_ms` does:

1. 3 untimed warmup launches.
2. `torch.cuda.synchronize()` to drain.
3. Record a start HIP event on `torch.cuda.current_stream()`.
4. 10 timed launches on that same stream.
5. Record an end HIP event, synchronize on it, report `elapsed_ms / 10`.

The CK DSL forward in static-offset mode is HIP-graph-captured (one
capture, then the timed loop replays the graph); this is the realistic
inference-benchmark mode and the only way the DSL pipeline can match
the per-launch overhead of a tuned C++ reference.

Between scenarios the harness runs an `_isolate_lane()` step
(`torch.cuda.synchronize` + `synchronize_and_release` +
`gc.collect`) to drop any retained args / workspace tensors / module
caches before the next scenario allocates its inputs. This avoids a
known ROCm 7.2 / torch 2.12 edge case where the recycled pool storage
is still in flight on the GPU command processor.

### Latest results (MI355X, gfx950, ROCm 7.2, torch 2.12)

`--attempts 10 --warmup 3 --dtype f16`. Speedup is `(other) / (CK DSL)`,
so values > 1 mean CK DSL is faster.

| Scenario | ck_dsl | torch | triton | ck_tile_cpp | ck_dsl vs torch | ck_dsl vs triton | ck_dsl vs ck_tile_cpp |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode_T1_E8_K2_H4096_I7168     | 0.443 ms | 0.920 ms | 14.912 ms | 0.124 ms | **2.08×** | **33.67×** | 0.28× |
| decode_T8_E8_K2_H4096_I7168     | 0.463 ms | 1.846 ms | 19.059 ms | (skip)   | **3.99×** | **41.18×** | — |
| batch32_E8_K2_H4096_I7168       | 0.537 ms | 2.292 ms | 20.145 ms | (skip)   | **4.27×** | **37.52×** | — |
| prefill_T128_E8_K2_H4096_I7168  | 0.887 ms | 2.426 ms | 28.878 ms | (skip)   | **2.73×** | **32.54×** | — |
| small_T32_E4_K2_H128_I256       | 0.039 ms | 0.815 ms | 0.045 ms  | 0.016 ms | **21.00×** | **1.16×** | 0.42× |

Reading the table:

- **CK DSL beats torch eager** on every scenario (geomean ≈ 4.5×;
  the torch reference is a vectorised per-expert mask + scatter, not
  a fused kernel).
- **CK DSL beats the purpose-written Triton baseline** by a wide
  margin on the four large-expert scenarios (32-41× — the Triton
  kernel is single-program-per-`(token, topk-slot)` with atomic-add
  into a shared `Y`; not tuned for these shapes), and edges it on the
  small validation shape (1.16×).
- **CK Tile C++ wins** the two scenarios where it doesn't crash. On
  this build, the binary segfaults with a memory-access fault on the
  three larger E=8 / H=4096 / I=7168 scenarios (decode_T8, batch32,
  prefill_T128) — a known issue with the C++ binary that does not
  affect the CK DSL row.

`max_abs(CK vs torch ref)` for the CK DSL column is below 5e-4 on
every scenario, well within the f16-with-fp32-accumulator tolerance
band for an MoE forward over O(7168) reduction terms.

`max_abs(Triton vs torch ref)` is similarly below 5e-4, confirming
the purpose-written Triton baseline is correct (it is just slow on
these shapes).

### Caveats / known issues

- **AITER `e2e_moe` Triton kernel crashes** with a memory-access
  fault on MI355X / gfx950 regardless of torch / ROCm version (likely
  an AITER kernel-arch mismatch). The script's `run_triton` path uses
  a purpose-written single-kernel fallback instead. Re-enable AITER
  once the gfx950 issue is fixed upstream.
- **`tile_example_fused_moe` C++ binary crashes** with a
  memory-access fault on E=8 / H=4096 / I=7168 shapes; the harness
  treats this as a skip (the row prints `(skip)` and the report's
  `ck_tile_cpp` field is `null`). The CK DSL row is unaffected.
- The CK DSL pipeline does **per-expert grouped-GEMM dispatch via a
  Python loop** with a small device-to-host copy of `Counts` and
  `Offsets` per dispatch; AITER's mega-kernel does in-kernel
  grouped-GEMM dispatch with no host roundtrip. This is the largest
  known overhead in the DSL path for shapes where per-expert GEMMs
  are small. HIP graph capture (used in static-offset mode) hides
  this on shapes where the routing is shape-stable.

### JSON report layout

Passing `--report PATH` writes a JSON list of per-scenario records:

```jsonc
[
  {
    "scenario": {
      "name": "decode_T1_E8_K2_H4096_I7168",
      "tokens": 1, "experts": 8, "topk": 2,
      "hidden": 4096, "intermediate": 7168,
      "dtype": "f16"
    },
    "results": {
      "ck_dsl":      { "backend": "ck_dsl",      "ok": true,  "ms": 0.443, "max_abs": 0.000488, "mean_abs": ..., "rel_max": ... },
      "torch_eager": { "backend": "torch_eager", "ok": true,  "ms": 0.920, "max_abs": 0.0,      "mean_abs": 0.0, "rel_max": 0.0 },
      "triton":      { "backend": "triton",      "ok": true,  "ms": 14.912, "max_abs": 0.000122, ... },
      "ck_tile_cpp": { "backend": "ck_tile_cpp", "ok": true,  "ms": 0.124, "max_abs": null, "mean_abs": null, "rel_max": null,
                       "note": "C++ binary, perf-only" }
    }
  },
  ...
]
```

---

## Tile / activation-barrier tuner: `tune_gate_up_silu.py`

This script tunes the experimental fused gate / up / SiLU MoE GEMM.
It compares **three activation-barrier strategies** across **five
MFMA tile candidates** for a given scenario.

### Three activation-barrier paths

1. **`packed`** — packed gate + up batched GEMM (`N = 2 × intermediate`)
   followed by a packed `silu_mul` post-pass. The current production
   default. One activation barrier (`silu_mul` after the GEMM).
2. **`dual`** — dual-B MFMA gate + up GEMM with the SiLU epilogue
   folded into the kernel. No separate `silu_mul` pass. Drives the
   `use_experimental_fused_gate_up_silu` spec flag.
3. **`interleaved`** — interleaved single-B MFMA gate + up GEMM with
   the SiLU epilogue folded into the kernel. The two GEMMs share a
   single B-operand load path. Drives the
   `use_experimental_interleaved_gate_up_silu` spec flag.

### Five tile candidates

Each candidate is a `TileSpec` from `instances/gemm_universal`:

| Name | tile (M×N×K) | warps (M×N) | atom |
|---|---|---:|---|
| `t16n128k64_w1x1_atom16` | 16×128×64 | 1×1 | 16×16×16 |
| `t16n256k64_w1x2_atom16` | 16×256×64 | 1×2 | 16×16×16 |
| `t32n128k64_w2x1_atom16` | 32×128×64 | 2×1 | 16×16×16 |
| `t32n256k64_w2x2_atom16` | 32×256×64 | 2×2 | 16×16×16 |
| `t32n128k64_w1x2_atom32` | 32×128×64 | 1×2 | 32×32×16 |

The grid is intentionally small: every candidate compiles a fresh
HSACO for the fused kernel and one for the batched down GEMM. The
focus is **static-offset shapes** (decode / small batch) where HIP
graph capture is valid and per-launch overhead is already amortized.

### Running

```bash
cd <composablekernel-checkout>
PYTHONPATH=python python \
  python/ck_dsl/examples/moe/tune_gate_up_silu.py \
  --scenario {small, decode1, decode8} \
  --attempts 10 --warmup 3
```

### Latest results (MI355X, gfx950, ROCm 7.2, torch 2.12)

`--attempts 10 --warmup 3`. Latency in ms (mean over 10 graph-replay
iterations); lower is better. The best per scenario is bolded.

#### Scenario `small_T32_E4_K2_H128_I256`

| Tile | packed | dual | interleaved |
|---|---:|---:|---:|
| `t16n128k64_w1x1_atom16` | 0.0469 | 0.0513 | 0.0448 |
| `t16n256k64_w1x2_atom16` | 0.0504 | 0.0571 | 0.0493 |
| `t32n128k64_w2x1_atom16` | 0.0488 | 0.0531 | 0.0477 |
| `t32n256k64_w2x2_atom16` | 0.0622 | 0.0690 | 0.0610 |
| `t32n128k64_w1x2_atom32` | 0.0391 | 0.0405 | **0.0383** |

#### Scenario `decode_T1_E8_K2_H4096_I7168`

| Tile | packed | dual | interleaved |
|---|---:|---:|---:|
| `t16n128k64_w1x1_atom16` | 0.6344 | 1.0609 | 0.6247 |
| `t16n256k64_w1x2_atom16` | 0.8072 | 1.2452 | 0.7251 |
| `t32n128k64_w2x1_atom16` | 0.9454 | 0.9313 | 0.9721 |
| `t32n256k64_w2x2_atom16` | 1.4887 | 1.2646 | 1.2431 |
| `t32n128k64_w1x2_atom32` | 0.4257 | 0.6249 | **0.4161** |

#### Scenario `decode_T8_E8_K2_H4096_I7168`

| Tile | packed | dual | interleaved |
|---|---:|---:|---:|
| `t16n128k64_w1x1_atom16` | 1.4840 | 1.0717 | 0.6347 |
| `t16n256k64_w1x2_atom16` | 0.8272 | 1.3995 | 0.7407 |
| `t32n128k64_w2x1_atom16` | 0.9639 | 0.9481 | 0.9633 |
| `t32n256k64_w2x2_atom16` | 1.2357 | 1.2772 | 1.3394 |
| `t32n128k64_w1x2_atom32` | 0.4412 | 0.4582 | **0.4225** |

### Findings

- **The 32×32 MFMA atom wins on every scenario.** The
  `t32n128k64_w1x2_atom32` tile is the best across all three
  scenarios, regardless of the activation-barrier path.
- **`interleaved` is the best path for the winning tile** on every
  scenario — it ties `packed` on small and beats `packed` / `dual`
  on both decode shapes.
- **The 16×16-atom tiles are dominated** on decode shapes — the
  best 16×16 row (`t16n128k64_w1x1_atom16` / `interleaved`) is
  ~50 % slower than the 32×32-atom winner on decode_T1 and
  ~50 % slower on decode_T8.
- **`packed` (the current production default) is competitive on
  small shapes** but loses to `interleaved` once the per-expert
  GEMMs grow beyond the small-validation regime.
- **Correctness** (`max_abs` vs torch reference) is identical
  across paths on each scenario: 1.9e-6 on small, 4.9e-4 on the two
  decode scenarios — well within f16-with-fp32-accumulator tolerance.

The recommended next step (when promoting `interleaved` past
experimental) is to wire the 32×32-atom + interleaved variant into
the `FusedMoeForward` selector for shapes where the per-expert GEMM
M dim is small enough that the gate / up activation barrier dominates.

---

## Reproducibility

```bash
# E2E perf
cd <composablekernel-checkout>
export AITER_PATH=<aiter-checkout>
PYTHONPATH="python:${AITER_PATH}" python \
  python/ck_dsl/examples/moe/fused_moe_e2e_perf.py \
  --attempts 10 --warmup 3 --report /tmp/moe_perf.json

# Tuner: small / decode1 / decode8
PYTHONPATH=python python \
  python/ck_dsl/examples/moe/tune_gate_up_silu.py \
  --scenario small --attempts 10 --warmup 3
PYTHONPATH=python python \
  python/ck_dsl/examples/moe/tune_gate_up_silu.py \
  --scenario decode1 --attempts 10 --warmup 3
PYTHONPATH=python python \
  python/ck_dsl/examples/moe/tune_gate_up_silu.py \
  --scenario decode8 --attempts 10 --warmup 3
```

`AITER_PATH` only needs to be set if you want the script's
`run_triton` path to import `aiter`-bundled Triton dependencies. The
purpose-written Triton kernel itself only needs `triton` to be
importable in the active environment.
