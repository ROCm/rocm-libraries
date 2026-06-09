# HSTU attention (Composable Kernel tile_engine)

Forward HSTU jagged attention via the **in-process dispatcher** (`libdispatcher_hstu_*.so`), not the legacy subprocess example binary.

## New vs old CK (`hstu_attention_fwd` branch)

| Capability | Old CK example | This integration |
|------------|----------------|------------------|
| Invocation | `tile_example_hstu_attention` subprocess | `hstu_utils.HstuRunner` ctypes |
| Kernel pick | Heuristic `get_hstu_attention_fwd_mtile` + `shall_use_splitkv` | Exhaustive sweep over compiled configs; `--best` = max TFLOPS |
| Instances | ~192 jagged/batched/group × feature cross-product | **16** trimmed jagged fwd (fp16/bf16 × causal × maxk 64/128 × mtile 64/128) |
| Modes trimmed | — | batched, group, softmax, bias, dropout, hdim 96/256 |
| Backward | Not on old branch | Not included (fwd-only) |

Split-KV is now built per-kernel: `use_splitkv: true` configs are compiled
**without** `-DHSTU_COMPILE_NO_SPLITKV=1` (see `hstu_compile_flags(...,
use_splitkv=...)` in `dispatcher/python/hstu_utils.py`). Constraint: split-KV
requires `mtile=64` (asserted in the dispatch header). For other configs the
flag is still defined, keeping the legacy non-splitkv compile path.

## Layout

- Example kernels: `example/ck_tile/53_hstu_attention/`
- Codegen: `dispatcher/codegen/hstu/` (`instance_gen.py`, `generate_fallback.py`)
- Dispatcher: `dispatcher/include/ck_tile/dispatcher/hstu_*.hpp`, `dispatcher/bindings/ctypes/hstu_ctypes_lib.cpp`, `dispatcher/python/hstu_utils.py`
- Benchmark: `tile_engine/ops/hstu/hstu_benchmark.py`
- Configs:
  - `configs/sweep_trimmed.json` — kernel config grid (like FMHA trait_config)
  - `configs/sweep_d64.json` — deployment d=64 exhaustive bf16 sweep
    (9 kernels: causal × `max_k ∈ {64, 96, 128}` × `mtile ∈ {64, 128}` ×
    `splitkv ∈ {false, true}`; splitkv requires `mtile=64`). `max_k > hdim_qk`
    enables the padded-`max_k` path so the dispatcher can compare native and
    padded tiles on the same shape.
  - `configs/fwd.json` — problems, masks, smoke_problems only
  - `configs/deployment_progression.json` — deployment shapes small→large (one index at a time)
  - `configs/deployment_hstu_mask.json` — full N=16384 table (4 cells), hstu mask, per-problem targets
  - `configs/deployment_reference_ms.json` — CK AMD / Triton genrec ms sidecar for `--reference`

## Build

```bash
cd /path/to/composablekernel
git checkout feature/hstu-dispatcher

# Dispatcher static lib (required for JIT link)
cmake -S dispatcher -B dispatcher/build -DGPU_TARGETS=gfx950
make -C dispatcher/build ck_tile_dispatcher -j

# Prebuilt single-lib fallback (optional)
make -C dispatcher/build hstu_python_libs -j
# -> dispatcher/build/examples/libdispatcher_hstu_lib.so
```

Note: `make python_libs` also builds GEMM/Conv/FMHA; FMHA fallback codegen may fail on some ROCm/clang combos. Use `hstu_python_libs` for HSTU-only prebuilt lib.

## Run benchmark (FMHA-style two-phase sweep)

```bash
export PYTHONPATH=/path/to/composablekernel/dispatcher/python:$PYTHONPATH

# Smoke: compile trimmed sweep + benchmark smoke problems
python tile_engine/ops/hstu/hstu_benchmark.py --smoke

# Prod fwd.json with best kernel per (shape, mask, dtype)
python tile_engine/ops/hstu/hstu_benchmark.py \
  --config tile_engine/ops/hstu/configs/sweep_trimmed.json \
  --problems tile_engine/ops/hstu/configs/fwd.json \
  --best --csv /tmp/hstu_best.csv

# Compile-only (no GPU timing)
python tile_engine/ops/hstu/hstu_benchmark.py --smoke --compile-only

# Prebuilt single lib (legacy mtile 0/64/128 env hack)
python tile_engine/ops/hstu/hstu_benchmark.py --lib dispatcher/build/examples/libdispatcher_hstu_lib.so --best
```

### List expanded kernel configs

```bash
python dispatcher/codegen/hstu/instance_gen.py tile_engine/ops/hstu/configs/sweep_trimmed.json --list
```

## Deployment reproduction (d=64, N=16384)

Matches the deployment comparison slide: **hstu mask**, fixed targets (train **10**, inference **300** per batch), `tflops_genrec` for mvonstra alignment. Run **one problem at a time** on GPU (small→large).

| Index | `problem_id` | B | H | N (UIH) | target |
|------|----------------|---|----|---------|--------|
| 0 | `step0_smoke` | 128 | 4 | 1024 | 10 |
| 1 | `step1_proxy_b128_n4096` | 128 | 4 | 4096 | 10 |
| 2 | `step2_infer_proxy_b120_n4096` | 120 | 4 | 4096 | 300 |
| 3 | `step3_train_b1024_n16384_h4` | 1024 | 4 | 16384 | 10 |
| 4 | `step4_train_b1024_n16384_h8` | 1024 | 8 | 16384 | 10 |
| 5 | `step5_infer_b120_n16384_h4` | 120 | 4 | 16384 | 300 |
| 6 | `step6_infer_b120_n16384_h8` | 120 | 8 | 16384 | 300 |

Reference ms (indices 3–6): see `configs/deployment_reference_ms.json`.

```bash
export CK=/workspaces/rocm-libraries/projects/composablekernel
export PYTHONPATH=$CK/dispatcher/python:$PYTHONPATH
cd $CK

# Compile d=64 sweep once (9 bf16 kernels: causal × maxk {64,96,128} × mtile {64,128}, splitkv mtile=64)
python tile_engine/ops/hstu/hstu_benchmark.py \
  --config tile_engine/ops/hstu/configs/sweep_d64.json \
  --problems tile_engine/ops/hstu/configs/deployment_progression.json \
  --problem-index 0 --compile-only

# Step by step on GPU (best kernel + CK table comparison)
for i in 0 1 2 3 4 5 6; do
  python tile_engine/ops/hstu/hstu_benchmark.py \
    --config tile_engine/ops/hstu/configs/sweep_d64.json \
    --problems tile_engine/ops/hstu/configs/deployment_progression.json \
    --problem-index $i --best
done

# Or by label
python tile_engine/ops/hstu/hstu_benchmark.py \
  --config tile_engine/ops/hstu/configs/sweep_d64.json \
  --problems tile_engine/ops/hstu/configs/deployment_progression.json \
  --only-problem step3_train_b1024_n16384_h4 --best

# Full deployment table only (4 shapes, indices 0–3 in deployment_hstu_mask.json)
python tile_engine/ops/hstu/hstu_benchmark.py \
  --config tile_engine/ops/hstu/configs/sweep_d64.json \
  --problems tile_engine/ops/hstu/configs/deployment_hstu_mask.json \
  --problem-index 0 --best
```

Fixed targets: each problem sets `target_size` and `num_targets_fixed: true` (every batch gets exactly that many candidate rows). This matches mvonstra `bench_hstu.py --target-size-fixed`; the benchmark already used a constant per-batch target list (not uniform random).

## TFLOPS columns

- **`ms`** — primary latency column in benchmark output
- `tflops_genrec` — jagged sum s_i² FLOPs aligned with mvonstra `recsys_harness/common.hstu_flops` (use for cross-harness comparison)
- `tflops` — dense upper-bound with optional causal 0.5 factor (legacy CK column)

## Harness bridge (mvonstra)

```bash
export CK_HSTU_DISPATCHER_LIB=/path/to/libdispatcher_hstu_lib.so
export PYTHONPATH=/path/to/composablekernel/dispatcher/python:$PYTHONPATH
```

Until `bench_hstu.py` gains a native provider, use `hstu_benchmark.py` for CK numbers with `--best`.

## Mask / shape conventions

Matches `recsys_harness/sweep_hstu.py` mask grids: causal, hstu (`target_size`), semi_local (`max_attn_len`), full_semi_local (local + contextual + targets). Shapes follow prod grid in `fwd.json`: `B∈{256,512}`, `N∈{4096,8192}`, `H∈{4,8}`, `D=128`.
