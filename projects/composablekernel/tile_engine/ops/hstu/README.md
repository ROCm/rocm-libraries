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
- Configs — exactly two sweep grids (`--config`) plus the problem file:
  - `configs/sweep_fast.json` — **default `--config`**. Small d=64 grid for quick
    iteration (11 kernels, all compile, no `--filter`): 9 base-tile kernels
    (`max_k ∈ {64, 96, 128}` × `mtile ∈ {64, 128}` × `splitkv ∈ {false, true}`,
    split-KV needs `mtile=64`, default WarpK) — including the legacy-heuristic
    baseline — **plus** a light tile-shape family pinned to the proven d=64
    winner `sequence<192,32,32,64,32,64>` (`warp_k=32`, `max_k=64`) so the fast
    sweep directly contains the fastest known tile kernel. Works for any d=64
    shape.
  - `configs/sweep_exhaustive.json` — full d=64 block-tile-shape sweep
    (`sequence<kM0,kN0,kN0Sub,kN1,kK1>` grid, `warp_k=32`, ~90 kernels). Use to
    find the best tile shape per shape.
  - `configs/fwd.json` — problems, masks, smoke_problems only (default `--problems`)

Only the axes the codegen actually consumes are kept in the sweep configs
(`data_type, use_causal, max_k, mtile, use_splitkv, km0/kn0/kn0sub/kn1/kk1,
warp_k`); silently-ignored knobs (pipeline/agpr/pingpong/trload/occupancy/
num_splits) are not present.

Deployment/single shapes no longer need a problems JSON — pass the shape
directly with the inline flags `--batch / --num-head / --seqlen / --hdim /
--target-size` (see *Deployment reproduction* below).

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

# Smoke: compile the default fast sweep + benchmark smoke problems
python tile_engine/ops/hstu/hstu_benchmark.py --smoke

# Prod fwd.json with best kernel per (shape, mask, dtype)
# (--config defaults to sweep_fast.json; pass --config sweep_exhaustive.json for the full grid)
python tile_engine/ops/hstu/hstu_benchmark.py \
  --problems tile_engine/ops/hstu/configs/fwd.json \
  --best --csv /tmp/hstu_best.csv

# Compile-only (no GPU timing)
python tile_engine/ops/hstu/hstu_benchmark.py --smoke --compile-only

# Prebuilt single lib (legacy mtile 0/64/128 env hack)
python tile_engine/ops/hstu/hstu_benchmark.py --lib dispatcher/build/examples/libdispatcher_hstu_lib.so --best
```

### List expanded kernel configs

```bash
python dispatcher/codegen/hstu/instance_gen.py tile_engine/ops/hstu/configs/sweep_fast.json --list
```

## Deployment reproduction (d=64, N=16384)

Matches the deployment comparison slide: **hstu mask**, fixed targets (train **10**, inference **300** per batch), `tflops_genrec` for mvonstra alignment. Pass the shape **directly on the CLI** with the inline flags — no problems JSON needed. The benchmark synthesizes a single problem (`problem_id` = `b{batch}_h{num_head}_n{seqlen}_d{hdim}`), hstu mask, `num_targets_fixed: true`, `target_size` from `--target-size`.

| B | H | N (UIH) | target | inline flags |
|---|----|---------|--------|--------------|
| 128 | 4 | 4096 | 10 | `--batch 128 --num-head 4 --seqlen 4096 --target-size 10` |
| 120 | 4 | 4096 | 300 | `--batch 120 --num-head 4 --seqlen 4096 --target-size 300` |
| 1024 | 4 | 16384 | 10 | `--batch 1024 --num-head 4 --seqlen 16384 --target-size 10` |
| 1024 | 8 | 16384 | 10 | `--batch 1024 --num-head 8 --seqlen 16384 --target-size 10` |
| 120 | 4 | 16384 | 300 | `--batch 120 --num-head 4 --seqlen 16384 --target-size 300` |
| 120 | 8 | 16384 | 300 | `--batch 120 --num-head 8 --seqlen 16384 --target-size 300` |

(`--hdim` defaults to 64 and is used for both `hdim_qk` and `hdim_v`; causal is on by default — use `--no-causal` to disable.)

The benchmark reports best-vs-heur only (the swept fastest kernel vs the legacy
dispatch heuristic's kernel); it does not load or print any external reference
(CK/Triton) numbers — just pass the problem size.

```bash
export CK=/workspaces/rocm-libraries/projects/composablekernel
export PYTHONPATH=$CK/dispatcher/python:$PYTHONPATH
cd $CK

# Compile the fast sweep once (11 bf16 kernels: 9 base-tile + 2 winner tile-shape variants)
# --config defaults to sweep_fast.json, so it can be omitted.
python tile_engine/ops/hstu/hstu_benchmark.py \
  --batch 128 --num-head 4 --seqlen 4096 --target-size 10 --compile-only

# A single deployment shape on GPU (best kernel + heur-vs-best comparison)
python tile_engine/ops/hstu/hstu_benchmark.py \
  --batch 120 --num-head 4 --seqlen 16384 --hdim 64 --target-size 300 --best

# Walk the deployment shapes small→large (fast sweep, default --config)
python tile_engine/ops/hstu/hstu_benchmark.py --batch 128  --num-head 4 --seqlen 4096  --target-size 10  --best
python tile_engine/ops/hstu/hstu_benchmark.py --batch 1024 --num-head 4 --seqlen 16384 --target-size 10  --best
python tile_engine/ops/hstu/hstu_benchmark.py --batch 120  --num-head 4 --seqlen 16384 --target-size 300 --best

# Exhaustive tile-shape search for one shape (~90 kernels — slower)
python tile_engine/ops/hstu/hstu_benchmark.py \
  --config tile_engine/ops/hstu/configs/sweep_exhaustive.json \
  --batch 120 --num-head 4 --seqlen 16384 --hdim 64 --target-size 300 --best
```

Fixed targets: the inline problem sets `target_size` (from `--target-size`) with `num_targets_fixed: true` (every batch gets exactly that many candidate rows). This matches mvonstra `bench_hstu.py --target-size-fixed`; the benchmark already used a constant per-batch target list (not uniform random).

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
