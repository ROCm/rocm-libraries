# HSTU attention (ck_tile example)

Ported from `composable_kernel` branch `hstu_attention_fwd` (`18_hstu_attention`).

This tree is wired as `53_hstu_attention` in the new repo (number 18 is `flatmm`).

## Trimmed build (harness default)

`generate_instances.py` emits **8** jagged forward instances:

- dtypes: fp16, bf16
- causal: has_causal, no_causal
- softmax: false, bias: false
- MaxK: 64, 128

Full old CK cross-product (batched/group/softmax/bias/dropout/hdim 96/256) can be restored by reverting `generate_instances.py` `headdims_fwd` / mode lists.

## New vs old CK

| Capability | Old CK (`18_hstu_attention`) | This integration |
|------------|------------------------------|------------------|
| Invocation | `tile_example_hstu_attention` subprocess | `hstu_utils.HstuRunner` ctypes (in-process) |
| Kernel pick | Heuristic `get_hstu_attention_fwd_mtile` + `shall_use_splitkv` | Benchmark sweep; `--best` = max TFLOPS |
| Instances | ~192 jagged/batched/group × feature cross-product | **8** jagged fwd (fp16/bf16 × causal × maxk 64/128) |
| Modes trimmed | — | batched, group, softmax, bias, dropout, hdim 96/256 |
| Backward | Not on old branch | Not included (fwd-only) |

Override heuristics for sweep (set before each run from ctypes):

- `HSTU_FORCE_MTILE=64|128` — force tile M
- `HSTU_FORCE_SPLITKV=1` / `HSTU_DISABLE_SPLITKV=1` — split-KV path (disabled in dispatcher build via `HSTU_COMPILE_NO_SPLITKV=1`)

## Build

Branch: `feature/hstu-dispatcher`

### Dispatcher shared library (recommended)

```bash
cd /path/to/composablekernel
git checkout feature/hstu-dispatcher

cmake -S dispatcher -B dispatcher/build -DGPU_TARGETS=gfx950 -DBUILD_DISPATCHER_EXAMPLES=ON
make -C dispatcher/build hstu_python_libs -j
# or: make -C dispatcher/build dispatcher_hstu_lib -j
```

Output: `dispatcher/build/examples/libdispatcher_hstu_lib.so`

Note: `make python_libs` also builds GEMM/Conv/FMHA; FMHA fallback codegen may fail on some ROCm/clang combos. Use `hstu_python_libs` for HSTU-only.

### Example binary (optional sanity)

```bash
cmake -S . -B build -DGPU_TARGETS=gfx950
cmake --build build --target tile_example_hstu_attention -j
```

Requires a ROCm GPU target matching `-DGPU_TARGETS`. Full root `cmake` may fail in minimal/CPU-only dev containers (unrelated CK library configure errors).

## Run benchmark (fastest-kernel policy)

```bash
export PYTHONPATH=/path/to/composablekernel/dispatcher/python:$PYTHONPATH

# Smoke (d=64 and d=128)
python tile_engine/ops/hstu/hstu_benchmark.py --smoke

# Prod-like subset; pick fastest kernel per (shape, mask)
python tile_engine/ops/hstu/hstu_benchmark.py configs/fwd.json --best --csv /tmp/hstu_best.csv

# Explicit library
python tile_engine/ops/hstu/hstu_benchmark.py \
  --lib dispatcher/build/examples/libdispatcher_hstu_lib.so --best
```

See also `tile_engine/ops/hstu/README.md` for mask/shape conventions and harness bridge notes.

## Dispatcher layout

- Registry/dispatcher: `dispatcher/include/ck_tile/dispatcher/hstu_*.hpp`, `dispatcher/src/hstu_*.cpp`
- ctypes: `dispatcher/bindings/ctypes/hstu_ctypes_lib.cpp`
- Python: `dispatcher/python/hstu_utils.py`
