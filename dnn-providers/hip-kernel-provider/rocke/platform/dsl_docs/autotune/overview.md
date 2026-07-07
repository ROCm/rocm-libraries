# Autotuner Overview

`helpers/autotune.py` is the in-process autotuner. It mirrors Triton's `@triton.autotune` design but specialized for the CK DSL's spec-dataclass + IR-builder + launcher pipeline.

## Concepts

- **Config**: one `AutotuneConfig(spec=..., name=..., extra={})` per point in the search space. `spec` is a kernel spec dataclass (e.g. `UniversalGemmSpec`, `ImplicitGemmConvSpec`).
- **Key**: a tuple of runtime shape / dtype / layout values, produced by the user-supplied `key_fn`. The autotuner caches the per-key winner.
- **Bench callable**: user-supplied `bench_fn(config, **runtime_args) -> ms`. The caller builds the kernel from `config.spec`, runs warmups, and returns a HIP-event-timed average (typically via `time_launches`).
- **Launch callable**: user-supplied `launch_fn(config, **runtime_args) -> None`, invoked once per real launch with the winning config.
- **Cache**: in-memory dict, plus an optional JSON file on disk keyed by the `key_fn` tuple.

## Top-Level API

```text
AutotuneConfig(spec, name, extra={})
AutotuneResult(config_name, ms_per_iter, error=None)   # .is_ok property
AutotuneKey(graph_hash, shape, dtype, layout="RCR", arch="gfx950",
            compiler="comgr", lowerer="unknown", spec_hash="any")
make_autotune_key(*, graph_hash, shape, dtype, layout="RCR", arch="gfx950",
                  compiler="comgr", lowerer="unknown", spec_hash="any")
Autotuner(configs, *, key_fn, bench_fn, launch_fn, cache_path=None,
          warmup_iters=10, bench_iters=50, verbose=False)
autotune_sweep(configs, *, bench_fn, on_progress=None)
    -> (winner_config, List[AutotuneResult])
spec_replace(spec, **kwargs)  # dataclasses.replace alias
```

## End-to-End Recipe

```python
from rocke.helpers import Autotuner, AutotuneConfig, autotune_sweep
from rocke.runtime.launcher import time_launches
from rocke.instances import (
    UniversalGemmSpec, TileSpec, TraitSpec,
    build_universal_gemm,
)

configs = [
    AutotuneConfig(name="t128_a32x32x16",
                   spec=UniversalGemmSpec(
                       name="hero",
                       tile=TileSpec(128,128,32, warp_m=2, warp_n=2,
                                     warp_tile_m=32, warp_tile_n=32, warp_tile_k=16),
                       trait=TraitSpec(pipeline="compv4", epilogue="cshuffle"),
                   )),
    AutotuneConfig(name="t256_a32x32x16",
                   spec=UniversalGemmSpec(
                       name="hero",
                       tile=TileSpec(256,128,32, warp_m=2, warp_n=2,
                                     warp_tile_m=32, warp_tile_n=32, warp_tile_k=16),
                       trait=TraitSpec(pipeline="compv4", epilogue="cshuffle"),
                   )),
]

# bench_fn owns build + compile + launch + timing for one config.
def bench_gemm(config, *, M, N, K, dtype, A, B, C, **_):
    kdef = build_universal_gemm(config.spec)
    # ... compile_kernel(kdef), construct a KernelLauncher, then:
    return time_launches(launcher_call, warmup=10, iters=100)  # ms/iter

# launch_fn dispatches a single real launch with the winning config.
def launch_gemm(config, *, M, N, K, dtype, A, B, C, **_):
    ...  # build/compile (cached) + launch once

tuner = Autotuner(
    configs,
    key_fn=lambda *, M, N, K, dtype, **_: (int(M), int(N), int(K), str(dtype)),
    bench_fn=bench_gemm,
    launch_fn=launch_gemm,
    cache_path="~/.cache/rocke_autotune.json",
)

# First call for a key sweeps + caches; later calls dispatch directly.
tuner(M=4096, N=4096, K=4096, dtype="fp16", A=A, B=B, C=C)
```

First call for a given key:

1. `key_fn(**runtime_args)` produces the cache key; on a miss the sweep runs.
2. The autotuner calls `bench_fn(config, **runtime_args)` for each config — the
   caller builds/compiles the kernel from `config.spec` and returns a
   HIP-event-timed ms/iter (typically via `time_launches`).
3. Picks the config with the lowest ms (`AutotuneResult.is_ok` filters errors).
4. Caches `(key, winner_name)` to disk.
5. Subsequent calls look up the cached winner and call `launch_fn` directly.

Subsequent calls for the same key:

1. Look up the cached winner.
2. Launch directly with no re-tuning.

## Why It's Fast On CK DSL

- Each config builds in ~15-30 ms warm (vs minutes per CK Tile template instantiation).
- A 10-20 config sweep finishes in 20-40 s on a current ROCm box.
- The HIP-event timer in `time_launches` is the same one production uses, so the chosen config reflects real device-side wall time.
- The disk cache makes a subsequent Python startup pay zero retuning cost.

## Cache Layout

The JSON cache is a flat dict:

```json
{
  "(4096, 4096, 4096, 'fp16')": "t128_a32x32x16",
  "(8192, 8192, 8192, 'fp16')": "t256_a32x32x16"
}
```

`AutotuneKey` extends the basic tuple with `arch`, `compiler`, and `lowerer` fields to avoid cache poisoning across ROCm versions or backend changes.

## Manual Sweep (No Caching)

`autotune_sweep(configs, *, bench_fn, on_progress=None)` runs the sweep without
any cache, returning a `(winner_config, results)` pair where `results` is a list
of `AutotuneResult` rows. Use this for one-shot exploration and CSV export.

```python
from rocke.helpers import autotune_sweep

# bench_fn takes only the config here (close over the fixed shape/args).
winner, results = autotune_sweep(
    configs,
    bench_fn=lambda cfg: bench_gemm(
        cfg, M=4096, N=4096, K=4096, dtype="fp16", A=A, B=B, C=C
    ),
)

for r in sorted(results, key=lambda r: r.ms_per_iter):
    print(f"{r.config_name:30s}  {r.ms_per_iter * 1000:8.2f} us  ok={r.is_ok}")
```

## Differences vs Triton autotune

- Configs are typed `Spec` dataclasses, not kwargs. The search space is checked at construction.
- Timing uses HIP events through `time_launches`, not host timing.
- The cache is in-memory by default; pass `cache_path=` to also persist it as JSON across processes.
- Build + launch is < 30 ms per config warm vs minutes per CK Tile template.
- Errors raised by `bench_fn` (validation failure, unsupported config) are recorded as `AutotuneResult(error=...)` rather than crashing the sweep.

## Failure Modes

- `key_fn` accidentally captures non-hashable values (lists, tensors). Use `int`, `str`, `tuple` only.
- `bench_fn` builds a kernel whose launch signature doesn't match the `runtime_args` it was handed. Validate the signature against the spec before adding the config.
- Cache file on a slow filesystem; the autotuner does a write per winner. Move the cache to a fast disk if many keys are seen.
- `bench_fn` uses too few iters to discount cold-cache effects on the first config. Run a throw-away warmup loop inside `bench_fn` before timing.

## When To Use vs The Sweep CLI

- `helpers/autotune.py::Autotuner` is for in-process selection from a known config set, with caching.
- `python -m rocke.sweep` + `python -m rocke.sweep_bench` is for offline cartesian sweeps with median + spread + CSV output. Use that when the goal is to *find* a good set of configs.
- The two compose: run the offline sweep, harvest the best 5-10 configs, register them as `AutotuneConfig`s in the dispatcher.
