# Autotuner Overview

`helpers/autotune.py` is the in-process autotuner. It mirrors Triton's `@triton.autotune` design but specialized for the CK DSL's spec-dataclass + IR-builder + launcher pipeline.

## Concepts

- **Config**: one `AutotuneConfig(spec=..., name=..., extra=...)` per point in the search space. `spec` is a kernel spec dataclass (e.g. `UniversalGemmSpec`, `ImplicitGemmConvSpec`).
- **Key**: a tuple of runtime shape / dtype / layout values. The autotuner caches the per-key winner.
- **Build callable**: user-supplied `build_fn(spec) -> KernelDef` plus a signature function and an `args_prepare` callable.
- **Cache**: in-memory dict, plus an optional JSON file on disk keyed by `AutotuneKey` (or a user-supplied `key_fn` tuple).

## Top-Level API

```text
AutotuneConfig(spec, name, extra={})
AutotuneResult(config_name, ms_per_iter, error)
AutotuneKey(graph_hash, shape, dtype, layout="RCR", arch="gfx950",
            compiler="comgr", lowerer="unknown", spec_hash="any")
make_autotune_key(...)
Autotuner(configs, key_fn, cache_path, build_fn, signature_fn, prepare_args)
autotune_sweep(configs, build_fn, prepare_args, signature_fn,
               iters=..., warmup=..., key=None) -> List[AutotuneResult]
spec_replace(spec, **kwargs)  # dataclasses.replace alias
```

## End-to-End Recipe

```python
from ck_dsl.helpers import (
    Autotuner, AutotuneConfig,
    UniversalGemmSpec, TileSpec, TraitSpec,
    gemm_args_signature,
)
from ck_dsl.instances import build_universal_gemm

def prepare_gemm_args(spec, M, N, K, dtype, A, B, C):
    return {"A": A, "B": B, "C": C, "M": M, "N": N, "K": K}

@Autotuner(
    configs=[
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
    ],
    key_fn=lambda M, N, K, dtype, A, B, C: (int(M), int(N), int(K), str(dtype)),
    cache_path="~/.cache/ck_dsl_autotune.json",
    build_fn=build_universal_gemm,
    signature_fn=lambda spec: gemm_args_signature(),
    prepare_args=prepare_gemm_args,
)
def launch_gemm(M, N, K, dtype, A, B, C):
    pass  # the autotuner inserts the launch; the body is ignored.
```

First call for a given key:

1. The autotuner builds each `config.spec` through `build_fn`.
2. Compiles each via `compile_kernel`.
3. Constructs a `KernelLauncher` per config.
4. Calls `time_launches(launcher_call, warmup=..., iters=...)`.
5. Picks the config with the lowest median ms.
6. Caches `(key, winner_name)` to disk.

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

## Manual Sweep (No Decorator)

`autotune_sweep(configs, build_fn, ...)` runs the sweep without caching, returning a list of `AutotuneResult` rows. Use this for one-shot exploration and CSV export.

```python
from ck_dsl.helpers import autotune_sweep

results = autotune_sweep(
    configs=configs,
    build_fn=build_universal_gemm,
    signature_fn=lambda spec: gemm_args_signature(),
    prepare_args=prepare_gemm_args,
    args=(4096, 4096, 4096, "fp16", A, B, C),
    iters=100,
    warmup=10,
)

for r in sorted(results, key=lambda r: r.ms_per_iter):
    print(f"{r.config_name:30s}  {r.ms_per_iter * 1000:8.2f} us  ok={r.is_ok}")
```

## Differences vs Triton autotune

- Configs are typed `Spec` dataclasses, not kwargs. The search space is checked at construction.
- Timing uses HIP events through `time_launches`, not host timing.
- The cache is persistent JSON, not in-memory only.
- Build + launch is < 30 ms per config warm vs minutes per CK Tile template.
- Errors during build (validation failure, unsupported config) are recorded as `AutotuneResult(error=...)` rather than crashing the sweep.

## Failure Modes

- `key_fn` accidentally captures non-hashable values (lists, tensors). Use `int`, `str`, `tuple` only.
- `prepare_args` returns a dict that doesn't match `signature_fn(spec)` for some config. Validate the signature against the spec before adding the config.
- Cache file on a slow filesystem; the autotuner does a write per winner. Move the cache to a fast disk if many keys are seen.
- `iters` too small to discount cold-cache effects on the first config. Use `discard_first=True` semantics by running a throw-away warmup loop before the sweep.

## When To Use vs The Sweep CLI

- `helpers/autotune.py::Autotuner` is for in-process selection from a known config set, with caching.
- `python -m ck_dsl.sweep` + `python -m ck_dsl.sweep_bench` is for offline cartesian sweeps with median + spread + CSV output. Use that when the goal is to *find* a good set of configs.
- The two compose: run the offline sweep, harvest the best 5-10 configs, register them as `AutotuneConfig`s in the dispatcher.
