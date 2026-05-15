# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Runtime autotuning for CK DSL kernels.

Mirrors Triton's ``@triton.autotune`` design but specialised for the CK
DSL's spec-dataclass + IR-builder + launcher pipeline:

- *Configs* are :class:`Spec` dataclass instances (e.g.
  :class:`UniversalGemmSpec`, :class:`ImplicitGemmConvSpec`), one per
  point in the search space.
- *Build* is a user-supplied callable that turns a ``Spec`` into a
  ``(launcher, kernel_def)`` pair.
- *Key* is the shape signature: tuples of integers / strings that the
  cache uses to look up the previously-chosen winner.
- *Cache* lives both in memory (a process-local ``dict``) and
  optionally on disk (a small JSON file keyed by ``key``).

End-to-end usage::

    from ck_dsl.helpers.autotune import Autotuner, AutotuneConfig

    @Autotuner(
        configs=[
            AutotuneConfig(name="t128x128x32", spec=spec_a),
            AutotuneConfig(name="t256x128x32", spec=spec_b),
            ...
        ],
        key_fn=lambda M, N, K, dtype: (M, N, K, dtype),
        cache_path="~/.cache/ck_dsl_autotune.json",
        build_fn=build_universal_gemm,
        signature_fn=gemm_args_signature,
        prepare_args=prepare_gemm_args,
    )
    def launch_gemm(M, N, K, dtype, A, B, C):
        ...

    launch_gemm(M=4096, N=4096, K=4096, dtype='fp16', A=A, B=B, C=C)

Differences vs Triton's autotune that matter on AMD:

1. **Search is over Spec dataclasses, not kwargs.** Each Spec carries
   its tile / pipeline / epilogue / chiplet-swizzle / waves-per-eu
   choice as a typed object, so the search space is type-checked at
   construction time, not at JIT time.

2. **HIP-event timing.** Uses :func:`time_launches` (the same timer
   the production code path uses) so autotune-picked configs reflect
   real device-side wall time, not host overhead.

3. **Persistent cache.** Triton recomputes its cache on every Python
   restart unless you save the autotune object. We persist to a small
   JSON file, so subsequent runs are zero-overhead.

4. **Sub-second iteration.** Each CK DSL config builds in <2s
   (LLVM-IR + comgr); a full 10-20-config sweep finishes in 20-40s.
   CK Tile's equivalent C++ template instantiation sweep takes
   minutes per config.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


__all__ = [
    "AutotuneConfig",
    "AutotuneResult",
    "Autotuner",
    "autotune_sweep",
]


@dataclass(frozen=True)
class AutotuneConfig:
    """One point in the autotuning search space.

    ``spec`` is the kernel spec dataclass (e.g. ``UniversalGemmSpec``).
    ``name`` is a short label used in cache keys and benchmark
    printouts; choose one that survives a JSON round-trip.

    Attributes
    ----------
    spec
        The kernel spec passed to the user's ``build_fn``.
    name
        Short human-readable label, e.g. ``"t128x128x32_wpe2"``.
    extra
        Free-form dict for things outside the spec — e.g. grid
        overrides, launch-time flags. Forwarded to ``build_fn`` via
        ``config.extra``.
    """

    spec: Any
    name: str
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AutotuneResult:
    """One row of the autotune sweep's measurement table."""

    config_name: str
    ms_per_iter: float
    error: Optional[str] = None

    @property
    def is_ok(self) -> bool:
        return self.error is None and math.isfinite(self.ms_per_iter)


# ----- in-memory + on-disk cache -----------------------------------


def _cache_key_to_str(key: Tuple[Any, ...]) -> str:
    """Stable string key for any tuple of hashable shape descriptors.

    Pickles ``key`` through ``repr`` (deterministic for ints, tuples
    of ints, strings), then SHA-1-truncated so the on-disk JSON
    stays compact.
    """
    raw = repr(key).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


class _DiskCache:
    """Tiny JSON-on-disk autotune cache.

    The on-disk schema is::

        { "<sha1(repr(key))>": {"key_repr": str, "winner": str,
                                "ms": float, "all": {name: ms, ...}} }

    Keeping the original ``repr(key)`` alongside the hash makes the
    cache file inspectable by hand.
    """

    def __init__(self, path: Optional[os.PathLike]) -> None:
        self.path = Path(os.path.expanduser(path)).resolve() if path else None
        self._mem: Dict[str, Dict[str, Any]] = {}
        if self.path is not None and self.path.exists():
            try:
                with self.path.open() as f:
                    self._mem = json.load(f)
            except Exception:
                # Corrupt cache file — start fresh; don't blow up.
                self._mem = {}

    def get(self, key: Tuple[Any, ...]) -> Optional[str]:
        h = _cache_key_to_str(key)
        entry = self._mem.get(h)
        return entry["winner"] if entry else None

    def put(
        self,
        key: Tuple[Any, ...],
        winner: str,
        ms: float,
        all_ms: Dict[str, float],
    ) -> None:
        h = _cache_key_to_str(key)
        # Filter non-finite entries so the on-disk JSON stays strict-spec
        # compliant (Infinity / NaN aren't valid JSON; some downstream
        # readers reject them). Configs that errored or timed out are
        # simply omitted from the ``all`` map — the winner column still
        # tells you which one was picked.
        self._mem[h] = {
            "key_repr": repr(key),
            "winner": winner,
            "ms": float(ms),
            "all": {k: float(v) for k, v in all_ms.items() if math.isfinite(v)},
        }
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open("w") as f:
                json.dump(self._mem, f, indent=2, sort_keys=True, allow_nan=False)
            tmp.replace(self.path)


# ----- core sweep --------------------------------------------------


def autotune_sweep(
    configs: Sequence[AutotuneConfig],
    *,
    bench_fn: Callable[[AutotuneConfig], float],
    on_progress: Optional[Callable[[AutotuneResult], None]] = None,
) -> Tuple[AutotuneConfig, List[AutotuneResult]]:
    """Run ``bench_fn`` for every config and return ``(winner, all_results)``.

    ``bench_fn(config)`` must return a per-call wall time in
    milliseconds (typically the output of :func:`time_launches`).
    Configs that throw are recorded with ``error`` set; they're
    excluded from the winner pick.
    """
    if not configs:
        raise ValueError("autotune_sweep: empty config list")
    results: List[AutotuneResult] = []
    for cfg in configs:
        try:
            ms = float(bench_fn(cfg))
            row = AutotuneResult(config_name=cfg.name, ms_per_iter=ms)
        except Exception as e:  # pragma: no cover — broad-catch by design
            row = AutotuneResult(
                config_name=cfg.name,
                ms_per_iter=float("inf"),
                error=f"{type(e).__name__}: {e}",
            )
        results.append(row)
        if on_progress is not None:
            on_progress(row)
    ok_rows = [r for r in results if r.is_ok]
    if not ok_rows:
        joined = "; ".join(f"{r.config_name}={r.error}" for r in results if r.error)
        raise RuntimeError(f"autotune_sweep: every config errored.\n  errors: {joined}")
    best = min(ok_rows, key=lambda r: r.ms_per_iter)
    winner = next(c for c in configs if c.name == best.config_name)
    return winner, results


# ----- decorator-style wrapper ------------------------------------


class Autotuner:
    """High-level autotune decorator.

    Wraps a user ``launch_fn(spec, **runtime_args) -> None``. On the
    first call for a given ``key_fn(**runtime_args)``, sweeps all
    configs, caches the winner (in-memory + optional disk), and from
    then on calls ``launch_fn(winner_spec, **runtime_args)`` directly.

    Parameters
    ----------
    configs
        List of :class:`AutotuneConfig` to consider.
    key_fn
        Function ``(**runtime_args) -> Tuple[Hashable, ...]`` that
        returns the shape signature used for cache lookup. Typically
        ``lambda *, M, N, K, dtype: (M, N, K, dtype)``.
    bench_fn
        ``(config: AutotuneConfig, **runtime_args) -> float ms``.
        Caller is responsible for building the kernel from
        ``config.spec``, running warmups, and returning a HIP-event
        timed average.
    launch_fn
        ``(config: AutotuneConfig, **runtime_args) -> None``. Called
        once per real launch with the winning config.
    cache_path
        Optional path to a JSON file; ``None`` keeps only the
        in-memory cache.
    warmup_iters / bench_iters
        Forwarded to ``bench_fn`` via ``runtime_args['_warmup_iters']``
        and ``['_bench_iters']`` (the caller can ignore them if their
        ``bench_fn`` has its own defaults).
    verbose
        Print one progress line per config + a final winner line.
    """

    def __init__(
        self,
        configs: Sequence[AutotuneConfig],
        *,
        key_fn: Callable[..., Tuple[Any, ...]],
        bench_fn: Callable[..., float],
        launch_fn: Callable[..., None],
        cache_path: Optional[os.PathLike] = None,
        warmup_iters: int = 10,
        bench_iters: int = 50,
        verbose: bool = False,
    ) -> None:
        if not configs:
            raise ValueError("Autotuner: empty config list")
        self.configs = list(configs)
        self._by_name = {c.name: c for c in self.configs}
        if len(self._by_name) != len(self.configs):
            raise ValueError("Autotuner: duplicate config names")
        self.key_fn = key_fn
        self.bench_fn = bench_fn
        self.launch_fn = launch_fn
        self.cache = _DiskCache(cache_path)
        self.warmup_iters = int(warmup_iters)
        self.bench_iters = int(bench_iters)
        self.verbose = bool(verbose)

    # ----- internal -----------------------------------------------

    def _emit(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr)

    def _select(self, **runtime_args: Any) -> AutotuneConfig:
        key = tuple(self.key_fn(**runtime_args))
        cached_name = self.cache.get(key)
        if cached_name is not None and cached_name in self._by_name:
            return self._by_name[cached_name]
        self._emit(f"[autotune] sweeping {len(self.configs)} configs for key={key!r}")

        bench_kwargs = dict(runtime_args)
        bench_kwargs["_warmup_iters"] = self.warmup_iters
        bench_kwargs["_bench_iters"] = self.bench_iters

        def _bench(cfg: AutotuneConfig) -> float:
            t_wall_0 = time.perf_counter()
            ms = float(self.bench_fn(cfg, **bench_kwargs))
            elapsed = time.perf_counter() - t_wall_0
            self._emit(
                f"[autotune]   {cfg.name:30s}: {ms * 1e3:.2f} us/iter"
                f"  (build+bench wall: {elapsed:.2f}s)"
            )
            return ms

        def _on_progress(row: AutotuneResult) -> None:
            if row.error is not None and self.verbose:
                self._emit(f"[autotune]   {row.config_name:30s}: SKIP {row.error}")

        winner, results = autotune_sweep(
            self.configs,
            bench_fn=_bench,
            on_progress=_on_progress,
        )
        all_ms = {r.config_name: r.ms_per_iter for r in results}
        self.cache.put(key, winner.name, all_ms[winner.name], all_ms)
        if self.verbose:
            sorted_rows = sorted(
                (r for r in results if r.is_ok),
                key=lambda r: r.ms_per_iter,
            )
            self._emit(
                f"[autotune] winner: {winner.name} "
                f"({sorted_rows[0].ms_per_iter * 1e3:.2f} us/iter)"
            )
        return winner

    # ----- public -------------------------------------------------

    def __call__(self, **runtime_args: Any) -> None:
        """Resolve the winning config and dispatch one launch."""
        cfg = self._select(**runtime_args)
        self.launch_fn(cfg, **runtime_args)

    def select(self, **runtime_args: Any) -> AutotuneConfig:
        """Return the winning config without dispatching a launch."""
        return self._select(**runtime_args)

    def clear_cache(self) -> None:
        """Wipe both in-memory and on-disk caches."""
        self.cache._mem = {}
        if self.cache.path is not None and self.cache.path.exists():
            self.cache.path.unlink()


# ----- spec helpers ------------------------------------------------


def spec_replace(spec: Any, **overrides: Any) -> Any:
    """Convenience: ``dataclasses.replace`` with a friendlier name.

    Use to build sweep config lists without re-typing every spec field::

        base = UniversalGemmSpec(name=..., tile=..., trait=...)
        configs = [
            AutotuneConfig(spec=spec_replace(base, name=n,
                                             tile=tile_a), name=n)
            for n, tile_a in tile_choices.items()
        ]

    Falls back to plain ``dataclasses.replace`` for spec dataclasses,
    and to a constructor kwargs dict otherwise.
    """
    if dataclasses.is_dataclass(spec):
        return dataclasses.replace(spec, **overrides)
    # Best-effort fallback for non-dataclass spec objects.
    kwargs = dict(getattr(spec, "__dict__", {}))
    kwargs.update(overrides)
    return type(spec)(**kwargs)
