# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GEMM sweep integration: profile rocKE's GEMM sweep with the primitives.

The reference for wiring the `rocke.benchmark.perf` primitives over rocKE's EXISTING
GEMM sweep. It:

  - reuses rocKE's own `rocke.benchmark.gemm.fp16_rcr_sweep` to expand + compile the
    variant set (NO new enumeration);
  - for each compiled variant runs the pipeline (`harness.profile` x repeats ->
    `aggregate` -> `store`) to produce one record per variant;
  - self-checks each variant against its prior stored run.

Boundary note: this stays on the "produce + simple local store" side. It is
single-box and writes only to the local user cache dir. The *system* work - choosing
which GPUs run the sweep, scheduling, running it at scale, and mass/central data
storage - belongs to the external perf framework, which drives the same primitives
through its own orchestration.

Run: `python -m rocke.benchmark.perf.examples.profile_gemm_sweep --arch gfx950 --shape 512x512x512`
(needs rocKE importable + a GPU). Identity = each variant's stable `cache_key`
(so a re-run pairs); the compiled symbol is the profiler `match`. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import namedtuple
from pathlib import Path
from typing import Optional, Sequence

from rocke.benchmark.perf import aggregate as _aggregate
from rocke.benchmark.perf import harness as _harness
from rocke.benchmark.perf import report as _report
from rocke.benchmark.perf import schema as _schema
from rocke.benchmark.perf.tool import selfcheck as _selfcheck
from rocke.benchmark.perf.tool import store as _store

# A rocKE-type-free view of one compiled variant (keeps profile_variants testable).
Variant = namedtuple("Variant", "cache_key kernel_name hsaco manifest shape")


def _launch_cmd(v: Variant) -> list[str]:
    """The kernel-launch command for a variant: rocKE's run_manifest."""
    m, n, k = v.shape["M"], v.shape["N"], v.shape["K"]
    return [
        sys.executable,
        "-m",
        "rocke.run_manifest",
        v.hsaco,
        v.manifest,
        "--shape",
        f"{m},{n},{k}",
        "--verify",
    ]


def _validated_variants(plan, builds) -> list[Variant]:
    """Convert successful builds to variants, failing on incomplete coverage."""
    planned = list(plan.variants)
    if not planned:
        raise RuntimeError("sweep produced no applicable variants for requested shapes")

    by_key = {build.cache_key: build for build in builds}
    variants: list[Variant] = []
    failures: list[str] = []
    for planned_variant in planned:
        build = by_key.get(planned_variant.cache_key)
        if build is None:
            failures.append(f"{planned_variant.cache_key}: missing build record")
            continue
        if not build.ok:
            failures.append(
                f"{planned_variant.cache_key}: {build.error or 'unknown build failure'}"
            )
            continue
        variants.append(
            Variant(
                cache_key=planned_variant.cache_key,
                kernel_name=build.kernel_name,
                hsaco=build.hsaco_path,
                manifest=build.manifest_path,
                shape={
                    "M": planned_variant.shape.M,
                    "N": planned_variant.shape.N,
                    "K": planned_variant.shape.K,
                },
            )
        )

    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise RuntimeError(
            f"{len(failures)} of {len(planned)} planned variants failed to build:\n"
            f"{details}"
        )
    return variants


def profile_variants(
    variants: Sequence[Variant],
    arch: str,
    *,
    repeats: int = 3,
    warmup: int = 0,
    cache=None,
    warn=None,
) -> list[dict]:
    """Profile each compiled variant -> aggregate -> store. Returns the records.

    `label` = variant.cache_key (stable identity); `match` = compiled symbol;
    `warmup` = warmup dispatches to drop from counter medians (the manifest's
    warmup_iters, so counters exclude cold-cache warmup).
    """
    records: list[dict] = []
    for v in variants:
        samples = [
            _harness.profile(
                _launch_cmd(v),
                arch,
                match=v.kernel_name,
                label=v.cache_key,
                op="gemm",
                shape=dict(v.shape),
                warmup=warmup,
                warn=warn,
            )
            for _ in range(max(1, repeats))
        ]
        rec = _aggregate.aggregate(samples)
        _store.append(rec, cache=cache)
        records.append(rec)
    return records


def _build_variants(
    shapes: Sequence[tuple], arch: str, output_dir: Path
) -> "tuple[list[Variant], int]":
    """Reuse rocKE's sweep to expand + compile variants (lazy rocKE import).

    Returns (variants, warmup_iters) - warmup_iters is the sweep config's warmup
    count, which the manifest uses, so the caller can drop it from counter medians.
    """
    from rocke.benchmark.gemm import fp16_rcr_sweep as sw  # noqa: WPS433 (lazy)

    gemm_shapes = tuple(
        sw.GemmSweepShape(M=m, N=n, K=k, label=(rest[0] if rest else ""))
        for (m, n, k, *rest) in shapes
    )
    cfg = sw.GemmSweepConfig(arch=arch, shapes=gemm_shapes)
    plan = sw.expand_sweep(cfg)
    builds = sw.compile_sweep_variants(plan, output_dir)
    return _validated_variants(plan, builds), cfg.warmup_iters


def profile_sweep(
    shapes: Sequence[tuple],
    arch: str,
    *,
    repeats: int = 3,
    cache=None,
    output_dir=None,
    warn=None,
) -> list[dict]:
    """Expand+compile rocKE's GEMM sweep, then profile each variant into the store."""
    out = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="rbsweep_"))
    variants, warmup = _build_variants(shapes, arch, out)
    return profile_variants(
        variants, arch, repeats=repeats, warmup=warmup, cache=cache, warn=warn
    )


def _parse_shape(text: str) -> tuple:
    """'MxNxK[:label]' -> (M, N, K, label)."""
    core, _, label = text.partition(":")
    try:
        m, n, k = (int(x) for x in core.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--shape must be MxNxK (e.g. 512x512x512), got {text!r}")
    return (m, n, k, label)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="examples.profile_gemm_sweep",
        description="Drive the rocke.benchmark.perf primitives over rocKE's GEMM sweep.",
    )
    p.add_argument("--arch", default="gfx950")
    p.add_argument(
        "--shape",
        action="append",
        default=None,
        help="MxNxK[:label]; repeatable (default: a small demo set)",
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cache", default=None)
    p.add_argument("--output-dir", dest="output_dir", default=None)
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)

    shapes = (
        [_parse_shape(s) for s in a.shape]
        if a.shape
        else [(128, 128, 64, "small"), (512, 512, 512, "balanced")]
    )
    warn = lambda m: print(f"warning: {m}", file=sys.stderr)
    try:
        recs = profile_sweep(
            shapes,
            a.arch,
            repeats=a.repeats,
            cache=a.cache,
            output_dir=a.output_dir,
            warn=warn,
        )
    except RuntimeError as exc:
        raise SystemExit(f"profile_gemm_sweep: {exc}") from exc

    # Self-check each variant against its prior stored run (reuses selfcheck).
    history = _store.load(cache=a.cache)
    checks = [_selfcheck.check_history(history, _schema.identity(r)) for r in recs]
    if a.json:
        print(
            json.dumps({"records": recs, "selfcheck": checks}, sort_keys=True, indent=2)
        )
    else:
        print(f"profiled {len(recs)} variant(s) on {a.arch}:")
        for r, c in zip(recs, checks):
            print(_report.format_record(r))
            print(_selfcheck.format_result(c))
            print()
    return 1 if any(c.get("verdict") == "regressed" for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
