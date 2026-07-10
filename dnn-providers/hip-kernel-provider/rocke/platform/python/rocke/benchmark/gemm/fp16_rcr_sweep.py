# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dispatcher-backed FP16 RCR GEMM sweep harness.

The flow follows ``dsl_docs/development/rocke_dispatcher_design.md``:

1. expand a bounded variant set from registered dispatcher candidates;
2. normalize each variant into a dispatcher ``KernelId``;
3. statically filter unsupported variants and dedupe by ``KernelId``;
4. optionally compile variants to HSACO + manifest artifacts;
5. optionally run correctness/benchmark through ``rocke.run_manifest``;
6. record all outcomes as structured JSON.

This harness intentionally lives under ``rocke.benchmark.gemm`` rather than the
dispatcher package: dispatch selects candidates, benchmark records evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

from ...dispatch import GemmRequest, dispatch_gemm_fp16
from ...dispatch.gemm import GEMM_FP16_REGISTRY, build_kernel
from ...helpers import compile_kernel, make_gemm_manifest, write_artifact

SWEEP_SCHEMA = "ck.dsl.benchmark.gemm.fp16_rcr_sweep/v1"


@dataclass(frozen=True)
class GemmSweepShape:
    M: int
    N: int
    K: int
    label: str = ""
    verify: bool = False

    def as_tuple(self) -> tuple[int, int, int]:
        return (int(self.M), int(self.N), int(self.K))

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GemmSweepConfig:
    arch: str = "gfx950"
    dtype: str = "fp16"
    layout: str = "RCR"
    algorithm: str = "auto"
    spec_id: str = "auto"
    shapes: tuple[GemmSweepShape, ...] = field(default_factory=tuple)
    warmup_iters: int = 5
    timed_iters: int = 100


@dataclass(frozen=True)
class GemmSweepVariant:
    shape: GemmSweepShape
    candidate: str
    family: str
    algorithm: str
    spec_id: str
    kernel_id: dict
    cache_key: str
    spec: dict
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    signature: tuple[dict, ...]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["shape"] = self.shape.as_dict()
        return d


@dataclass(frozen=True)
class GemmFilteredVariant:
    shape: GemmSweepShape
    candidate: str
    reason: str

    def as_dict(self) -> dict:
        return {
            "shape": self.shape.as_dict(),
            "candidate": self.candidate,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GemmBuildRecord:
    cache_key: str
    ok: bool
    hsaco_path: str = ""
    manifest_path: str = ""
    kernel_name: str = ""
    timings_ms: Mapping[str, float] = field(default_factory=dict)
    hsaco_bytes: int = 0
    error: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GemmRunRecord:
    cache_key: str
    shape: GemmSweepShape
    ok: bool
    verify: bool
    stdout: str = ""
    stderr: str = ""
    ms: Optional[float] = None
    tflops: Optional[float] = None
    gbps: Optional[float] = None
    error: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["shape"] = self.shape.as_dict()
        return d


@dataclass(frozen=True)
class GemmSweepPlan:
    config: GemmSweepConfig
    variants: tuple[GemmSweepVariant, ...]
    filtered: tuple[GemmFilteredVariant, ...]

    def as_dict(self) -> dict:
        return {
            "schema": SWEEP_SCHEMA,
            "config": {
                **asdict(self.config),
                "shapes": [s.as_dict() for s in self.config.shapes],
            },
            "variants": [v.as_dict() for v in self.variants],
            "filtered": [f.as_dict() for f in self.filtered],
        }


def default_gemm_shapes() -> tuple[GemmSweepShape, ...]:
    """Representative GEMM shapes from shipped examples/docs."""
    return (
        GemmSweepShape(128, 128, 64, "small-correctness", verify=True),
        GemmSweepShape(512, 512, 8192, "k-heavy"),
        GemmSweepShape(1024, 1024, 1024, "balanced"),
        GemmSweepShape(8192, 512, 512, "tall-skinny"),
        GemmSweepShape(2048, 2048, 2048, "balanced-large"),
        GemmSweepShape(4096, 4096, 512, "wide-mn"),
        GemmSweepShape(4096, 4096, 4096, "square-4096"),
    )


def _request_for_shape(config: GemmSweepConfig, shape: GemmSweepShape) -> GemmRequest:
    return GemmRequest(
        M=shape.M,
        N=shape.N,
        K=shape.K,
        arch=config.arch,
        dtype=config.dtype,
        layout=config.layout,
        algorithm=config.algorithm,
        spec_id=config.spec_id,
    )


def expand_sweep(config: GemmSweepConfig) -> GemmSweepPlan:
    """Expand a config into supported dispatcher variants and filtered records."""
    shapes = config.shapes or default_gemm_shapes()
    normalized = GemmSweepConfig(**{**asdict(config), "shapes": tuple(shapes)})
    variants: list[GemmSweepVariant] = []
    filtered: list[GemmFilteredVariant] = []
    seen: set[str] = set()

    for shape in shapes:
        req = _request_for_shape(normalized, shape)
        for candidate in GEMM_FP16_REGISTRY.candidates():
            ok, reason = candidate.supports(req)
            if not ok:
                filtered.append(GemmFilteredVariant(shape, candidate.name, reason))
                continue
            result = dispatch_gemm_fp16(
                GemmRequest(
                    M=shape.M,
                    N=shape.N,
                    K=shape.K,
                    arch=normalized.arch,
                    dtype=normalized.dtype,
                    layout=normalized.layout,
                    algorithm=candidate.algorithm,
                    spec_id=candidate.spec_id,
                )
            )
            if result.kernel_id.cache_key in seen:
                continue
            seen.add(result.kernel_id.cache_key)
            variants.append(
                GemmSweepVariant(
                    shape=shape,
                    candidate=result.candidate.name,
                    family=result.candidate.family,
                    algorithm=result.candidate.algorithm,
                    spec_id=result.candidate.spec_id,
                    kernel_id=result.kernel_id.as_dict(),
                    cache_key=result.kernel_id.cache_key,
                    spec=asdict(result.spec),
                    grid=result.grid,
                    block=result.block,
                    signature=tuple(result.signature),
                )
            )

    return GemmSweepPlan(
        config=normalized,
        variants=tuple(variants),
        filtered=tuple(filtered),
    )


def _variant_request(v: GemmSweepVariant) -> GemmRequest:
    return GemmRequest(
        M=v.shape.M,
        N=v.shape.N,
        K=v.shape.K,
        arch=v.kernel_id["arch"],
        algorithm=v.algorithm,
        spec_id=v.spec_id,
    )


def compile_variant(
    variant: GemmSweepVariant,
    output_dir: Path,
    *,
    warmup_iters: int,
    timed_iters: int,
) -> GemmBuildRecord:
    """Compile one variant and write HSACO + manifest artifacts."""
    try:
        req = _variant_request(variant)
        result = dispatch_gemm_fp16(req)
        artifact = compile_kernel(build_kernel(result), arch=req.arch)
        spec = result.spec
        case_dir = output_dir / _safe_dir_name(variant.cache_key)
        manifest = make_gemm_manifest(
            artifact=artifact,
            block_m=spec.tile.tile_m,
            block_n=spec.tile.tile_n,
            block_k=spec.tile.tile_k,
            threads_per_block=spec.block_size,
            default_shape=variant.shape.as_tuple(),
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
            args_signature=list(result.signature),
            atoms=["dispatcher-gemm-fp16-rcr-sweep"],
            extra={"dispatcher_kernel_id": result.kernel_id.as_dict()},
        )
        write_artifact(artifact, case_dir, manifest)
        return GemmBuildRecord(
            cache_key=variant.cache_key,
            ok=True,
            hsaco_path=str(case_dir / f"{artifact.kernel_name}.hsaco"),
            manifest_path=str(case_dir / "manifest.json"),
            kernel_name=artifact.kernel_name,
            timings_ms=dict(artifact.timings),
            hsaco_bytes=artifact.hsaco_bytes,
        )
    except Exception as exc:
        return GemmBuildRecord(
            cache_key=variant.cache_key,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def compile_sweep_variants(
    plan: GemmSweepPlan,
    output_dir: Path,
    *,
    parallel: int = 1,
) -> tuple[GemmBuildRecord, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if parallel <= 1:
        return tuple(
            compile_variant(
                v,
                output_dir,
                warmup_iters=plan.config.warmup_iters,
                timed_iters=plan.config.timed_iters,
            )
            for v in plan.variants
        )
    records: list[GemmBuildRecord] = []
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futs = [
            pool.submit(
                compile_variant,
                v,
                output_dir,
                warmup_iters=plan.config.warmup_iters,
                timed_iters=plan.config.timed_iters,
            )
            for v in plan.variants
        ]
        for fut in as_completed(futs):
            records.append(fut.result())
    return tuple(sorted(records, key=lambda r: r.cache_key))


def _parse_perf(
    stdout: str,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Parse the structured ``PerfJSON:`` line emitted by ``run_manifest``.

    Both ends live in this package, so we parse the machine-readable line
    rather than the human ``Perf:`` string. Returns ``(None, None, None)`` only
    when the line is absent or malformed; the caller treats that as a failure.
    """
    for line in stdout.splitlines():
        if not line.startswith("PerfJSON:"):
            continue
        try:
            payload = json.loads(line.removeprefix("PerfJSON:").strip())
            return (
                float(payload["ms"]),
                float(payload["tflops"]),
                float(payload["gbps"]),
            )
        except Exception:
            return None, None, None
    return None, None, None


def run_build_record(
    record: GemmBuildRecord,
    variant: GemmSweepVariant,
    *,
    timeout_s: int = 180,
) -> GemmRunRecord:
    if not record.ok:
        return GemmRunRecord(
            cache_key=record.cache_key,
            shape=variant.shape,
            ok=False,
            verify=variant.shape.verify,
            error=f"build failed: {record.error}",
        )
    cmd = [
        sys.executable,
        "-m",
        "rocke.run_manifest",
        record.hsaco_path,
        record.manifest_path,
        "--shape",
        ",".join(str(x) for x in variant.shape.as_tuple()),
    ]
    if variant.shape.verify:
        cmd.append("--verify")
    env = dict(os.environ)
    py_root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = py_root + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        env=env,
    )
    ms, tflops, gbps = _parse_perf(proc.stdout or "")
    metrics_ok = ms is not None and tflops is not None and gbps is not None
    ok = proc.returncode == 0 and metrics_ok
    if proc.returncode != 0:
        error = f"run_manifest rc={proc.returncode}"
    elif not metrics_ok:
        error = "run_manifest exited 0 but no parseable PerfJSON metrics found"
    else:
        error = ""
    return GemmRunRecord(
        cache_key=record.cache_key,
        shape=variant.shape,
        ok=ok,
        verify=variant.shape.verify,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        ms=ms,
        tflops=tflops,
        gbps=gbps,
        error=error,
    )


def run_sweep_variants(
    plan: GemmSweepPlan,
    builds: Sequence[GemmBuildRecord],
    *,
    parallel: int = 1,
    timeout_s: int = 180,
) -> tuple[GemmRunRecord, ...]:
    variant_by_key = {v.cache_key: v for v in plan.variants}
    tasks = [
        (b, variant_by_key[b.cache_key])
        for b in builds
        if b.cache_key in variant_by_key
    ]
    if parallel <= 1:
        return tuple(run_build_record(b, v, timeout_s=timeout_s) for b, v in tasks)
    records: list[GemmRunRecord] = []
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futs = [
            pool.submit(run_build_record, b, v, timeout_s=timeout_s) for b, v in tasks
        ]
        for fut in as_completed(futs):
            records.append(fut.result())
    return tuple(sorted(records, key=lambda r: r.cache_key))


def write_sweep_json(
    path: Path,
    plan: GemmSweepPlan,
    *,
    builds: Sequence[GemmBuildRecord] = (),
    runs: Sequence[GemmRunRecord] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = plan.as_dict()
    doc["builds"] = [b.as_dict() for b in builds]
    doc["runs"] = [r.as_dict() for r in runs]
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


_CSV_KERNEL_KEY_COLUMNS = (
    "candidate",
    "algorithm",
    "spec_id",
    "arch",
    "abi_version",
    "spec_hash",
    "request_hash",
)

_CSV_CONFIG_COLUMNS = (
    "dtype_a",
    "dtype_b",
    "dtype_c",
    "dtype_acc",
    "layout",
    "tile_m",
    "tile_n",
    "tile_k",
    "warp_m",
    "warp_n",
    "warp_k",
    "warp_tile_m",
    "warp_tile_n",
    "warp_tile_k",
    "pipeline",
    "scheduler",
    "epilogue",
    "pad_m",
    "pad_n",
    "pad_k",
    "persistent",
    "block_size",
    "wave_size",
)


def _variant_kernel_columns(variant: GemmSweepVariant) -> dict:
    """Flatten a variant's kernel identity + config into CSV cells.

    This is what lets a measured ``tflops`` row map back to a specific kernel
    without the JSON sidecar: ``cache_key`` is the unique key, and the kernel
    identity (``KernelId`` fields) plus the spec knobs (tile/warp/pipeline/...)
    are the features a heuristic would train on.
    """
    kid = variant.kernel_id
    spec = variant.spec
    tile = spec.get("tile", {})
    trait = spec.get("trait", {})
    data = spec.get("data", {})
    cols = {col: kid.get(col, "") for col in _CSV_KERNEL_KEY_COLUMNS}
    cols.update(
        {
            "dtype_a": data.get("dtype_a", ""),
            "dtype_b": data.get("dtype_b", ""),
            "dtype_c": data.get("dtype_c", ""),
            "dtype_acc": data.get("dtype_acc", ""),
            "layout": data.get("layout", ""),
            "tile_m": tile.get("tile_m", ""),
            "tile_n": tile.get("tile_n", ""),
            "tile_k": tile.get("tile_k", ""),
            "warp_m": tile.get("warp_m", ""),
            "warp_n": tile.get("warp_n", ""),
            "warp_k": tile.get("warp_k", ""),
            "warp_tile_m": tile.get("warp_tile_m", ""),
            "warp_tile_n": tile.get("warp_tile_n", ""),
            "warp_tile_k": tile.get("warp_tile_k", ""),
            "pipeline": trait.get("pipeline", ""),
            "scheduler": trait.get("scheduler", ""),
            "epilogue": trait.get("epilogue", ""),
            "pad_m": trait.get("pad_m", ""),
            "pad_n": trait.get("pad_n", ""),
            "pad_k": trait.get("pad_k", ""),
            "persistent": trait.get("persistent", ""),
            "block_size": spec.get("block_size", ""),
            "wave_size": spec.get("wave_size", ""),
        }
    )
    return cols


def write_sweep_csv(
    path: Path,
    runs: Sequence[GemmRunRecord],
    *,
    plan: Optional[GemmSweepPlan] = None,
) -> None:
    """Write a flat per-run CSV joining measured perf to the kernel config.

    Each row is keyed by ``cache_key`` (the dispatcher ``KernelId.cache_key``)
    and carries the kernel identity + config columns (tile/warp/pipeline/dtype/
    ...) alongside the measured ``ms``/``tflops``/``gbps``. That makes the CSV
    self-contained: a measured number maps back to a unique kernel without the
    JSON sidecar. ``plan`` supplies the per-variant config; if omitted, the
    config columns are left blank (perf-only output).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    variant_by_key = {v.cache_key: v for v in plan.variants} if plan else {}
    base_columns = [
        "cache_key",
        "M",
        "N",
        "K",
        "label",
        "ok",
        "verify",
        "ms",
        "tflops",
        "gbps",
        "error",
    ]
    fieldnames = (
        base_columns + list(_CSV_KERNEL_KEY_COLUMNS) + list(_CSV_CONFIG_COLUMNS)
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in runs:
            row = {
                "cache_key": r.cache_key,
                "M": r.shape.M,
                "N": r.shape.N,
                "K": r.shape.K,
                "label": r.shape.label,
                "ok": r.ok,
                "verify": r.verify,
                "ms": r.ms,
                "tflops": r.tflops,
                "gbps": r.gbps,
                "error": r.error,
            }
            variant = variant_by_key.get(r.cache_key)
            if variant is not None:
                row.update(_variant_kernel_columns(variant))
            writer.writerow(row)


def _safe_dir_name(cache_key: str) -> str:
    return cache_key.replace(":", "_").replace("/", "_")[:180]


def _parse_shape(raw: str) -> GemmSweepShape:
    parts = raw.split(":")
    dims = [int(x) for x in parts[0].replace(",", " ").split()]
    if len(dims) != 3:
        raise ValueError(f"shape must be M,N,K; got {raw!r}")
    label = parts[1] if len(parts) > 1 else ""
    verify = len(parts) > 2 and parts[2].lower() in ("1", "true", "yes", "verify")
    return GemmSweepShape(dims[0], dims[1], dims[2], label=label, verify=verify)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--layout", default="RCR")
    parser.add_argument("--algorithm", default="auto")
    parser.add_argument("--spec-id", default="auto")
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        help="M,N,K[:label[:verify]]. Repeatable. Defaults to representative shapes.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--timed-iters", type=int, default=100)
    parser.add_argument("--timeout-s", type=int, default=180)
    args = parser.parse_args(argv)

    shapes = (
        tuple(_parse_shape(s) for s in args.shape)
        if args.shape
        else default_gemm_shapes()
    )
    plan = expand_sweep(
        GemmSweepConfig(
            arch=args.arch,
            dtype=args.dtype,
            layout=args.layout,
            algorithm=args.algorithm,
            spec_id=args.spec_id,
            shapes=shapes,
            warmup_iters=args.warmup_iters,
            timed_iters=args.timed_iters,
        )
    )
    builds: tuple[GemmBuildRecord, ...] = ()
    runs: tuple[GemmRunRecord, ...] = ()
    if args.compile or args.run:
        builds = compile_sweep_variants(
            plan, args.output_dir / "artifacts", parallel=args.parallel
        )
    if args.run:
        runs = run_sweep_variants(
            plan,
            builds,
            parallel=1,
            timeout_s=args.timeout_s,
        )
    json_path = args.json or (args.output_dir / "gemm_fp16_rcr_sweep.json")
    write_sweep_json(json_path, plan, builds=builds, runs=runs)
    if args.csv or runs:
        write_sweep_csv(
            args.csv or (args.output_dir / "gemm_fp16_rcr_sweep.csv"),
            runs,
            plan=plan,
        )
    print(
        f"expanded {len(plan.variants)} variants, filtered {len(plan.filtered)}, "
        f"builds {len(builds)}, runs {len(runs)}"
    )
    print(f"wrote {json_path}")
    return 1 if any(not r.ok for r in runs) or any(not b.ok for b in builds) else 0


if __name__ == "__main__":
    raise SystemExit(main())
