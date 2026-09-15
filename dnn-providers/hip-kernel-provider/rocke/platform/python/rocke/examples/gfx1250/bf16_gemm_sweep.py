#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Staged BF16 RCR Universal GEMM sweep for gfx1250.

The target shape has an odd N. Universal GEMM masks the C tail but currently
expects operand padding for out-of-range B rows, so this harness allocates one
zero-filled B buffer large enough for every candidate's padded N extent.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import os
import statistics
import struct
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec,
)
from rocke.helpers import compile_kernel
from rocke.runtime.hip_module import Runtime
from rocke.sweep import BuildRecord, build_all_instances

DEFAULT_CONFIG = Path(__file__).with_name("bf16_gemm_sweep_config.json")


def load_config(path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _make_spec(
    config: Dict[str, Any],
    tile_m: int,
    tile_n: int,
    tile_k: int,
    warp_m: int,
    warp_n: int,
    pipeline: str,
    *,
    scheduler: str,
    epilogue: str,
    waves_per_eu: Optional[int] = None,
    lds_swizzle: bool = False,
    lds_k_pad: int = 0,
    direct_to_lds: bool = False,
    dtl_prefetch: bool = False,
) -> UniversalGemmSpec:
    target = config["target"]
    warp_tile_m, warp_tile_n, warp_tile_k = target["warp_tile"]
    dtype = str(target["dtype"])
    return UniversalGemmSpec(
        name=f"{target['arch']}_{dtype}_sweep",
        tile=TileSpec(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=warp_m,
            warp_n=warp_n,
            warp_k=1,
            warp_tile_m=int(warp_tile_m),
            warp_tile_n=int(warp_tile_n),
            warp_tile_k=int(warp_tile_k),
        ),
        trait=TraitSpec(
            pipeline=pipeline,  # type: ignore[arg-type]
            scheduler=scheduler,  # type: ignore[arg-type]
            epilogue=epilogue,  # type: ignore[arg-type]
            pad_m=True,
            pad_n=True,
            pad_k=True,
            waves_per_eu=waves_per_eu,
            lds_swizzle=lds_swizzle,
            lds_k_pad=lds_k_pad,
            direct_to_lds=direct_to_lds,
            dtl_prefetch=dtl_prefetch,
        ),
        data=DataSpec(
            dtype_a=dtype,
            dtype_b=dtype,
            dtype_c=dtype,
            dtype_acc="fp32",
            layout=target["layout"],
        ),
        wave_size=int(target["wave_size"]),
    )


def _spec_identity(spec: UniversalGemmSpec, *, arch: str) -> str:
    import hashlib

    payload = {
        "arch": arch,
        "tile": asdict(spec.tile),
        "trait": asdict(spec.trait),
        "data": asdict(spec.data),
        "wave_size": spec.wave_size,
        "block_size": spec.block_size,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _dedupe_valid(
    specs: Iterable[UniversalGemmSpec], *, arch: str
) -> List[UniversalGemmSpec]:
    out: List[UniversalGemmSpec] = []
    seen = set()
    for spec in specs:
        ok, _reason = is_valid_spec(spec, arch=arch)
        key = _spec_identity(spec, arch=arch)
        if ok and key not in seen:
            seen.add(key)
            out.append(spec)
    return out


def enumerate_tile_configs(config: Dict[str, Any]) -> List[UniversalGemmSpec]:
    tile = config["tile_config"]
    traits = config["trait_config"]
    specs = (
        _make_spec(
            config,
            tm,
            tn,
            tk,
            wm,
            wn,
            traits["pipelines"][0],
            scheduler=traits["schedulers"][0],
            epilogue=traits["epilogues"][0],
            waves_per_eu=traits["waves_per_eu"][0],
            lds_swizzle=traits["lds_swizzle"][0],
            lds_k_pad=traits["lds_k_pad"][0],
        )
        for tm in tile["tile_m"]
        for tn in tile["tile_n"]
        for tk in tile["tile_k"]
        for wm in tile["warp_m"]
        for wn in tile["warp_n"]
    )
    return _dedupe_valid(specs, arch=config["target"]["arch"])


def enumerate_trait_configs(
    config: Dict[str, Any], finalists: Sequence[UniversalGemmSpec]
) -> List[UniversalGemmSpec]:
    traits = config["trait_config"]
    specs = []
    for base in finalists:
        for pipeline in traits["pipelines"]:
            for scheduler in traits["schedulers"]:
                for epilogue in traits["epilogues"]:
                    for waves_per_eu in traits["waves_per_eu"]:
                        for lds_swizzle in traits["lds_swizzle"]:
                            for lds_k_pad in traits["lds_k_pad"]:
                                for direct_to_lds in traits.get(
                                    "direct_to_lds", [False]
                                ):
                                    for dtl_prefetch in traits.get(
                                        "dtl_prefetch", [False]
                                    ):
                                        specs.append(
                                            replace(
                                                base,
                                                trait=replace(
                                                    base.trait,
                                                    pipeline=pipeline,
                                                    scheduler=scheduler,
                                                    epilogue=epilogue,
                                                    waves_per_eu=waves_per_eu,
                                                    lds_swizzle=lds_swizzle,
                                                    lds_k_pad=lds_k_pad,
                                                    direct_to_lds=direct_to_lds,
                                                    dtl_prefetch=dtl_prefetch,
                                                ),
                                            )
                                        )
    return _dedupe_valid(specs, arch=config["target"]["arch"])


def required_padded_n(n: int, specs: Sequence[UniversalGemmSpec]) -> int:
    if not specs:
        return n
    return max(math.ceil(n / spec.tile.tile_n) * spec.tile.tile_n for spec in specs)


def _spec_from_dict(data: Dict[str, Any]) -> UniversalGemmSpec:
    return UniversalGemmSpec(
        name=str(data["name"]),
        tile=TileSpec(**data["tile"]),
        trait=TraitSpec(**data["trait"]),
        data=DataSpec(**data["data"]),
        wave_size=int(data["wave_size"]),
        block_size=int(data["block_size"]),
    )


def _as_u8_buffer(array) -> ctypes.Array:
    return (ctypes.c_ubyte * int(array.nbytes)).from_buffer(array)


def _float32_to_bf16(np, values):
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return ((bits + bias) >> 16).astype(np.uint16)


def _bf16_to_float32(np, values):
    bits = np.ascontiguousarray(values, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


@dataclass
class PreparedProblem:
    rt: Runtime
    shape: Tuple[int, int, int]
    padded_n: int
    a_dev: int
    b_dev: int
    c_dev: int
    c_host: Any
    reference: Any

    @classmethod
    def create(
        cls,
        shape: Tuple[int, int, int],
        padded_n: int,
        *,
        with_reference: bool,
    ) -> "PreparedProblem":
        import numpy as np

        m, n, k = shape
        rng = np.random.default_rng(0xC0FFEE)
        a_values = rng.integers(-5, 6, size=(m, k), dtype=np.int16)
        b_values = rng.integers(-5, 6, size=(n, k), dtype=np.int16)
        a = _float32_to_bf16(np, a_values)
        b = np.zeros((padded_n, k), dtype=np.uint16)
        b[:n] = _float32_to_bf16(np, b_values)
        c = np.empty((m, n), dtype=np.uint16)
        reference = None
        if with_reference:
            a_f32 = _bf16_to_float32(np, a)
            b_f32 = _bf16_to_float32(np, b[:n])
            reference = _bf16_to_float32(
                np, _float32_to_bf16(np, a_f32 @ b_f32.T)
            )

        rt = Runtime()
        a_dev = rt.alloc(a.nbytes)
        b_dev = rt.alloc(b.nbytes)
        c_dev = rt.alloc(c.nbytes)
        rt.memcpy_h2d(a_dev, _as_u8_buffer(a), a.nbytes)
        rt.memcpy_h2d(b_dev, _as_u8_buffer(b), b.nbytes)
        rt.memset(c_dev, 0, c.nbytes)
        return cls(rt, shape, padded_n, a_dev, b_dev, c_dev, c, reference)

    @property
    def args(self) -> bytes:
        m, n, k = self.shape
        return struct.pack(
            "<QQQiii", self.a_dev, self.b_dev, self.c_dev, m, n, k
        )

    def close(self) -> None:
        self.rt.sync()
        for ptr in (self.a_dev, self.b_dev, self.c_dev):
            self.rt.free(ptr)


def _launch_geometry(
    spec: UniversalGemmSpec, shape: Tuple[int, int, int]
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    m, n, _k = shape
    return (
        (
            math.ceil(n / spec.tile.tile_n),
            math.ceil(m / spec.tile.tile_m),
            1,
        ),
        (spec.block_size, 1, 1),
    )


def _time_function(
    rt: Runtime,
    fn,
    grid: Tuple[int, int, int],
    block: Tuple[int, int, int],
    args: bytes,
    *,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        rt.launch(fn, grid, block, args)
    rt.wait_stream(0)
    begin = rt.event()
    finish = rt.event()
    try:
        begin.record()
        for _ in range(iters):
            rt.launch(fn, grid, block, args)
        finish.record()
        finish.synchronize()
        elapsed = begin.elapsed_to(finish) / iters
    finally:
        begin.destroy()
        finish.destroy()
        rt.wait_stream(0)
    return elapsed


def benchmark_record(
    problem: PreparedProblem,
    record: BuildRecord,
    *,
    arch: str,
    warmup: int,
    iters: int,
    attempts: int,
) -> Dict[str, Any]:
    spec = _spec_from_dict(record.spec_dict)
    result: Dict[str, Any] = {
        "id": _spec_identity(spec, arch=arch),
        "name": record.name,
        "spec": record.spec_dict,
        "build_ok": record.ok,
        "build_error": record.error,
        "build_ms": record.total_build_ms,
        "build_timings_ms": {
            "ir_build": record.ir_build_ms,
            "ir_lower": record.ir_lower_ms,
            "comgr": record.comgr_ms,
            "total": record.total_build_ms,
        },
        "hsaco": record.hsaco_path,
        "hsaco_bytes": record.hsaco_bytes,
        "elf_meta": record.elf_meta,
        "verified": False,
    }
    if not record.ok:
        result["error"] = record.error or "build failed"
        return result

    module = None
    try:
        module = problem.rt.load_module(Path(record.hsaco_path).read_bytes())
        fn = module.get_function(record.name)
        grid, block = _launch_geometry(spec, problem.shape)
        samples = [
            _time_function(
                problem.rt,
                fn,
                grid,
                block,
                problem.args,
                warmup=warmup,
                iters=iters,
            )
            for _ in range(attempts)
        ]
        ms = statistics.median(samples)
        m, n, k = problem.shape
        result.update(
            {
                "samples_ms": samples,
                "median_ms": ms,
                "best_ms": min(samples),
                "tflops": 2.0 * m * n * k / (ms * 1.0e9),
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        problem.rt.wait_stream(0)
        if module is not None:
            module.unload()
    return result


def verify_record(
    problem: PreparedProblem,
    record: BuildRecord,
    result: Dict[str, Any],
    *,
    tolerance: float,
) -> bool:
    import numpy as np

    if problem.reference is None or not record.ok:
        return False
    spec = _spec_from_dict(record.spec_dict)
    module = None
    try:
        module = problem.rt.load_module(Path(record.hsaco_path).read_bytes())
        fn = module.get_function(record.name)
        grid, block = _launch_geometry(spec, problem.shape)
        problem.rt.memset(problem.c_dev, 0, problem.c_host.nbytes)
        problem.rt.launch_blocking(fn, grid, block, problem.args)
        problem.rt.memcpy_d2h(
            _as_u8_buffer(problem.c_host),
            problem.c_dev,
            problem.c_host.nbytes,
        )
        actual = _bf16_to_float32(np, problem.c_host)
        error = np.abs(actual - problem.reference)
        bad = error > tolerance + tolerance * np.abs(problem.reference)
        incorrect = int(np.count_nonzero(bad))
        result.update(
            {
                "verified": incorrect == 0,
                "max_abs_diff": float(error.max()),
                "incorrect": incorrect,
                "elements": int(actual.size),
            }
        )
        return incorrect == 0
    except Exception as exc:
        result["verify_error"] = f"{type(exc).__name__}: {exc}"
        return False
    finally:
        if module is not None:
            module.unload()


def rank_results(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        (r for r in results if "median_ms" in r and not r.get("error")),
        key=lambda r: float(r["median_ms"]),
    )


def _record_by_id(
    records: Sequence[BuildRecord], *, arch: str
) -> Dict[str, BuildRecord]:
    return {
        _spec_identity(_spec_from_dict(record.spec_dict), arch=arch): record
        for record in records
    }


def _write_results(
    output_dir: Path, stem: str, results: Sequence[Dict[str, Any]]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{stem}.json").write_text(
        json.dumps(list(results), indent=2, sort_keys=True)
    )
    fields = (
        "rank",
        "id",
        "name",
        "median_ms",
        "best_ms",
        "tflops",
        "verified",
        "max_abs_diff",
        "incorrect",
        "build_ms",
        "hsaco_bytes",
        "error",
        "verify_error",
        "hsaco",
    )
    ranked_ids = {r["id"]: i + 1 for i, r in enumerate(rank_results(results))}
    with (output_dir / f"{stem}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow({"rank": ranked_ids.get(result["id"], ""), **result})


def _load_tile_finalists(
    output_dir: Path, count: int
) -> Tuple[List[UniversalGemmSpec], List[Dict[str, Any]]]:
    path = output_dir / "tile.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is required when run_stages contains 'trait'"
        )
    results = json.loads(path.read_text())
    ranked = rank_results(results)
    verified = [r for r in ranked if r.get("verified")]
    selected = (verified or ranked)[:count]
    return [_spec_from_dict(r["spec"]) for r in selected], results


def _record_from_result(result: Dict[str, Any]) -> BuildRecord:
    timings = result.get("build_timings_ms", {})
    return BuildRecord(
        name=result["name"],
        spec_dict=result["spec"],
        ok=bool(result.get("build_ok")),
        error=str(result.get("build_error", "")),
        hsaco_path=str(result["hsaco"]),
        hsaco_bytes=int(result.get("hsaco_bytes", 0)),
        block_m=int(result["spec"]["tile"]["tile_m"]),
        block_n=int(result["spec"]["tile"]["tile_n"]),
        block_k=int(result["spec"]["tile"]["tile_k"]),
        threads_per_block=int(result["spec"]["block_size"]),
        ir_build_ms=float(timings.get("ir_build", 0.0)),
        ir_lower_ms=float(timings.get("ir_lower", 0.0)),
        comgr_ms=float(timings.get("comgr", 0.0)),
        elf_meta=dict(result.get("elf_meta", {})),
    )


def _load_trait_results(
    output_dir: Path,
) -> Tuple[List[BuildRecord], List[Dict[str, Any]]]:
    path = output_dir / "trait.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is required when run_stages contains 'final'"
        )
    results = json.loads(path.read_text())
    return [_record_from_result(result) for result in results], results


def _independent_final_timings(
    script: Path,
    output_dir: Path,
    record: BuildRecord,
    shape: Tuple[int, int, int],
    padded_n: int,
    *,
    arch: str,
    warmup: int,
    iters: int,
) -> List[float]:
    payload = {
        "record": asdict(record),
        "shape": list(shape),
        "padded_n": padded_n,
        "arch": arch,
        "warmup": warmup,
        "iters": iters,
    }
    worker_input = output_dir / "final_worker.json"
    worker_input.write_text(json.dumps(payload, indent=2))
    samples = []
    env = os.environ.copy()
    python_root = str(script.parents[3])
    env["PYTHONPATH"] = (
        python_root
        if not env.get("PYTHONPATH")
        else python_root + os.pathsep + env["PYTHONPATH"]
    )
    for attempt in range(6):
        proc = subprocess.run(
            [sys.executable, str(script), "--worker-record", str(worker_input)],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        marker = "ROCKE_SWEEP_WORKER_RESULT="
        lines = [line for line in proc.stdout.splitlines() if line.startswith(marker)]
        if proc.returncode != 0 or not lines:
            raise RuntimeError(
                f"final timing worker failed ({proc.returncode}): "
                f"{proc.stdout[-1000:]}{proc.stderr[-1000:]}"
            )
        ms = float(json.loads(lines[-1][len(marker) :])["median_ms"])
        if attempt:
            samples.append(ms)
    return samples


def measure_uncached_build(spec: UniversalGemmSpec, *, arch: str) -> Dict[str, Any]:
    artifact = compile_kernel(
        build_universal_gemm(spec, arch=arch),
        arch=arch,
        capture_ir_text=False,
    )
    return {
        "timings_ms": artifact.timings,
        "hsaco_bytes": artifact.hsaco_bytes,
    }


def _run_worker(path: Path) -> int:
    payload = json.loads(path.read_text())
    record = BuildRecord(**payload["record"])
    shape = tuple(int(x) for x in payload["shape"])
    problem = PreparedProblem.create(
        shape, int(payload["padded_n"]), with_reference=False
    )
    try:
        result = benchmark_record(
            problem,
            record,
            arch=str(payload["arch"]),
            warmup=int(payload["warmup"]),
            iters=int(payload["iters"]),
            attempts=1,
        )
    finally:
        problem.close()
    print("ROCKE_SWEEP_WORKER_RESULT=" + json.dumps(result, sort_keys=True))
    return 0 if "median_ms" in result else 1


def _build_and_benchmark(
    specs: Sequence[UniversalGemmSpec],
    *,
    cache_dir: Path,
    shape: Tuple[int, int, int],
    padded_n: int,
    arch: str,
    isa: str,
    workers: Optional[int],
    warmup: int,
    iters: int,
    attempts: int,
) -> Tuple[List[BuildRecord], List[Dict[str, Any]], PreparedProblem]:
    records = build_all_instances(
        specs,
        cache_dir=cache_dir,
        arch=arch,
        isa=isa,
        parallel=workers,
    )
    problem = PreparedProblem.create(shape, padded_n, with_reference=True)
    results = []
    for index, record in enumerate(records, 1):
        result = benchmark_record(
            problem,
            record,
            arch=arch,
            warmup=warmup,
            iters=iters,
            attempts=attempts,
        )
        results.append(result)
        if index % 25 == 0 or index == len(records):
            ranked = rank_results(results)
            best = (
                f"{ranked[0]['median_ms']:.6f} ms, "
                f"{ranked[0]['tflops']:.3f} TFLOP/s"
                if ranked
                else "none"
            )
            print(f"[{index}/{len(records)}] best={best}", flush=True)
    return records, results, problem


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="JSON file containing stages, search spaces, and benchmark settings",
    )
    parser.add_argument("--worker-record", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_record:
        return _run_worker(args.worker_record)

    config = load_config(args.config)
    target = config["target"]
    problem = config["problem"]
    selection = config["selection"]
    benchmark = config["benchmark"]
    stages = {"tile", "trait", "final"}
    arch = str(target["arch"])
    isa = str(target["isa"])
    shape = (int(problem["m"]), int(problem["n"]), int(problem["k"]))
    output_dir = Path(config["output_dir"])
    cache_dir = output_dir / "hsaco_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "arch": arch,
        "dtype": target["dtype"],
        "layout": target["layout"],
        "shape": list(shape),
        "config": str(args.config.resolve()),
    }

    if "tile" in stages:
        tile_specs = enumerate_tile_configs(config)
        padded_n = required_padded_n(shape[1], tile_specs)
        print(
            f"tile: {len(tile_specs)} candidates; B rows {shape[1]}->{padded_n}",
            flush=True,
        )
        tile_records, tile_results, tile_problem = _build_and_benchmark(
            tile_specs,
            cache_dir=cache_dir,
            shape=shape,
            padded_n=padded_n,
            arch=arch,
            isa=isa,
            workers=benchmark["workers"],
            warmup=benchmark["warmup"],
            iters=benchmark["iters"],
            attempts=benchmark["attempts"],
        )
        try:
            tile_record_map = _record_by_id(tile_records, arch=arch)
            for result in rank_results(tile_results)[
                : int(selection["tile_finalists"])
            ]:
                verify_record(
                    tile_problem,
                    tile_record_map[result["id"]],
                    result,
                    tolerance=float(selection["tolerance"]),
                )
        finally:
            tile_problem.close()
        _write_results(output_dir, "tile", tile_results)
        finalists = [
            _spec_from_dict(result["spec"])
            for result in rank_results(tile_results)
            if result.get("verified")
        ][: int(selection["tile_finalists"])]
        if not finalists:
            raise RuntimeError("no tile finalist passed verification")
        summary["tile"] = {
            "candidate_count": len(tile_specs),
            "verified_finalists": len(finalists),
            "best": rank_results(tile_results)[0],
        }
    else:
        finalists, tile_results = _load_tile_finalists(
            output_dir, int(selection["tile_finalists"])
        )
        padded_n = required_padded_n(shape[1], enumerate_tile_configs(config))
        summary["tile"] = {
            "candidate_count": len(tile_results),
            "verified_finalists": len(finalists),
            "best": rank_results(tile_results)[0],
        }

    if "trait" in stages:
        trait_specs = enumerate_trait_configs(config, finalists)
        print(f"trait: {len(trait_specs)} candidates", flush=True)
        trait_records, trait_results, trait_problem = _build_and_benchmark(
            trait_specs,
            cache_dir=cache_dir,
            shape=shape,
            padded_n=padded_n,
            arch=arch,
            isa=isa,
            workers=benchmark["workers"],
            warmup=benchmark["warmup"],
            iters=benchmark["iters"],
            attempts=benchmark["attempts"],
        )
        trait_problem.close()
        _write_results(output_dir, "trait", trait_results)
        summary["trait"] = {
            "candidate_count": len(trait_specs),
            "best_screening_result": rank_results(trait_results)[0],
        }
    elif "final" in stages:
        trait_records, trait_results = _load_trait_results(output_dir)
        summary["trait"] = {
            "candidate_count": len(trait_results),
            "best_screening_result": rank_results(trait_results)[0],
        }

    if "final" in stages:
        final_problem = PreparedProblem.create(shape, padded_n, with_reference=True)
        trait_record_map = _record_by_id(trait_records, arch=arch)
        robust_finalists = []
        try:
            for result in rank_results(trait_results):
                if not verify_record(
                    final_problem,
                    trait_record_map[result["id"]],
                    result,
                    tolerance=float(selection["tolerance"]),
                ):
                    continue
                record = trait_record_map[result["id"]]
                samples = _independent_final_timings(
                    Path(__file__).resolve(),
                    output_dir,
                    record,
                    shape,
                    padded_n,
                    arch=arch,
                    warmup=int(benchmark["warmup"]),
                    iters=int(benchmark["iters"]),
                )
                final_ms = statistics.median(samples)
                m, n, k = shape
                result["independent_samples_ms"] = samples
                result["final_median_ms"] = final_ms
                result["final_tflops"] = 2.0 * m * n * k / (final_ms * 1.0e9)
                robust_finalists.append(result)
                if len(robust_finalists) >= int(selection["final_timed"]):
                    break
        finally:
            final_problem.close()
        _write_results(output_dir, "trait", trait_results)
        if not robust_finalists:
            raise RuntimeError("no trait candidate passed final verification")

        winner = min(
            robust_finalists, key=lambda result: float(result["final_median_ms"])
        )
        winner["uncached_build"] = measure_uncached_build(
            _spec_from_dict(winner["spec"]), arch=arch
        )
        summary["final"] = {
            "robust_finalist_count": len(robust_finalists),
            "winner": winner,
        }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    if "final" in summary:
        winner = summary["final"]["winner"]
        print(
            "winner: "
            f"{winner['final_median_ms']:.6f} ms, "
            f"{winner['final_tflops']:.3f} TFLOP/s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
