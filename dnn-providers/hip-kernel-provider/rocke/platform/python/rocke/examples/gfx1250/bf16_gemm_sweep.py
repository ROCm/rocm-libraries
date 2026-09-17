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
import tempfile
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
    universal_gemm_grid,
)
from rocke.helpers import compile_kernel
from rocke.runtime.hip_module import HipError, Runtime
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
    tdm: bool = False,
    tdm_depth: int = 1,
    persistent_ctas: int = 0,
) -> UniversalGemmSpec:
    target = config["target"]
    problem = config["problem"]
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
            # A guard is dead code when the tile grid lands exactly on the
            # extent, and ``pad_n`` forfeits the cshuffle wide store, so each
            # one follows the shape rather than being forced on.
            pad_m=int(problem["m"]) % tile_m != 0,
            pad_n=int(problem["n"]) % tile_n != 0,
            pad_k=int(problem["k"]) % tile_k != 0,
            waves_per_eu=waves_per_eu,
            lds_swizzle=lds_swizzle,
            lds_k_pad=lds_k_pad,
            direct_to_lds=direct_to_lds,
            dtl_prefetch=dtl_prefetch,
            tdm=tdm,
            tdm_depth=tdm_depth,
            persistent=persistent_ctas > 0,
            persistent_ctas=persistent_ctas,
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


def _load_paths(traits: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The global->LDS mechanisms to try, as trait overrides.

    The three paths are mutually exclusive, so they are enumerated as a list
    rather than multiplied out: pairing ``dtl_prefetch`` with
    ``direct_to_lds=False``, or ``tdm_depth`` with ``tdm=False``, would inflate
    the grid with combinations ``is_valid_spec`` rejects anyway.
    """
    paths: List[Dict[str, Any]] = []
    for direct_to_lds in traits.get("direct_to_lds", [False]):
        for dtl_prefetch in traits.get("dtl_prefetch", [False]):
            if dtl_prefetch and not direct_to_lds:
                continue
            paths.append(
                {
                    "direct_to_lds": direct_to_lds,
                    "dtl_prefetch": dtl_prefetch,
                    "tdm": False,
                    "tdm_depth": 1,
                }
            )
    for tdm in traits.get("tdm", [False]):
        if not tdm:
            continue
        for depth in traits.get("tdm_depth", [1]):
            paths.append(
                {
                    "direct_to_lds": False,
                    "dtl_prefetch": False,
                    "tdm": True,
                    "tdm_depth": int(depth),
                }
            )
    return paths


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
                                for path in _load_paths(traits):
                                    for pers in traits.get("persistent_ctas", [0]):
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
                                                    persistent=int(pers) > 0,
                                                    persistent_ctas=int(pers),
                                                    **path,
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


def _float32_to_fp16(np, values):
    return (
        np.ascontiguousarray(values, dtype=np.float32)
        .astype(np.float16)
        .view(np.uint16)
    )


def _fp16_to_float32(np, values):
    return np.ascontiguousarray(values, dtype=np.uint16).view(np.float16).astype(
        np.float32
    )


# The device buffers are raw u16 either way, so only the encode/decode of the
# 16-bit element differs between the two operand types the sweep supports.
_ENCODE_16 = {"bf16": _float32_to_bf16, "fp16": _float32_to_fp16}
_DECODE_16 = {"bf16": _bf16_to_float32, "fp16": _fp16_to_float32}


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
    dtype: str = "bf16"

    @classmethod
    def create(
        cls,
        shape: Tuple[int, int, int],
        padded_n: int,
        *,
        with_reference: bool,
        dtype: str = "bf16",
    ) -> "PreparedProblem":
        import numpy as np

        if dtype not in _ENCODE_16:
            raise ValueError(
                f"sweep host buffers support {sorted(_ENCODE_16)}, got {dtype!r}"
            )
        encode = _ENCODE_16[dtype]
        decode = _DECODE_16[dtype]
        m, n, k = shape
        rng = np.random.default_rng(0xC0FFEE)
        a_values = rng.integers(-5, 6, size=(m, k), dtype=np.int16)
        b_values = rng.integers(-5, 6, size=(n, k), dtype=np.int16)
        a = encode(np, a_values)
        b = np.zeros((padded_n, k), dtype=np.uint16)
        b[:n] = encode(np, b_values)
        c = np.empty((m, n), dtype=np.uint16)
        reference = None
        if with_reference:
            a_f32 = decode(np, a)
            b_f32 = decode(np, b[:n])
            reference = decode(np, encode(np, a_f32 @ b_f32.T))

        rt = Runtime()
        a_dev = rt.alloc(a.nbytes)
        b_dev = rt.alloc(b.nbytes)
        c_dev = rt.alloc(c.nbytes)
        rt.memcpy_h2d(a_dev, _as_u8_buffer(a), a.nbytes)
        rt.memcpy_h2d(b_dev, _as_u8_buffer(b), b.nbytes)
        rt.memset(c_dev, 0, c.nbytes)
        return cls(rt, shape, padded_n, a_dev, b_dev, c_dev, c, reference, dtype)

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
    return universal_gemm_grid(spec, m, n), (spec.block_size, 1, 1)


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


# HIP error codes that latch onto the context instead of being returned once:
# an illegal access (700), a device-side abort (710) and an unspecified launch
# failure (719) all leave every later call in the process returning the same
# code. Recovery needs a fresh process, which is what ``_benchmark_records``
# provides by keeping the launch loop in a respawnable child.
_STICKY_HIP_ERRORS = ("hipError(700)", "hipError(710)", "hipError(719)")

# Exit code a benchmark worker uses to tell the parent "I hit a sticky error
# and cannot run anything else"; distinct from a crash or a clean finish.
_WORKER_FAULT_EXIT = 70
_WORKER_RESULT_PREFIX = "ROCKE_SWEEP_WORKER_RESULT="

# A worker that dies before timing anything is normally a faulting candidate,
# but it is also what a broken environment looks like. Give up rather than
# respawn once per remaining candidate.
_MAX_BARREN_SPAWNS = 5


def _is_device_fault(exc: BaseException) -> bool:
    return isinstance(exc, HipError) and any(
        code in str(exc) for code in _STICKY_HIP_ERRORS
    )


def _note_failure(result: Dict[str, Any], exc: BaseException) -> None:
    """Fold ``exc`` into ``result`` without overwriting an earlier diagnosis."""
    result.setdefault("error", f"{type(exc).__name__}: {exc}")
    if _is_device_fault(exc):
        result["device_fault"] = True


def _result_header(
    record: BuildRecord, spec: UniversalGemmSpec, *, arch: str
) -> Dict[str, Any]:
    """The build-side half of a result row, before any timing is attempted.

    Split out so a candidate that takes its benchmark process down with it can
    still be reported with the same shape as one that ran.
    """
    return {
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
    result = _result_header(record, spec, arch=arch)
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
        _note_failure(result, exc)
    finally:
        # A sticky error is re-raised by every later HIP call, so an unguarded
        # drain here raises a *second* exception out of the ``finally`` and
        # discards the first. That is why a faulting sweep used to die with a
        # traceback through ``wait_stream`` that never named the candidate
        # responsible. Cleanup is best-effort; the diagnosis stays in
        # ``result`` so the caller can attribute and report the fault.
        try:
            problem.rt.wait_stream(0)
        except Exception as exc:
            _note_failure(result, exc)
        if module is not None:
            try:
                module.unload()
            except Exception as exc:
                _note_failure(result, exc)
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
        actual = _DECODE_16[problem.dtype](np, problem.c_host)
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
    dtype: str = "bf16",
) -> List[float]:
    payload = {
        "record": asdict(record),
        "shape": list(shape),
        "padded_n": padded_n,
        "arch": arch,
        "warmup": warmup,
        "iters": iters,
        "dtype": dtype,
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
        shape,
        int(payload["padded_n"]),
        with_reference=False,
        dtype=str(payload.get("dtype", "bf16")),
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
    print(_WORKER_RESULT_PREFIX + json.dumps(result, sort_keys=True))
    return 0 if "median_ms" in result else 1


def _run_worker_range(path: Path) -> int:
    """Benchmark prebuilt records from ``start`` onwards, streaming results.

    Every finished candidate is flushed to stdout as soon as it has a number,
    so the parent keeps all of them even when a fault takes this process down
    mid-flight. The first candidate with no line of its own is the culprit.
    """
    payload = json.loads(path.read_text())
    records = [BuildRecord(**entry) for entry in payload["records"]]
    problem = PreparedProblem.create(
        tuple(int(x) for x in payload["shape"]),
        int(payload["padded_n"]),
        with_reference=False,
        dtype=str(payload.get("dtype", "bf16")),
    )
    status = 0
    try:
        for index in range(int(payload["start"]), len(records)):
            result = benchmark_record(
                problem,
                records[index],
                arch=str(payload["arch"]),
                warmup=int(payload["warmup"]),
                iters=int(payload["iters"]),
                attempts=int(payload.get("attempts", 1)),
            )
            print(
                _WORKER_RESULT_PREFIX
                + json.dumps({"index": index, "result": result}, sort_keys=True),
                flush=True,
            )
            if result.get("device_fault"):
                # The context is poisoned; nothing after this would be timed
                # correctly. Hand back to the parent for a fresh process.
                status = _WORKER_FAULT_EXIT
                break
    finally:
        try:
            problem.close()
        except Exception:
            pass
    return status


def _worker_env() -> Dict[str, str]:
    """Child environment that can import ``rocke`` however the parent did."""
    env = dict(os.environ)
    roots = [p for p in sys.path if p]
    existing = env.get("PYTHONPATH")
    if existing:
        roots.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(roots))
    return env


def _benchmark_records(
    records: Sequence[BuildRecord],
    *,
    shape: Tuple[int, int, int],
    padded_n: int,
    arch: str,
    warmup: int,
    iters: int,
    attempts: int,
    dtype: str,
) -> List[Dict[str, Any]]:
    """Benchmark every record, surviving candidates that fault the device.

    An illegal access latches onto the HIP context, so one bad candidate makes
    every later launch in the same process fail too -- which is how a single
    fault used to abort a whole stage and discard hundreds of good results.
    The launch loop therefore runs in a child process: when a candidate
    poisons the context, or takes the child down outright, the parent names
    that candidate, respawns from the next index, and the stage still
    finishes. Keeping the launches out of the parent also leaves the parent's
    own context clean for the verification pass that follows.
    """
    results: List[Optional[Dict[str, Any]]] = [None] * len(records)
    if not records:
        return []
    payload = {
        "records": [asdict(record) for record in records],
        "shape": list(shape),
        "padded_n": padded_n,
        "arch": arch,
        "warmup": warmup,
        "iters": iters,
        "attempts": attempts,
        "dtype": dtype,
    }
    env = _worker_env()
    next_index = 0
    barren_spawns = 0
    with tempfile.TemporaryDirectory(prefix="rocke_sweep_") as tmp:
        spec_path = Path(tmp) / "range.json"
        while next_index < len(records):
            payload["start"] = next_index
            spec_path.write_text(json.dumps(payload))
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-range",
                    str(spec_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            highest = next_index - 1
            tail: List[str] = []
            assert proc.stdout is not None
            for line in proc.stdout:
                if not line.startswith(_WORKER_RESULT_PREFIX):
                    sys.stderr.write(line)
                    tail = (tail + [line.rstrip()])[-20:]
                    continue
                streamed = json.loads(line[len(_WORKER_RESULT_PREFIX) :])
                index = int(streamed["index"])
                results[index] = streamed["result"]
                highest = max(highest, index)
                _print_progress(index + 1, len(records), results)
            code = proc.wait()

            barren_spawns = 0 if highest >= next_index else barren_spawns + 1
            if barren_spawns >= _MAX_BARREN_SPAWNS:
                raise RuntimeError(
                    f"benchmark worker produced no results in "
                    f"{barren_spawns} consecutive attempts (exit {code}); "
                    "last output:\n" + "\n".join(tail)
                )

            resume = highest + 1
            if code == _WORKER_FAULT_EXIT and highest >= next_index:
                print(
                    f"  device fault on [{highest}] {records[highest].name}; "
                    "restarting the benchmark worker",
                    flush=True,
                )
            elif code != 0 and resume < len(records):
                # The child died without reporting, so the candidate it had in
                # flight is the one after the last streamed result.
                victim = records[resume]
                result = _result_header(
                    victim, _spec_from_dict(victim.spec_dict), arch=arch
                )
                result["error"] = f"benchmark worker died (exit {code})"
                result["device_fault"] = True
                results[resume] = result
                print(
                    f"  benchmark worker died (exit {code}) on "
                    f"[{resume}] {victim.name}; restarting past it",
                    flush=True,
                )
                resume += 1
            next_index = resume
    return [result for result in results if result is not None]


def _print_progress(
    done: int, total: int, results: Sequence[Optional[Dict[str, Any]]]
) -> None:
    if done % 25 and done != total:
        return
    ranked = rank_results([r for r in results if r is not None])
    best = (
        f"{ranked[0]['median_ms']:.6f} ms, {ranked[0]['tflops']:.3f} TFLOP/s"
        if ranked
        else "none"
    )
    print(f"[{done}/{total}] best={best}", flush=True)


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
    dtype: str = "bf16",
    with_reference: bool = True,
) -> Tuple[List[BuildRecord], List[Dict[str, Any]], PreparedProblem]:
    records = build_all_instances(
        specs,
        cache_dir=cache_dir,
        arch=arch,
        isa=isa,
        parallel=workers,
    )
    results = _benchmark_records(
        records,
        shape=shape,
        padded_n=padded_n,
        arch=arch,
        warmup=warmup,
        iters=iters,
        attempts=attempts,
        dtype=dtype,
    )
    faulted = [r for r in results if r.get("device_fault")]
    if faulted:
        print(
            f"{len(faulted)} candidate(s) faulted the device and were skipped:",
            flush=True,
        )
        for result in faulted:
            print(f"  {result['name']}: {result['error']}", flush=True)
    # Created after the launch loop, which ran entirely in child processes, so
    # this context has never seen a faulting kernel and is safe to verify on.
    problem = PreparedProblem.create(
        shape, padded_n, with_reference=with_reference, dtype=dtype
    )
    return records, results, problem


def _csv(cast):
    """A comma-separated list of ``cast``, for the list-valued config keys."""

    def parse(text: str) -> List[Any]:
        return [cast(part.strip()) for part in text.split(",") if part.strip()]

    return parse


def _opt_int(text: str) -> Optional[int]:
    """``waves_per_eu`` accepts ``null`` to mean "leave it to the backend"."""
    return None if text.lower() in ("null", "none") else int(text)


def _flag(text: str) -> bool:
    lowered = text.lower()
    if lowered in ("true", "1", "yes", "on"):
        return True
    if lowered in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {text!r}")


# ``--config`` supplies the baseline and every entry here is a flag that
# overrides one key in it, so a sweep can be driven entirely from the command
# line without authoring a JSON file. Adding a knob is one line.
_CONFIG_OVERRIDES: Tuple[Tuple[str, Tuple[str, ...], Any, str], ...] = (
    ("--arch", ("target", "arch"), str, "target architecture"),
    ("--isa", ("target", "isa"), str, "LLVM target triple"),
    ("--dtype", ("target", "dtype"), str, "operand dtype: bf16 or fp16"),
    ("--layout", ("target", "layout"), str, "operand layout, e.g. RCR"),
    ("--wave-size", ("target", "wave_size"), int, "lanes per wave"),
    ("--warp-tile", ("target", "warp_tile"), _csv(int), "MMA atom as m,n,k"),
    ("--m", ("problem", "m"), int, "problem M extent"),
    ("--n", ("problem", "n"), int, "problem N extent"),
    ("--k", ("problem", "k"), int, "problem K extent"),
    ("--tile-m", ("tile_config", "tile_m"), _csv(int), "block M tiles"),
    ("--tile-n", ("tile_config", "tile_n"), _csv(int), "block N tiles"),
    ("--tile-k", ("tile_config", "tile_k"), _csv(int), "block K tiles"),
    ("--warp-m", ("tile_config", "warp_m"), _csv(int), "waves along M"),
    ("--warp-n", ("tile_config", "warp_n"), _csv(int), "waves along N"),
    ("--pipelines", ("trait_config", "pipelines"), _csv(str), "pipeline names"),
    ("--schedulers", ("trait_config", "schedulers"), _csv(str), "scheduler names"),
    ("--epilogues", ("trait_config", "epilogues"), _csv(str), "epilogue names"),
    (
        "--waves-per-eu",
        ("trait_config", "waves_per_eu"),
        _csv(_opt_int),
        "occupancy hints; 'null' keeps the backend default",
    ),
    ("--lds-swizzle", ("trait_config", "lds_swizzle"), _csv(_flag), "LDS XOR swizzle"),
    ("--lds-k-pad", ("trait_config", "lds_k_pad"), _csv(int), "LDS row pad elements"),
    (
        "--direct-to-lds",
        ("trait_config", "direct_to_lds"),
        _csv(_flag),
        "DirectToLDS load path",
    ),
    (
        "--dtl-prefetch",
        ("trait_config", "dtl_prefetch"),
        _csv(_flag),
        "DirectToLDS prefetch ping-pong",
    ),
    ("--tdm", ("trait_config", "tdm"), _csv(_flag), "tensor-descriptor mover path"),
    ("--tdm-depth", ("trait_config", "tdm_depth"), _csv(int), "TDM pipeline depth"),
    (
        "--persistent-ctas",
        ("trait_config", "persistent_ctas"),
        _csv(int),
        "persistent CTA counts to try (0 = problem-sized grid)",
    ),
    ("--tile-finalists", ("selection", "tile_finalists"), int, "tiles kept after screening"),
    ("--final-timed", ("selection", "final_timed"), int, "candidates re-timed out-of-process"),
    ("--tolerance", ("selection", "tolerance"), float, "verification tolerance"),
    ("--workers", ("benchmark", "workers"), int, "parallel build workers"),
    ("--warmup", ("benchmark", "warmup"), int, "warmup launches per candidate"),
    ("--iters", ("benchmark", "iters"), int, "timed launches per sample"),
    ("--attempts", ("benchmark", "attempts"), int, "samples per candidate, median taken"),
    ("--output-dir", ("output_dir",), str, "where results and the HSACO cache go"),
)


def _apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    """Write every supplied flag into ``config``; return what was changed."""
    applied: List[str] = []
    for flag, path, _kind, _help in _CONFIG_OVERRIDES:
        value = getattr(args, flag.lstrip("-").replace("-", "_"))
        if value is None:
            continue
        section = config
        for key in path[:-1]:
            section = section.setdefault(key, {})
        section[path[-1]] = value
        applied.append(f"{'.'.join(path)}={value}")
    return applied


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="JSON file of stages, search spaces, and benchmark settings; every "
        "key can be overridden by the flags below",
    )
    for flag, path, kind, help_text in _CONFIG_OVERRIDES:
        parser.add_argument(
            flag,
            type=kind,
            default=None,
            help=f"{help_text} [overrides {'.'.join(path)}]",
        )
    parser.add_argument("--worker-record", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-range", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_record:
        return _run_worker(args.worker_record)
    if args.worker_range:
        return _run_worker_range(args.worker_range)

    config = load_config(args.config)
    overrides = _apply_overrides(config, args)
    if overrides:
        print("overrides: " + "  ".join(overrides), flush=True)
    target = config["target"]
    problem = config["problem"]
    selection = config["selection"]
    benchmark = config["benchmark"]
    stages = {"tile", "trait", "final"}
    arch = str(target["arch"])
    isa = str(target["isa"])
    dtype = str(target["dtype"])
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
        # The path alone cannot describe the run once flags override it, and
        # the file itself keeps changing between runs, so store what was used.
        "config_overrides": overrides,
        "resolved_config": config,
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
            dtype=dtype,
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
            dtype=dtype,
            with_reference=False,
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
        final_problem = PreparedProblem.create(
            shape, padded_n, with_reference=True, dtype=dtype
        )
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
                    dtype=dtype,
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
