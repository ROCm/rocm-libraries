#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""HSTU dispatcher Python utilities (ctypes, in-process)."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from dispatcher_common import detect_gpu_arch, get_dispatcher_root
except ImportError:

    def get_dispatcher_root() -> Path:
        return Path(__file__).parent.parent

    def detect_gpu_arch(fallback: str = "gfx950") -> str:
        try:
            out = subprocess.check_output(
                ["rocminfo"], text=True, stderr=subprocess.DEVNULL
            )
            for line in out.splitlines():
                if "Name:" in line and "gfx" in line:
                    return line.split()[-1].strip()
        except Exception:
            pass
        return fallback


@dataclass
class HstuResult:
    success: bool
    output: Optional[np.ndarray] = None
    time_ms: float = 0.0
    tflops: float = 0.0
    tflops_genrec: float = 0.0
    error: str = ""


@dataclass
class HstuProblem:
    batch: int = 256
    num_head: int = 4
    hdim_qk: int = 128
    hdim_v: int = 128
    max_seqlen_q: int = 4096
    total_tokens: int = 0
    use_causal: bool = True
    window_size: int = 0
    contextual_seqlen: int = 0
    min_full_attn_seqlen: int = 0
    target_size: int = 0
    data_type: str = "bf16"

    @property
    def scale_s(self) -> float:
        return 1.0 / (self.hdim_qk**0.5)

    @property
    def attn_scale(self) -> float:
        return 1.0 / float(self.max_seqlen_q)

    @property
    def num_ops(self) -> int:
        b, h, n, d_qk, d_v = (
            self.batch,
            self.num_head,
            self.max_seqlen_q,
            self.hdim_qk,
            self.hdim_v,
        )
        return 2 * b * h * n * n * (d_qk + d_v)


@dataclass
class HstuKernelConfig:
    name: str = "jagged_fwd"
    data_type: str = "bf16"
    use_causal: bool = True
    max_k: int = 128
    mtile: int = 128
    use_splitkv: bool = False
    disable_splitkv: int = 1
    gfx_arch: str = "gfx950"
    # Block-tile shape overrides for sequence<kM0,kN0,kN0Sub,kN1,kK1,kQKHeaddim>.
    # 0 == "use the base dim" from HstuAttentionNoSoftmaxFwdBlockTile, so a config
    # that leaves these at 0 generates a kernel byte-identical to the legacy
    # 5-axis (data_type/use_causal/max_k/mtile/use_splitkv) sweep. A nonzero value
    # pins that tile dim through the dispatch template overrides.
    km0: int = 0
    kn0: int = 0
    kn0sub: int = 0
    kn1: int = 0
    kk1: int = 0
    # Warp-K of the 16x16x{K} bf16 MFMA family. 0 == "use the dispatch default"
    # (WarpK=16 -> 16x16x16); a nonzero value pins it (32 -> 16x16x32). Threaded
    # through the same byte-identical-base discipline as the tile fields.
    warp_k: int = 0

    def to_codegen_json(self) -> str:
        algorithm = {"mtile": self.mtile}
        # Only emit tile fields when overridden so the codegen json (and thus the
        # generated kernel name / cpp) stays identical for base-tile configs.
        for key, val in (
            ("km0", self.km0),
            ("kn0", self.kn0),
            ("kn0sub", self.kn0sub),
            ("kn1", self.kn1),
            ("kk1", self.kk1),
            ("warp_k", self.warp_k),
        ):
            if val:
                algorithm[key] = val
        return json.dumps(
            {
                "arch": self.gfx_arch,
                "signature": {
                    "family": "jagged_fwd",
                    "data_type": self.data_type,
                    "use_causal": self.use_causal,
                    "max_k": self.max_k,
                    "use_splitkv": self.use_splitkv,
                },
                "algorithm": algorithm,
            }
        )


@dataclass
class HstuSetupResult:
    success: bool
    config: Optional[HstuKernelConfig] = None
    runner: Optional["HstuRunner"] = None
    library_path: str = ""
    error: str = ""
    build_time_s: float = 0.0


def hstu_flops_genrec(
    batch: int,
    seq_offsets: np.ndarray,
    num_head: int,
    hdim_qk: int,
    hdim_v: int,
    mode: str = "fwd",
) -> float:
    """Jagged sum s_i^2 FLOPs (mvonstra recsys_harness/common.hstu_flops)."""
    f1 = 0.0
    f2 = 0.0
    for i in range(batch):
        s = int(seq_offsets[i + 1] - seq_offsets[i])
        f1 += 2 * num_head * hdim_qk * (s**2) // 2
        f2 += 2 * num_head * hdim_v * (s**2) // 2
    if mode == "fwd":
        return f1 + f2
    if mode == "bwd":
        return 3 * f1 + 2 * f2
    return 4 * f1 + 3 * f2


class HstuDispatcherLib:
    _SEARCH = (
        "build/examples/libdispatcher_hstu_lib.so",
        "dispatcher/build/examples/libdispatcher_hstu_lib.so",
        "build/libdispatcher_hstu_lib.so",
    )

    def __init__(self, lib_path: Optional[Path] = None):
        root = get_dispatcher_root()
        path = lib_path
        if path is None:
            for rel in self._SEARCH:
                cand = root.parent / rel if "dispatcher/" in rel else root / rel
                if cand.exists():
                    path = cand
                    break
        if path is None or not path.exists():
            raise FileNotFoundError(
                "libdispatcher_hstu_lib.so not found; build dispatcher examples "
                "(cmake -S dispatcher -B dispatcher/build && make -C dispatcher/build hstu_python_libs)"
            )
        self._lib = ctypes.CDLL(str(path))
        self._path = path
        self._setup_signatures()

    @classmethod
    def load(cls, path: str | Path) -> "HstuDispatcherLib":
        return cls(Path(path))

    def _setup_signatures(self) -> None:
        L = self._lib
        L.hstu_dispatcher_initialize.argtypes = [ctypes.c_char_p]
        L.hstu_dispatcher_initialize.restype = ctypes.c_int
        L.hstu_dispatcher_cleanup.argtypes = []
        L.hstu_dispatcher_cleanup.restype = None
        L.hstu_dispatcher_run_jagged_fwd.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_float),
        ]
        L.hstu_dispatcher_run_jagged_fwd.restype = ctypes.c_int

    def initialize(self, arch: Optional[str] = None) -> None:
        arch_b = (arch or detect_gpu_arch()).encode()
        if self._lib.hstu_dispatcher_initialize(arch_b) != 0:
            raise RuntimeError("hstu_dispatcher_initialize failed")

    def cleanup(self) -> None:
        self._lib.hstu_dispatcher_cleanup()


class HstuRunner:
    def __init__(self, lib: HstuDispatcherLib, config: Optional[HstuKernelConfig] = None):
        self.lib = lib
        self.config = config
        lib.initialize()

    @classmethod
    def from_prebuilt(cls, lib_path: Optional[Path] = None) -> "HstuRunner":
        return cls(HstuDispatcherLib(lib_path))

    @classmethod
    def from_library(cls, path: str, arch: Optional[str] = None) -> "HstuRunner":
        del arch
        return cls(HstuDispatcherLib.load(path))

    def run(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        seq_offsets: np.ndarray,
        problem: HstuProblem,
        config: Optional[HstuKernelConfig] = None,
        num_targets: Optional[np.ndarray] = None,
    ) -> HstuResult:
        kcfg = config or self.config or HstuKernelConfig()
        o = np.empty_like(v)
        time_ms = ctypes.c_float(0.0)
        nt_ptr = None
        if num_targets is not None and num_targets.size > 0:
            nt = np.ascontiguousarray(num_targets.astype(np.int32))
            nt_ptr = nt.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        off = np.ascontiguousarray(seq_offsets.astype(np.int32))
        rc = self.lib._lib.hstu_dispatcher_run_jagged_fwd(
            q.ctypes.data_as(ctypes.c_void_p),
            k.ctypes.data_as(ctypes.c_void_p),
            v.ctypes.data_as(ctypes.c_void_p),
            o.ctypes.data_as(ctypes.c_void_p),
            off.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            nt_ptr,
            problem.batch,
            problem.num_head,
            problem.hdim_qk,
            problem.hdim_v,
            problem.max_seqlen_q,
            problem.total_tokens,
            int(problem.use_causal),
            problem.window_size,
            problem.contextual_seqlen,
            problem.min_full_attn_seqlen,
            kcfg.mtile,
            int(kcfg.use_splitkv),
            int(kcfg.disable_splitkv),
            problem.scale_s,
            problem.attn_scale,
            kcfg.data_type.encode(),
            ctypes.byref(time_ms),
        )
        if rc != 0:
            return HstuResult(False, error=f"hstu_dispatcher_run_jagged_fwd rc={rc}")
        t_ms = float(time_ms.value)
        genrec_flops = hstu_flops_genrec(
            problem.batch, off, problem.num_head, problem.hdim_qk, problem.hdim_v
        )
        tflops = problem.num_ops / (t_ms * 1e-3) / 1e12 if t_ms > 0 else 0.0
        tflops_genrec = genrec_flops / (t_ms * 1e-3) / 1e12 if t_ms > 0 else 0.0
        if problem.use_causal and t_ms > 0:
            causal_ratio = 0.5
            tflops = problem.num_ops * causal_ratio / (t_ms * 1e-3) / 1e12
        return HstuResult(
            True, output=o, time_ms=t_ms, tflops=tflops, tflops_genrec=tflops_genrec
        )

    def cleanup(self) -> None:
        self.lib.cleanup()


def build_jagged_problem(
    batch: int,
    num_head: int,
    hdim_qk: int,
    hdim_v: int,
    uih_lengths: List[int],
    num_targets: Optional[List[int]] = None,
    contextual_seqlen: int = 0,
    data_type: str = "bf16",
    use_causal: bool = True,
    window_size: int = 0,
) -> Tuple[HstuProblem, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build jagged Q/K/V and offsets matching mvonstra harness layout."""
    if num_targets is None:
        num_targets = [0] * batch
    eff = [uih_lengths[i] + num_targets[i] + contextual_seqlen for i in range(batch)]
    offsets = [0]
    for L in eff:
        offsets.append(offsets[-1] + L)
    total = offsets[-1]
    max_seq = max(eff)
    rng = np.random.default_rng(1001)
    if data_type == "bf16":
        q = rng.standard_normal((total, num_head, hdim_qk), dtype=np.float32).astype(np.float16)
        k = rng.standard_normal((total, num_head, hdim_qk), dtype=np.float32).astype(np.float16)
        v = rng.standard_normal((total, num_head, hdim_v), dtype=np.float32).astype(np.float16)
    else:
        q = rng.standard_normal((total, num_head, hdim_qk), dtype=np.float16)
        k = rng.standard_normal((total, num_head, hdim_qk), dtype=np.float16)
        v = rng.standard_normal((total, num_head, hdim_v), dtype=np.float16)
    prob = HstuProblem(
        batch=batch,
        num_head=num_head,
        hdim_qk=hdim_qk,
        hdim_v=hdim_v,
        max_seqlen_q=max_seq,
        total_tokens=total,
        use_causal=use_causal,
        window_size=window_size,
        contextual_seqlen=contextual_seqlen,
        target_size=max(num_targets) if num_targets else 0,
        data_type=data_type,
    )
    off = np.array(offsets, dtype=np.int32)
    nt = np.array(num_targets, dtype=np.int32)
    return prob, q, k, v, off, nt


def expand_sweep_from_json(config_path: Path, arch: str) -> List[HstuKernelConfig]:
    """Load sweep JSON via codegen instance_gen."""
    codegen_dir = get_dispatcher_root() / "codegen"
    sys.path.insert(0, str(codegen_dir))
    from hstu.instance_gen import expand_sweep  # noqa: WPS433

    return expand_sweep(config_path, arch)


def _find_hipcc() -> str:
    for path in ["/opt/rocm/bin/hipcc", "/usr/bin/hipcc"]:
        if os.path.exists(path):
            return path
    return "hipcc"


def _find_static_lib() -> Optional[Path]:
    root = get_dispatcher_root()
    for rel in ["build/libck_tile_dispatcher.a", "build/lib/libck_tile_dispatcher.a"]:
        p = root / rel
        if p.exists():
            return p
    return None


def hstu_compile_flags(arch: str, hipcc: str = "", use_splitkv: bool = False) -> List[str]:
    if not hipcc:
        hipcc = _find_hipcc()
    root = get_dispatcher_root()
    hstu_dir = root.parent / "example" / "ck_tile" / "53_hstu_attention"
    flags = [
        hipcc,
        "-c",
        "-fPIC",
        "-O3",
        "-DNDEBUG",
        f"--offload-arch={arch}",
        "-std=c++17",
        f"-I{root.parent / 'include'}",
        f"-I{root / 'include'}",
        f"-I{root.parent}",
        f"-I{hstu_dir}",
        "-Wno-undefined-func-template",
        "-Wno-float-equal",
        "-fgpu-flush-denormals-to-zero",
        "-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3",
    ]
    if not use_splitkv:
        flags.append("-DHSTU_COMPILE_NO_SPLITKV=1")
    if arch.startswith("gfx9"):
        flags.append("-DCK_USE_XDL")
    if arch.startswith("gfx950") or arch == "gfx95":
        flags.append("-DCK_GFX950_SUPPORT")
        flags.append("-DCK_USE_GFX950")
    return flags


def _run_compile_job(job):
    cmd, obj_str, name, label = job
    if os.path.exists(obj_str):
        return (name, True, "")
    err_path = obj_str + ".err"
    with open(err_path, "w") as ef:
        rc = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=ef)
    if rc != 0:
        try:
            err = open(err_path).read()[:200]
        except Exception:
            err = f"rc={rc}"
        return (name, False, err)
    try:
        os.unlink(err_path)
    except OSError:
        pass
    return (name, True, "")


def setup_multiple_hstu_dispatchers(
    configs: List[HstuKernelConfig],
    output_dir: Optional[Path] = None,
    verbose: bool = False,
    max_workers: Optional[int] = None,
    executor=None,
    progress_callback=None,
) -> List[HstuSetupResult]:
    """Pipelined JIT: codegen -> hipcc compile -> link per kernel config."""
    if not configs:
        return []

    root = get_dispatcher_root()
    codegen_dir = root / "codegen"
    ctypes_src = root / "bindings" / "ctypes" / "hstu_ctypes_lib.cpp"
    static_lib = _find_static_lib()
    hipcc = _find_hipcc()
    arch = configs[0].gfx_arch

    if output_dir is None:
        output_dir = root / "build" / "examples" / "hstu_jit"
    output_dir.mkdir(parents=True, exist_ok=True)

    if static_lib is None:
        return [
            HstuSetupResult(
                success=False, config=c, error="libck_tile_dispatcher.a not found"
            )
            for c in configs
        ]

    results: dict[str, HstuSetupResult] = {}

    def _codegen(cfg: HstuKernelConfig):
        out = output_dir / cfg.name
        lib_path = out / f"libdispatcher_hstu_{cfg.name}.so"
        if lib_path.exists():
            results[cfg.name] = HstuSetupResult(
                success=True, config=cfg, library_path=str(lib_path)
            )
            return (cfg.name, cfg, out, True)
        if out.exists() and not (out / "hstu_python_dispatch.hpp").exists():
            err_file = out / "_codegen_err.txt"
            if err_file.exists():
                results[cfg.name] = HstuSetupResult(
                    success=False, config=cfg, error="Codegen failed (cached)"
                )
                return (cfg.name, cfg, out, False)
        out.mkdir(parents=True, exist_ok=True)
        if (out / "hstu_python_dispatch.hpp").exists():
            return (cfg.name, cfg, out, True)
        err_file = out / "_codegen_err.txt"
        with open(err_file, "w") as ef:
            rc = subprocess.call(
                [
                    sys.executable,
                    str(codegen_dir / "hstu" / "generate_fallback.py"),
                    "--output-dir",
                    str(out),
                    "--gpu-target",
                    cfg.gfx_arch,
                    "--config-json",
                    cfg.to_codegen_json(),
                ],
                stdout=subprocess.DEVNULL,
                stderr=ef,
                cwd=str(codegen_dir),
            )
        ok = rc == 0 and (out / "hstu_python_dispatch.hpp").exists()
        if not ok:
            err_msg = err_file.read_text()[:200] if err_file.exists() else "unknown"
            results[cfg.name] = HstuSetupResult(
                success=False, config=cfg, error=f"Codegen failed: {err_msg}"
            )
        return (cfg.name, cfg, out, ok)

    codegen_results = []
    for i, cfg in enumerate(configs):
        codegen_results.append(_codegen(cfg))
        if progress_callback:
            progress_callback("codegen", i + 1, len(configs))

    compile_jobs = []
    config_dirs: dict[str, tuple[HstuKernelConfig, Path]] = {}
    failed_names: set = set()

    for name, cfg, out, ok in codegen_results:
        if not ok or name in results:
            continue
        config_dirs[name] = (cfg, out)
        base_flags = hstu_compile_flags(arch, hipcc, use_splitkv=cfg.use_splitkv)
        for cpp in out.glob("hstu_*.cpp"):
            obj = cpp.with_suffix(".o")
            if not obj.exists():
                compile_jobs.append(
                    (base_flags + [str(cpp), "-o", str(obj)], str(obj), name, "kernel")
                )
        ctypes_obj = out / "hstu_ctypes_lib.o"
        if not ctypes_obj.exists():
            dispatch = out / "hstu_python_dispatch.hpp"
            compile_jobs.append(
                (
                    base_flags
                    + [
                        f"-I{out}",
                        f"-include{dispatch}",
                        f'-DGFX_ARCH="{arch}"',
                        str(ctypes_src),
                        "-o",
                        str(ctypes_obj),
                    ],
                    str(ctypes_obj),
                    name,
                    "ctypes",
                )
            )

    if compile_jobs:
        _own_pool = None
        _pool = executor
        if _pool is None:
            workers = max_workers or min(len(compile_jobs), os.cpu_count() or 4)
            _own_pool = ProcessPoolExecutor(max_workers=workers)
            _pool = _own_pool
        try:
            done_count = 0
            total_jobs = len(compile_jobs)
            for name, ok, err in _pool.map(_run_compile_job, compile_jobs):
                done_count += 1
                if progress_callback:
                    progress_callback("compile", done_count, total_jobs)
                if not ok:
                    failed_names.add(name)
                    if name not in results:
                        cfg, _ = config_dirs[name]
                        results[name] = HstuSetupResult(
                            success=False, config=cfg, error=f"Compile: {err}"
                        )
        finally:
            if _own_pool is not None:
                _own_pool.shutdown(wait=True)

    def _link(item):
        name, (cfg, out) = item
        if name in failed_names or name in results:
            return
        objs = list(out.glob("*.o"))
        lib_path = out / f"libdispatcher_hstu_{name}.so"
        if not lib_path.exists():
            r = subprocess.run(
                [
                    hipcc,
                    "-shared",
                    "-fPIC",
                    *[str(o) for o in objs],
                    str(static_lib),
                    "-o",
                    str(lib_path),
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                results[name] = HstuSetupResult(
                    success=False, config=cfg, error=f"Link: {r.stderr[:200]}"
                )
                return
        results[name] = HstuSetupResult(
            success=True, config=cfg, library_path=str(lib_path)
        )

    for item in config_dirs.items():
        _link(item)

    out_list = [
        results.get(c.name, HstuSetupResult(success=False, config=c, error="skipped"))
        for c in configs
    ]

    for s in out_list:
        if s.success and s.library_path and s.runner is None:
            try:
                s.runner = HstuRunner.from_library(s.library_path, arch)
                s.runner.config = s.config
            except Exception as e:
                s.success = False
                s.error = f"Load failed: {e}"

    if verbose:
        built = sum(1 for s in out_list if s.success)
        print(f"HSTU JIT: built {built}/{len(configs)}")

    return out_list


def default_kernel_configs(data_type: str = "bf16") -> List[HstuKernelConfig]:
    """Legacy prebuilt-lib sweep (mtile env override). Prefer expand_sweep_from_json."""
    configs = []
    for mtile in (0, 64, 128):
        configs.append(
            HstuKernelConfig(
                name=f"{data_type}_mtile{mtile}",
                data_type=data_type,
                mtile=mtile,
                use_splitkv=False,
                disable_splitkv=1,
            )
        )
    return configs
