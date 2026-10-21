#!/usr/bin/env python3
"""Five-way gfx942 attention_dense ablation using the CURRENT kernel's spec knobs.

No edited emitter or source duplication. Each variant is a separate compile-time
Gfx942AttentionDenseSpec, compiled before timing, with a direct KernelLauncher.

Method: one Python/PyTorch/ROCm process + stream, one Q/K/V input set, one
chunked-FP32 reference, pre-timing correctness for all 5, round-robin warmup,
5 randomized rounds x 50 launches, post-batch correctness of the last launch
in every timed batch, median-of-rounds, same-session ratios.

Absolute latency CSVs are local-only. Only relative results belong in qualifying
public case studies (subject to platform/AGENTS.md compliance review).
Static HSACO/ISA counts can be extracted separately with
bench_rocke_5way_static.py.

Default cohort: the twelve BF16 D128 causal shapes from the case-study table
(B=1/16, S=1024/2048/4096/8192/16384 where listed, Hq=32, Hkv=8/16),
with production dispatch resolving persistent mode and all non-ablation knobs.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    name: str
    use_cfvst: bool
    lds_num_buffers: int
    v_reg_prefetch: bool


VARIANTS = (
    Variant("baseline", False, 1, False),
    Variant("cfvst", True, 1, False),
    Variant("cfvst_klds", True, 2, False),
    Variant("cfvst_vpf", True, 1, True),
    Variant("cfvst_klds_vpf", True, 2, True),
)


@dataclass(frozen=True)
class Shape:
    batch: int
    seqlen: int
    hq: int
    hkv: int
    d: int
    dtype: str

    @property
    def label(self) -> str:
        return (
            f"b{self.batch}_s{self.seqlen}_hq{self.hq}_hkv{self.hkv}"
            f"_d{self.d}_{self.dtype}"
        )


# All twelve BF16 D128 shapes in the case-study performance table, in table
# order. Compile, validate, and time all five variants on EACH shape.
DEFAULT_SHAPES = (
    Shape(1, 1024, 32, 8, 128, "bf16"),
    Shape(1, 1024, 32, 16, 128, "bf16"),
    Shape(1, 2048, 32, 8, 128, "bf16"),
    Shape(1, 2048, 32, 16, 128, "bf16"),
    Shape(1, 4096, 32, 8, 128, "bf16"),
    Shape(1, 4096, 32, 16, 128, "bf16"),
    Shape(1, 8192, 32, 8, 128, "bf16"),
    Shape(1, 8192, 32, 16, 128, "bf16"),
    Shape(1, 16384, 32, 8, 128, "bf16"),
    Shape(16, 4096, 32, 8, 128, "bf16"),
    Shape(16, 4096, 32, 16, 128, "bf16"),
    Shape(16, 8192, 32, 8, 128, "bf16"),
)


def parse_shape(value: str) -> Shape:
    try:
        parts = [s.strip() for s in value.split(",")]
        if len(parts) != 6:
            raise ValueError("expected exactly six comma-separated entries")
        b, s, hq, hkv, d = (int(x) for x in parts[:5])
        dtype = parts[5].lower()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--shape must be B,S,Hq,Hkv,D,dtype, e.g. 1,4096,32,8,128,bf16"
        ) from exc
    # In the uploaded attention_dense.py, KDB/VPF are only valid for BF16 D128.
    if dtype != "bf16" or d != 128:
        raise argparse.ArgumentTypeError("five-way comparison requires bf16 D128")
    if min(b, s, hq, hkv) <= 0 or hq % hkv:
        raise argparse.ArgumentTypeError(
            "positive extents and Hq divisible by Hkv required"
        )
    if s % 256 or s % 64:
        raise argparse.ArgumentTypeError(
            "S must be divisible by block_m=256 and block_n=64"
        )
    return Shape(b, s, hq, hkv, d, dtype)


def locate_rocke(explicit: str | None) -> Path:
    starts = []
    if explicit:
        starts.append(Path(explicit).expanduser())
    starts.extend((Path.cwd(), Path(__file__).resolve().parent))
    for start in starts:
        for path in (start, *start.parents):
            for candidate in (path, path / "dnn-providers/hip-kernel-provider/rocke"):
                if (candidate / "library/kernels/gfx942/attention_dense.py").is_file():
                    return candidate.resolve()
    raise SystemExit("ROCKE root not found: pass --rocke-root")


def add_module_paths(rocke: Path) -> None:
    for p in (rocke / "library", rocke / "platform/python", rocke):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def fp32_reference(torch, q, k, v, *, scale: float, chunk: int):
    """Causal GQA SDPA in FP32, chunked by query rows to bound score memory."""
    b, sq, hq, d = q.shape
    _, sk, hkv, _ = k.shape
    gqa = hq // hkv
    qf = q.float().permute(0, 2, 1, 3).contiguous()
    kf = k.float().permute(0, 2, 1, 3).contiguous()
    vf = v.float().permute(0, 2, 1, 3).contiguous()
    result = torch.empty((b, hq, sq, d), device=q.device, dtype=torch.float32)
    key_indices = torch.arange(sk, device=q.device).view(1, 1, 1, sk)
    with torch.no_grad():
        for hkv_i in range(hkv):
            first, last = hkv_i * gqa, (hkv_i + 1) * gqa
            kt = kf[:, hkv_i : hkv_i + 1].transpose(-1, -2)
            val = vf[:, hkv_i : hkv_i + 1]
            for qs in range(0, sq, chunk):
                qe = min(qs + chunk, sq)
                scores = torch.matmul(qf[:, first:last, qs:qe], kt) * scale
                qi = torch.arange(qs, qe, device=q.device).view(1, 1, -1, 1)
                scores.masked_fill_(key_indices > qi, float("-inf"))
                probabilities = torch.softmax(scores, dim=-1)
                result[:, first:last, qs:qe] = torch.matmul(probabilities, val)
    return result.permute(0, 2, 1, 3).contiguous()


def validate(torch, actual, reference, *, atol: float, rtol: float):
    actual32 = actual.float()
    diff = (actual32 - reference).abs()
    max_abs = float(diff.max().item())
    max_rel = float((diff / reference.abs().clamp_min(1e-6)).max().item())
    # Also checks dtype-appropriately for NaNs, infinities and individual outliers.
    torch.testing.assert_close(actual32, reference, atol=atol, rtol=rtol)
    return max_abs, max_rel


def timed_batch(torch, launcher, values, config, stream, repeat: int) -> float:
    """Exactly N direct calls between GPU events, without compile/cache/wrapper."""
    first = torch.cuda.Event(enable_timing=True)
    last = torch.cuda.Event(enable_timing=True)
    first.record(stream)
    for _ in range(repeat):
        launcher(values, config=config)
    last.record(stream)
    last.synchronize()
    return float(first.elapsed_time(last)) / repeat


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _command_output(cmd: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT, timeout=12
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _package_version(package: str) -> str | None:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


def _git_provenance(root: Path) -> dict:
    repo = _command_output(["git", "rev-parse", "--show-toplevel"], cwd=root)
    head = _command_output(["git", "rev-parse", "HEAD"], cwd=root)
    branch = _command_output(["git", "branch", "--show-current"], cwd=root)
    status = _command_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=root
    )
    return {
        "repository_root": repo,
        "head_sha": head,
        "branch": branch,
        "working_tree_dirty": None if status is None else bool(status),
    }


def _require_clean_checkout(root: Path) -> dict:
    """Capture the measured Git state before creating outputs or compiling."""
    state = _git_provenance(root)
    if not state["repository_root"] or not state["head_sha"] or state["working_tree_dirty"] is None:
        raise SystemExit("Cannot verify Git HEAD/worktree; refusing benchmark")
    if state["working_tree_dirty"]:
        status = _command_output(
            ["git", "status", "--short", "--untracked-files=all"], cwd=root
        )
        raise SystemExit(
            "Benchmark requires a clean committed checkout. "
            "Commit/stash changes and remove untracked inputs first.\n"
            + (status or "Git status unavailable")
        )
    print(f"VERIFIED CLEAN checkout: {state['head_sha']}", flush=True)
    return state


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _loaded_hip_runtime_info() -> dict:
    """Identify the HIP runtime actually used by rocKE, without loading another.

    rocKE caches the ctypes handle in hip_module._resolve_hip(). Resolve the
    version symbol on THAT handle. Linux /proc/self/maps identifies the file
    backing the symbol's executable mapping, not a second libamdhip64 found
    via LD_LIBRARY_PATH or an arbitrary package version.
    """
    from rocke.runtime import hip_module

    hip = hip_module._resolve_hip()
    version_fn = hip.hipRuntimeGetVersion
    version_fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
    version_fn.restype = ctypes.c_int
    encoded_version = ctypes.c_int(-1)
    status = version_fn(ctypes.byref(encoded_version))
    if status != 0 or encoded_version.value <= 0:
        raise RuntimeError(
            "hipRuntimeGetVersion on rocKE's loaded HIP library failed: "
            f"status={status}, version={encoded_version.value}"
        )

    symbol_ptr = ctypes.cast(version_fn, ctypes.c_void_p).value
    if not symbol_ptr:
        raise RuntimeError("Cannot resolve hipRuntimeGetVersion function address")

    # Resolve the actual mapped shared object from the symbol's instruction
    # address. /proc/self/maps is available on the Linux ROCm benchmark host.
    mapped_path = None
    try:
        with Path("/proc/self/maps").open(encoding="utf-8") as maps:
            for line in maps:
                fields = line.rstrip("\n").split(maxsplit=5)
                if len(fields) < 6:
                    continue
                lo, hi = (int(x, 16) for x in fields[0].split("-", 1))
                if lo <= symbol_ptr < hi:
                    if fields[5].startswith("/"):
                        mapped_path = fields[5]
                    break
    except OSError as exc:
        raise RuntimeError("Cannot inspect /proc/self/maps for HIP library") from exc
    if mapped_path is None:
        raise RuntimeError(
            "Cannot identify the mapped HIP shared object containing "
            "hipRuntimeGetVersion; refusing ambiguous runtime provenance"
        )

    # HIP_VERSION encoding, not CUDA's encoding:
    # major*10_000_000 + minor*100_000 + patch.
    raw = encoded_version.value
    major = raw // 10_000_000
    minor = (raw // 100_000) % 100
    patch = raw % 100_000
    return {
        "version": f"{major}.{minor}.{patch}",
        "version_encoded": raw,
        "loaded_library_path": mapped_path,
        "rocke_library_opened_as": str(hip._name),
        "query": "hipRuntimeGetVersion from rocke.runtime.hip_module's cached HIP handle",
    }


# These are ROCm components that can be loaded by PyTorch, rocKE or the
# compiler. The package version is NOT necessarily the version of every library
# loaded into the process, so capture the actual mapped shared objects too.
_ROCM_SHARED_OBJECT = re.compile(
    r"^lib(?:amdhip64|hsa-runtime64|amd_comgr|hiprtc|hipblas(?:lt)?|"
    r"rocblas|rocsolver|rocsparse|rocfft|hipfft|hiprand|rocrand|"
    r"miopen|rccl|roctracer64|rocprofiler(?:-sdk)?(?:64)?|"
    r"rocprim|hipcub)(?:[.\-]|$)",
    flags=re.IGNORECASE,
)


def _rocm_sdk_package_versions() -> dict[str, str]:
    """Installed ROCm wheel metadata, not a claim about loaded DSOs."""
    distributions = (
        "rocm",
        "rocm-sdk",
        "rocm-sdk-core",
        "rocm-sdk-runtime",
        "rocm-sdk-libraries",
        "rocm-hip-runtime",
        "rocm-comgr",
        "_rocm_sdk_core",
    )
    return {
        name: version
        for name in distributions
        if (version := _package_version(name)) is not None
    }


def _loaded_rocm_shared_objects() -> list[dict]:
    """Actual ROCm shared objects mapped into THIS running Linux process.

    Read /proc/self/maps to get mapped file identity instead of searching PATH,
    LD_LIBRARY_PATH, or an installation directory. The same basename can be
    loaded from different directories, so preserve every distinct mapping.
    This does not infer an unexposed per-library ROCm release version.
    """
    found: dict[tuple[str, str, str], dict] = {}
    try:
        with Path("/proc/self/maps").open(encoding="utf-8") as maps:
            for line in maps:
                fields = line.rstrip("\n").split(maxsplit=5)
                if len(fields) < 6:
                    continue
                mapped_path = fields[5]
                if not mapped_path.startswith("/"):
                    continue
                # A mapped library might have been unlinked after loading.
                name = Path(mapped_path.removesuffix(" (deleted)")).name
                if not _ROCM_SHARED_OBJECT.match(name):
                    continue
                key = (mapped_path, fields[3], fields[4])
                found[key] = {
                    "name": name,
                    "mapped_path": mapped_path,
                    "device": fields[3],
                    "inode": fields[4],
                }
    except OSError as exc:
        raise RuntimeError("Cannot inspect /proc/self/maps for ROCm libraries") from exc
    return sorted(found.values(), key=lambda r: (r["name"], r["mapped_path"]))


def _environment(
    torch, device_id: int, rocke: Path, script_file: Path, git_state: dict
) -> dict:
    # Hardware identity comes from rocKE's HIP runtime, not PyTorch properties.
    from rocke.core.lower_llvm import _resolve_llvm_flavor
    from rocke.runtime.comgr import loaded_compiler_info
    from rocke.runtime.hip_module import (
        get_device_name,
        get_device_num_cus,
        get_device_target_id,
    )

    gpu = {
        "name": get_device_name(device_id),
        "architecture": get_device_target_id(device_id),
        "cu_count_reported_by_hip": get_device_num_cus(device_id),
    }
    # Record the loaded COMGR compiler rather than an environment override or
    # an unrelated clang executable found on PATH.
    compiler = loaded_compiler_info()
    llvm_flavor = _resolve_llvm_flavor()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu,
        "software": {
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "pytorch": torch.__version__,
            "pytorch_hip": getattr(torch.version, "hip", None),
            "hip_runtime": _loaded_hip_runtime_info(),
            "triton": _package_version("triton"),
            "rocm_sdk_package_versions": _rocm_sdk_package_versions(),
        },
        "compile_backend": "python",
        "llvm_flavor": llvm_flavor,
        "loaded_compiler": {
            "llvm_version": (
                ".".join(map(str, compiler.llvm_version))
                if compiler is not None and compiler.llvm_version is not None
                else None
            ),
            "detection_source": compiler.source if compiler is not None else None,
            "comgr_library": compiler.comgr_path if compiler is not None else None,
            "query_library": (
                compiler.query_library_path if compiler is not None else None
            ),
        },
        "git": git_state,
        "source_files": {
            "attention_dense.py": str(
                rocke / "library/kernels/gfx942/attention_dense.py"
            ),
            "attention_dense_sha256": _file_sha256(
                rocke / "library/kernels/gfx942/attention_dense.py"
            ),
            "benchmark_script": str(script_file),
            "benchmark_script_sha256": _file_sha256(script_file),
            "builder_sha256": _file_sha256(
                rocke / "library/builders/gfx942/attention/prefill/attention_dense_prefill.py"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rocke-root", help="Directory containing library/kernels/gfx942"
    )
    parser.add_argument(
        "--shape",
        type=parse_shape,
        action="append",
        help="B,S,Hq,Hkv,128,bf16 (repeatable; overrides default twelve shapes)",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=942)
    parser.add_argument("--reference-chunk", type=int, default=128)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--out-dir", default="~/rocke_5way_specs_results")
    args = parser.parse_args()
    if args.rounds < 3 or args.repeat <= 0 or args.warmup < 0:
        parser.error("require rounds >= 3, repeat > 0 and warmup >= 0")
    if args.reference_chunk <= 0:
        parser.error("reference-chunk must be positive")
    if args.atol < 0 or args.rtol < 0:
        parser.error("atol and rtol must be nonnegative")

    rocke = locate_rocke(args.rocke_root)
    measured_git = _require_clean_checkout(rocke)
    add_module_paths(rocke)

    import torch
    import kernels.gfx942.attention_dense as dense
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime import KernelLauncher, LaunchConfig
    from rocke.runtime.hip_module import (
        current_device,
        get_device_arch,
        get_device_count,
    )
    from dispatch.attention import AttentionRequest, attention_tuning_spec

    if get_device_count() == 0:
        raise SystemExit("ROCm GPU is not visible through HIP")
    for required in ("Gfx942AttentionDenseSpec", "attention_dense_runtime_args"):
        if not hasattr(dense, required):
            raise SystemExit(f"Expected updated attention_dense.py; missing {required}")
    source_file = Path(dense.__file__).resolve()
    expected_file = (rocke / "library/kernels/gfx942/attention_dense.py").resolve()
    if source_file != expected_file:
        raise SystemExit(
            f"Loaded wrong emitter! expected {expected_file}, got {source_file}"
        )
    for knob in ("lds_num_buffers", "v_reg_prefetch", "use_cfvst"):
        if knob not in dense.Gfx942AttentionDenseSpec.__dataclass_fields__:
            raise SystemExit(f"Current Gfx942AttentionDenseSpec is missing knob {knob}")

    # ROCm/PyTorch represents HIP GPU streams through torch.cuda.
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    stream = torch.cuda.current_stream(device=device)
    stream_ptr = int(stream.cuda_stream)
    rng = random.Random(args.seed)
    shapes = args.shape if args.shape else list(DEFAULT_SHAPES)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # HIP's active device is shared with PyTorch's current CUDA/HIP stream.
    # Query hardware through rocKE rather than through torch.cuda.
    device_id = current_device()  # HIP's current device; no torch device-property query
    if get_device_arch(device_id) != "gfx942":
        raise SystemExit("This benchmark requires a gfx942 GPU")
    environment = _environment(
        torch, device_id, rocke, Path(__file__).resolve(), measured_git
    )
    hip_runtime = environment["software"]["hip_runtime"]
    print(
        f"ROCKE_ROOT={rocke} | GPU={environment['gpu']['name']} "
        f"({environment['gpu']['cu_count_reported_by_hip']} CUs) | "
        f"Git={environment['git']['head_sha']} | LLVM={environment['llvm_flavor']}"
    )
    print(f"HIP runtime={hip_runtime['version']} | DSO={hip_runtime['loaded_library_path']}")
    print(f"HIP stream={stream_ptr} | arms={[v.name for v in VARIANTS]}")

    raw_rows: list[dict] = []
    validation_rows: list[dict] = []
    summary_rows: list[dict] = []
    ratio_rows: list[dict] = []
    resolved_rows: list[dict] = []
    print(f"Running {len(shapes)} shape(s): {[s.label for s in shapes]}")
    for shape_index, shape in enumerate(shapes, start=1):
        print(f"\n=== Shape {shape_index}/{len(shapes)}: {shape.label} ===", flush=True)
        # Exact production request: mirror the gfx942 dense builder's dense_request
        # (BF16 causal, square self-attention), rather than hand-building a spec.
        req = AttentionRequest(
            batch=shape.batch,
            nhead_q=shape.hq,
            nhead_k=shape.hkv,
            seqlen_q=shape.seqlen,
            seqlen_k=shape.seqlen,
            hdim_q=shape.d,
            hdim_v=shape.d,
            arch="gfx942",
            mask_type=1,  # causal; same as dense_request(..., causal=True)
            dtype=shape.dtype,
            algorithm="attention_dense",
            spec_id="gfx942_dense",
        )
        production = attention_tuning_spec(req, "gfx942_dense").kernel_spec
        if not isinstance(production, dense.Gfx942AttentionDenseSpec):
            raise RuntimeError(
                f"{shape.label}: dispatch returned {type(production).__name__}, "
                "not Gfx942AttentionDenseSpec"
            )

        # The production BF16 D128 default should have none of the three opt-in
        # optimizations on. Refuse to compare if dispatch policy changes later.
        if (
            production.resolved_use_cfvst(),
            production.lds_num_buffers,
            production.v_reg_prefetch,
        ) != (False, 1, False):
            raise RuntimeError(
                f"{shape.label}: dispatch baseline now has pipeline knobs enabled; "
                "redefine the ablation arms before benchmarking"
            )
        specs = {"baseline": production}  # Exact production spec
        for variant in VARIANTS[1:]:
            specs[variant.name] = replace(
                production,
                use_cfvst=variant.use_cfvst,
                lds_num_buffers=variant.lds_num_buffers,
                v_reg_prefetch=variant.v_reg_prefetch,
            )
        for name, spec in specs.items():
            ok, reason = dense.supports_attention_dense(spec, arch="gfx942")
            if not ok:
                raise RuntimeError(f"{shape.label}/{name} rejected: {reason}")

        # Trace full specs, including policy resolutions that dataclass fields
        # otherwise encode as None. 
        def describe_spec(spec):
            return {
                "fields": asdict(spec),
                "resolved_use_cfvst": spec.resolved_use_cfvst(),
                "resolved_use_exp2_fast": spec.resolved_use_exp2_fast(),
                "resolved_persist_decode": spec.resolved_persist_decode,
                "resolved_v_row_pad": spec.resolved_v_row_pad(),
                "resolved_use_v_swizzle": spec.resolved_use_v_swizzle(),
                "kernel_name": spec.kernel_name(),
                "grid": dense.attention_dense_grid(spec),
                "block": dense.attention_dense_block(spec),
            }

        resolved_rows.append(
            {
                "shape": shape.label,
                "request": asdict(req),
                "source": "dispatch.attention.attention_tuning_spec(req, 'gfx942_dense').kernel_spec",
                "production_baseline": True,
                "arms": {name: describe_spec(spec) for name, spec in specs.items()},
            }
        )
        print(
            f"  dispatch: persistent={production.persistent}, "
            f"NP={production.num_persistent}, "
            f"decode={production.resolved_persist_decode}, "
            f"exp2_fast={production.resolved_use_exp2_fast()}, "
            f"bm/bn={production.block_m}/{production.block_n}, "
            f"wpe={production.waves_per_eu}"
        )
        names = {name: spec.kernel_name() for name, spec in specs.items()}
        if len(set(names.values())) != len(VARIANTS):
            raise RuntimeError(f"Compile-time kernel-name collision: {names}")

        torch.manual_seed(0)
        q = (
            torch.randn(
                (shape.batch, shape.seqlen, shape.hq, 128),
                device=device,
                dtype=torch.bfloat16,
            )
            * 0.2
        ).contiguous()
        k = (
            torch.randn(
                (shape.batch, shape.seqlen, shape.hkv, 128),
                device=device,
                dtype=torch.bfloat16,
            )
            * 0.2
        ).contiguous()
        v_tensor = (torch.randn_like(k) * 0.2).contiguous()
        outputs = {v.name: torch.empty_like(q) for v in VARIANTS}
        scale = 1.0 / math.sqrt(128)
        launchers = {}
        configurations = {}
        values = {}
        compiled_artifacts = {}
        print("  Compiling five independent kernels and saving HSACOs...")
        for variant in VARIANTS:
            name = variant.name
            spec = specs[name]
            artifact = compile_kernel(
                dense.build_attention_dense(spec, arch="gfx942"),
                arch="gfx942",
                backend="python",
                capture_ir_text=False,
            )
            hsaco_dir = out_dir / "hsaco" / shape.label
            hsaco_dir.mkdir(parents=True, exist_ok=True)

            hsaco_path = hsaco_dir / f"{name}.hsaco"
            hsaco_path.write_bytes(artifact.hsaco)
            if artifact.kernel_name != names[name]:
                raise RuntimeError(f"Artifact/kernel name mismatch: {name}")
            compiled_artifacts[name] = {
                "kernel_name": artifact.kernel_name,
                "hsaco_sha256": hashlib.sha256(artifact.hsaco).hexdigest(),
                "lds_bytes": dense._lds_bytes(spec),
            }
            launchers[name] = KernelLauncher(
                hsaco=artifact.hsaco,
                kernel_name=artifact.kernel_name,
                signature=dense.attention_dense_signature(spec),
            )
            configurations[name] = LaunchConfig(
                grid=dense.attention_dense_grid(spec),
                block=dense.attention_dense_block(spec),
                stream=stream_ptr,
            )
            values[name] = {
                "q_ptr": q,
                "k_ptr": k,
                "v_ptr": v_tensor,
                "o_ptr": outputs[name],
                "scale": scale,
                # REQUIRED for updated runtime-shape kernel ABI and fast division.
                **dense.attention_dense_runtime_args(spec),
            }

        resolved_rows[-1]["compiled_artifacts"] = compiled_artifacts
        print("Building one chunked FP32 SDPA reference...")
        reference = fp32_reference(
            torch, q, k, v_tensor, scale=scale, chunk=args.reference_chunk
        )
        def check_and_record(name: str, stage: str, round_id: int | str = ""):
            """Validate against the same FP32 reference and record the exact gate."""
            try:
                max_abs, max_rel = validate(
                    torch, outputs[name], reference, atol=args.atol, rtol=args.rtol
                )
            except (AssertionError, RuntimeError) as exc:
                raise RuntimeError(
                    f"ABORT: correctness failed: {shape.label}/{name}, "
                    f"stage={stage}, round={round_id}"
                ) from exc
            validation_rows.append(dict(
                shape=shape.label, variant=name, stage=stage, round=round_id,
                passed=1, max_abs=f"{max_abs:.9g}", max_rel=f"{max_rel:.9g}",
                atol=args.atol, rtol=args.rtol,
            ))
            return max_abs, max_rel

        print("  Validating five compiled kernels...")
        for variant in VARIANTS:
            name = variant.name
            # Validate through exactly the launcher/config used for timed calls.
            outputs[name].fill_(float("nan"))
            launchers[name](values[name], config=configurations[name])
            torch.cuda.synchronize(device)
            max_abs, max_rel = check_and_record(name, "pre_timing")
            print(f"  PASS {name:20s} max_abs={max_abs:.6g} max_rel={max_rel:.6g}")
        # Keep the FP32 reference alive for post-batch validation.
        torch.cuda.synchronize(device)

        print(f"Round-robin warmup: {args.warmup} launches per variant")
        for _ in range(args.warmup):
            for variant in VARIANTS:
                launchers[variant.name](
                    values[variant.name], config=configurations[variant.name]
                )
        torch.cuda.synchronize(device)

        samples = {variant.name: [] for variant in VARIANTS}
        for round_id in range(1, args.rounds + 1):
            shuffled = list(VARIANTS)
            rng.shuffle(shuffled)
            print(f"Round {round_id}: {[variant.name for variant in shuffled]}")
            for position, variant in enumerate(shuffled, 1):
                name = variant.name
                ms = timed_batch(
                    torch,
                    launchers[name],
                    values[name],
                    configurations[name],
                    stream,
                    args.repeat,
                )
                # Timed events have already synchronized. Only the last output
                # of this batch survives; validate it outside the timed window.
                max_abs, max_rel = check_and_record(name, "post_batch", round_id)
                samples[name].append(ms)
                raw_rows.append(
                    dict(
                        shape=shape.label,
                        round=round_id,
                        order=position,
                        variant=name,
                        use_cfvst=int(variant.use_cfvst),
                        lds_num_buffers=variant.lds_num_buffers,
                        v_reg_prefetch=int(variant.v_reg_prefetch),
                        repeat=args.repeat,
                        ms=f"{ms:.9f}",
                    )
                )
                print(f"  {name:20s} {ms:.6f} ms | PASS post-batch (abs={max_abs:.4g})")

        del reference
        median = {name: statistics.median(times) for name, times in samples.items()}
        print("  Median ms: " + ", ".join(
            f"{name}={median[name]:.6f}" for name in median
        ))
        for variant in VARIANTS:
            name = variant.name
            x = median[name]
            summary_rows.append(
                dict(
                    shape=shape.label,
                    variant=name,
                    median_ms=f"{x:.9f}",
                    min_ms=f"{min(samples[name]):.9f}",
                    max_ms=f"{max(samples[name]):.9f}",
                    speedup_vs_baseline=f"{median['baseline']/x:.9f}",
                    speedup_vs_cfvst=f"{median['cfvst']/x:.9f}",
                    rounds=args.rounds,
                    repeat=args.repeat,
                )
            )

        comparisons = (
            ("baseline", "cfvst"),
            ("cfvst", "cfvst_klds"),
            ("cfvst", "cfvst_vpf"),
            ("cfvst_klds", "cfvst_klds_vpf"),
            ("cfvst_vpf", "cfvst_klds_vpf"),
            ("baseline", "cfvst_klds_vpf"),
        )
        print("Same-session comparisons (A/B speedup, percent lower latency vs A):")
        for before, after in comparisons:
            a, b = median[before], median[after]
            if a <= 0 or b <= 0:
                raise RuntimeError("GPU timing returned nonpositive latency")
            speedup = a / b
            latency_reduction_pct = 100 * (a - b) / a
            paired = [a_round / b_round for a_round, b_round in
                      zip(samples[before], samples[after])]
            ratio_rows.append(
                dict(
                    shape=shape.label,
                    numerator=before,
                    denominator=after,
                    ratio_a_over_b=f"{speedup:.9f}",
                    latency_reduction_pct=f"{latency_reduction_pct:.6f}",
                    paired_min=f"{min(paired):.9f}",
                    paired_max=f"{max(paired):.9f}",
                )
            )
            print(
                f"  {before:20s} / {after:20s} {speedup:.5f}x, {latency_reduction_pct:+.3f}%"
            )

    (out_dir / "resolved_specs.json").write_text(
        json.dumps(resolved_rows, indent=2, default=str) + "\n"
    )
    # Stable CSV schemas: the static analyzer and external result consumers
    # depend on these filenames/field names, so keep them byte-for-byte.
    exports = (
        ("ablation_runs.csv", "shape round order variant use_cfvst lds_num_buffers v_reg_prefetch repeat ms", raw_rows),
        ("ablation_summary.csv", "shape variant median_ms min_ms max_ms speedup_vs_baseline speedup_vs_cfvst rounds repeat", summary_rows),
        ("validation.csv", "shape variant stage round passed max_abs max_rel atol rtol", validation_rows),
        ("ablation_ratios.csv", "shape numerator denominator ratio_a_over_b latency_reduction_pct paired_min paired_max", ratio_rows),
    )
    for filename, schema, rows in exports:
        write_csv(out_dir / filename, tuple(schema.split()), rows)
    environment["run"] = {
        "shapes": [asdict(sh) for sh in shapes],
        "variant_count": len(VARIANTS),
        "correctness_atol": args.atol,
        "correctness_rtol": args.rtol,
        "reference": "chunked independent FP32 causal GQA SDPA",
        "input_dtype": "bf16",
        "warmup_per_arm": args.warmup,
        "rounds": args.rounds,
        "launches_per_round_per_arm": args.repeat,
        "randomization_seed": args.seed,
        "input_seed": 0,
        "current_stream_handle": stream_ptr,
        "validation_scope": "pre-timing and final output after every timed batch",
        "validation_checks_per_shape": len(VARIANTS) * (1 + args.rounds),
        "all_validation_passed": all(v["passed"] == 1 for v in validation_rows),
        "dispatch_baseline_for_all_shapes": True,
        "exact_python_command": shlex.join([sys.executable, *sys.argv]),
        "working_directory": str(Path.cwd()),
    }
    # Snapshot after compilation/launch so lazily loaded ROCm DSOs appear.
    environment["software"]["rocm_libraries_loaded_at_end"] = (
        _loaded_rocm_shared_objects()
    )
    (out_dir / "provenance.json").write_text(
        json.dumps(environment, indent=2, default=str) + "\n"
    )
    print(f"\nOutputs in {out_dir}")
    print("Reproducability: provenance.json, resolved_specs.json, validation.csv, "
          "ablation_ratios.csv; keep latency CSVs internal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())