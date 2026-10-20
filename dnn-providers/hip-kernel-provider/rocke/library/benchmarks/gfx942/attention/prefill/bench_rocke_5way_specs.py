#!/usr/bin/env python3
"""Five-way gfx942 attention_dense ablation using the CURRENT kernel's spec knobs.

No edited emitter or source duplication. Each variant is a separate compile-time
Gfx942AttentionDenseSpec, compiled before timing, with a direct KernelLauncher.

Method: one Python/PyTorch/ROCm process + stream, one Q/K/V input set, one
chunked-FP32 reference, pre-timing correctness for all 5, round-robin warmup,
5 randomized rounds x 50 launches, post-batch correctness of the last launch
in every timed batch, median-of-rounds, same-session ratios.

Default cohort: the twelve BF16 D128 causal shapes from the case-study table
(B=1/16, S=1024/2048/4096/8192/16384 where listed, Hq=32, Hkv=8/16),
with production dispatch resolving persistent mode and all non-ablation knobs.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
import shlex
import json
import hashlib
import math
import os
from pathlib import Path
import random
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
        raise argparse.ArgumentTypeError("positive extents and Hq divisible by Hkv required")
    if s % 256 or s % 64:
        raise argparse.ArgumentTypeError("S must be divisible by block_m=256 and block_n=64")
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


def git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"



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
    # Untracked scripts are deliberately identified as a count, not silently
    # treated as part of the recorded SHA. Source hashes below identify their bytes.
    status = _command_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"], cwd=root
    )
    status_lines = (status or "").splitlines()
    tracked_status = _command_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"], cwd=root
    )
    dirty = None if status is None else bool(status_lines)
    return {
        "repository_root": repo, "head_sha": head, "branch": branch,
        "working_tree_dirty": dirty,
        "tracked_files_dirty": bool(tracked_status) if tracked_status is not None else None,
        "untracked_entry_count": sum(1 for x in status_lines if x.startswith("??")),
        "changed_or_untracked_paths_count": len(status_lines) if status is not None else None,
        "note": (
            "If working_tree_dirty, the HEAD SHA alone does not reproduce this run; "
            "use the recorded source hashes and preserve the changes."
        ),
    }


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_digest(hsaco) -> str | None:
    """Record HSACO identity when the compiler exposes bytes or a real filename."""
    if isinstance(hsaco, (bytes, bytearray, memoryview)):
        return hashlib.sha256(hsaco).hexdigest()
    if isinstance(hsaco, (str, Path)):
        try:
            path = Path(hsaco)
            return _file_sha256(path) if path.is_file() else None
        except (OSError, ValueError):
            return None
    return None


def _environment(torch, device_id: int, rocke: Path, script_file: Path) -> dict:
    # Hardware identity comes from rocKE's HIP runtime, not PyTorch properties.
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
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu,
        "software": {
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "pytorch": torch.__version__,
            "pytorch_hip": getattr(torch.version, "hip", None),
            "pytorch_git_version": getattr(torch.version, "git_version", None),
            "triton": _package_version("triton"),
            "rocm_sdk_core": _package_version("_rocm_sdk_core"),
            "clang_version_on_path_NOT_necessarily_comgr": _command_output(["clang", "--version"]),
            "hipcc_version_on_path_if_available": _command_output(["hipcc", "--version"]),
        },
        "compile_backend": "python",
        "compiler_flavor_requested": os.environ.get("ROCKE_LLVM_FLAVOR"),
        "environment_variables": {
            name: os.environ.get(name) for name in (
                "ROCKE_ROOT", "ROCKE_BACKEND", "ROCKE_LLVM_FLAVOR",
                "ROCKE_COMGR_LIB", "ROCKE_HIP_LIB", "PYTHONPATH",
                "LD_LIBRARY_PATH",
            )
        },
        "git": _git_provenance(rocke),
        "source_files": {
            "attention_dense.py": str(
                rocke / "library/kernels/gfx942/attention_dense.py"
            ),
            "attention_dense_sha256": _file_sha256(
                rocke / "library/kernels/gfx942/attention_dense.py"
            ),
            "benchmark_script": str(script_file),
            "benchmark_script_sha256": _file_sha256(script_file),
        },
    }


def _replay_script(path: Path, environment_variables: dict) -> None:
    """Exact Python invocation and relevant wheel/ROCm environment at this run."""
    names = (
        "ROCKE_ROOT", "ROCKE_BACKEND", "ROCKE_LLVM_FLAVOR",
        "ROCKE_COMGR_LIB", "ROCKE_HIP_LIB", "PYTHONPATH", "LD_LIBRARY_PATH",
    )
    lines = ["#!/usr/bin/env bash", "set -Eeuo pipefail", "",
             "# Replays the captured Python invocation in the captured environment.",
             "# The pinned source files and wheel environment must still exist."]
    for name in names:
        value = environment_variables.get(name)
        if value is not None:
            lines.append(f"export {name}={shlex.quote(value)}")
    lines.append("")
    lines.append("exec " + shlex.join([sys.executable, *sys.argv]))
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o755)


def _report_text(environment: dict, *, args, shapes, resolved_rows,
                 validation_rows, samples_by_shape, summary_rows, ratio_rows) -> str:
    """Relative-only LOCAL DRAFT. Never publish absolute achieved latencies."""
    e = environment
    gpu = e["gpu"]
    soft = e["software"]
    git = e["git"]
    shape_count = len(shapes)
    full = {
        r["shape"]: r for r in ratio_rows
        if r["numerator"] == "baseline" and r["denominator"] == "cfvst_klds_vpf"
    }
    validated = all(v["passed"] == 1 for v in validation_rows)
    prod = all(r["production_baseline"] for r in resolved_rows)
    lines = [
        "# gfx942 BF16 D128 causal prefill: five-way ablation results",
        "",
        "**LOCAL DRAFT — review software-version disclosure before publishing.**",
        "Performance evidence below is relative-only. Internal CSV files hold absolute",
        "GPU timings and should not be copied into public reports.", "",
        f"Measured scope: {shape_count} BF16 D128 causal GQA shapes, gfx942; "
        f"device={gpu['name']}, CUs={gpu['cu_count_reported_by_hip']}; "
        "five compile-time variants of the current `attention_dense.py`.",
        f"Source revision: `{git['head_sha'] or 'unknown'}`; "
        f"working tree dirty: `{git['working_tree_dirty']}`. "
        f"Production dispatch baseline for all shapes: `{prod}`.",
        "", "## Benchmark identity and replay", "",
        f"- Harness: `{e['source_files']['benchmark_script']}`",
        f"- Harness SHA256: `{e['source_files']['benchmark_script_sha256']}`",
        f"- Kernel SHA256: `{e['source_files']['attention_dense_sha256']}`",
        "- Dispatch: `attention_tuning_spec(req, 'gfx942_dense').kernel_spec`",
        "- Each non-baseline arm changes only `use_cfvst`, `lds_num_buffers`, and `v_reg_prefetch`.",
        f"- Compile: backend=`{e['compile_backend']}`, "
        f"ROCKE_LLVM_FLAVOR=`{e['compiler_flavor_requested']}`; "
        f"COMGR library=`{e['environment_variables'].get('ROCKE_COMGR_LIB')}`",
        f"- Python={soft['python_version']}, PyTorch={soft['pytorch']}, "
        f"PyTorch HIP={soft['pytorch_hip']}, Triton={soft['triton']}",
        "- Exact Python replay (including flags and environment): `replay_command.sh`",
        "- Complete dispatch specs, including resolved `persistent`, `exp2_fast`, block sizes, "
        "WPE, and kernel names: `resolved_specs.json`",
        "- Full environment details and Git state: `provenance.json`",
        "", "## Correctness and measurement protocol", "",
        "- Independent reference: chunked FP32 causal GQA SDPA; shared Q/K/V across all arms.",
        f"- All {len(validation_rows)} validation checks (pre-timing and after each "
        f"timed batch) passed `torch.testing.assert_close` against FP32 "
        f"reference: **{validated}**, "
        f"atol={args.atol:g}, rtol={args.rtol:g}, input dtype=BF16.",
        "- Post-batch correctness is checked outside GPU-event timing; each check "
        "verifies the last of the batch's repeated launches, not every launch.",
        "- Same Python process, same current HIP stream, five independent compiled kernels.",
        f"- Warmup: {args.warmup} round-robin launches/arm; "
        f"measurement: {args.rounds} randomized rounds, {args.repeat} direct "
        "KernelLauncher calls/arm/round; GPU events and synchronization.",
        "- Reported performance: ratio of each arm's **median round latency** to the",
        "  baseline median (not median of per-round ratios).",
        "- Spread: per-arm minimum and maximum round latency expressed as percentage",
        "  deviations from that arm's median. These are relative spread figures, not absolute latency.",
        "", "## Five compile-time arms", "",
        "| Arm | use_cfvst | lds_num_buffers (K) | v_reg_prefetch |",
        "|---|---:|---:|---:|",
    ]
    for v in VARIANTS:
        lines.append(f"| {v.name} | {v.use_cfvst} | {v.lds_num_buffers} | {v.v_reg_prefetch} |")
    lines += ["", "## Full pipeline vs production baseline", "",
              "| B | S | Hq | Hkv | D | Median speedup | Lower latency |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for sh in shapes:
        r = full[sh.label]
        lines.append(
            f"| {sh.batch} | {sh.seqlen} | {sh.hq} | {sh.hkv} | {sh.d} | "
            f"{float(r['ratio_a_over_b']):.4f}x | "
            f"{float(r['latency_reduction_pct']):.2f}% |"
        )
    lines += ["", "## Five-way ablation by shape (relative values only)", "",
              "| Shape | Arm | Median speedup vs baseline | Latency reduction vs baseline | "
              "Round min..max deviation from arm median |",
              "|---|---|---:|---:|---:|"]
    by_summary = {(r["shape"], r["variant"]): r for r in summary_rows}
    for sh in shapes:
        for v in VARIANTS:
            r = by_summary[(sh.label, v.name)]
            s = samples_by_shape[sh.label][v.name]
            med = statistics.median(s)
            lo = 100.0 * (min(s) / med - 1.0)
            hi = 100.0 * (max(s) / med - 1.0)
            speedup = float(r["speedup_vs_baseline"])
            red = 100 * (1 - 1 / speedup)
            lines.append(
                f"| {sh.label} | {v.name} | {speedup:.4f}x | {red:+.2f}% | "
                f"{lo:+.2f}% .. {hi:+.2f}% |"
            )
    lines += ["", "## Same-session incremental comparisons", "",
              "Per comparison below: ratio of medians from the same session, with the",
              "range of **paired within-round** ratios (round N vs round N).",
              "", "| Shape | Compared arms (A/B) | Ratio of medians | "
              "Paired-round ratio min..max |",
              "|---|---|---:|---:|"]
    for r in ratio_rows:
        sh = r["shape"]
        a, b = r["numerator"], r["denominator"]
        ax, bx = samples_by_shape[sh][a], samples_by_shape[sh][b]
        paired = [u / v for u, v in zip(ax, bx)]
        lines.append(
            f"| {sh} | {a} / {b} | {float(r['ratio_a_over_b']):.4f}x | "
            f"{min(paired):.4f}x .. {max(paired):.4f}x |"
        )
    lines += ["", "## Interpretation and limitations", "",
              "- The comparisons above isolate CFVST, KDB, VPF and their combination.",
              "- This measured run establishes only the stated GPU and 12 input configurations.",
              "- Results are opt-in: the production dispatcher resolves the non-optimization",
              "  fields, but KDB/VPF may not be reachable from tuning/dispatch by default.",
              "- This script does **not** prove results on another CU count or compiler flavor.",
              "- The original case-study's 15.67% claim cannot be attributed to a",
              "  particular new shape without its original measurement record.",
              "- Review the software stack's public-release status before posting versions.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocke-root", help="Directory containing library/kernels/gfx942")
    parser.add_argument("--shape", type=parse_shape, action="append", help="B,S,Hq,Hkv,128,bf16 (repeatable; overrides default twelve shapes)")
    # Production dispatch owns these by default. Explicit deviations require an
    # opt-in flag and are labelled diagnostic, NOT production-baseline results.
    parser.add_argument("--persistent", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--num-persistent", type=int, default=None)
    parser.add_argument("--persist-decode", choices=("auto", "qb_major", "hkv_major"), default="auto")
    parser.add_argument("--allow-nonproduction-baseline", action="store_true",
                        help="permit explicit persistent/NP/decode overrides that differ from dispatch; labels results diagnostic")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=942)
    parser.add_argument("--reference-chunk", type=int, default=128)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--out-dir", default="results/5way_specs")
    args = parser.parse_args()
    if args.rounds < 1 or args.repeat <= 0 or args.warmup < 0:
        parser.error("require rounds >= 1, repeat > 0 and warmup >= 0")
    if (args.num_persistent is not None and args.num_persistent <= 0) or args.reference_chunk <= 0:
        parser.error("num-persistent, if specified, and reference-chunk must be positive")
    if args.atol < 0 or args.rtol < 0:
        parser.error("atol and rtol must be nonnegative")

    rocke = locate_rocke(args.rocke_root)
    add_module_paths(rocke)

    import torch
    import kernels.gfx942.attention_dense as dense
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime import KernelLauncher, LaunchConfig
    from dispatch.attention import AttentionRequest, attention_tuning_spec

    if not torch.cuda.is_available():
        raise SystemExit("ROCm GPU is not visible through torch.cuda")
    for required in ("Gfx942AttentionDenseSpec", "attention_dense_runtime_args"):
        if not hasattr(dense, required):
            raise SystemExit(f"Expected updated attention_dense.py; missing {required}")
    source_file = Path(dense.__file__).resolve()
    expected_file = (rocke / "library/kernels/gfx942/attention_dense.py").resolve()
    if source_file != expected_file:
        raise SystemExit(f"Loaded wrong emitter! expected {expected_file}, got {source_file}")
    source_text = source_file.read_text()
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
    # Resolve the index of the GPU used by PyTorch's benchmark stream so HIP
    # metadata refers to that same device, even when it is not device 0.
    device_id = device.index if device.index is not None else torch.cuda.current_device()
    environment = _environment(torch, device_id, rocke, Path(__file__).resolve())
    print(f"ROCKE_ROOT={rocke}\nattention_dense={source_file}\nGPU={environment['gpu']['name']}")
    print(f"Git head={environment['git']['head_sha']} dirty={environment['git']['working_tree_dirty']} "
          f"LLVM flavor={environment['compiler_flavor_requested']} "
          f"CUs={environment['gpu']['cu_count_reported_by_hip']}", flush=True)
    print(f"current_stream={stream_ptr} | variants={[v.name for v in VARIANTS]}")

    raw_rows: list[dict] = []
    validation_rows: list[dict] = []
    summary_rows: list[dict] = []
    ratio_rows: list[dict] = []
    resolved_rows: list[dict] = []
    diagnostic_shapes: list[str] = []
    samples_by_shape: dict[str, dict[str, list[float]]] = {}
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

        # Optional execution overrides exist only for explicit diagnostic runs.
        # Crucially, all other fields (especially exp2 policy, padding, geometry,
        # WPE, interleave, and persistent) come from the dispatch-resolved spec.
        requested_overrides = {}
        if args.persistent != "auto":
            requested_overrides["persistent"] = args.persistent == "on"
        if args.num_persistent is not None:
            requested_overrides["num_persistent"] = args.num_persistent
        if args.persist_decode != "auto":
            requested_overrides["persist_decode"] = args.persist_decode
        deviations = {
            field: (getattr(production, field), value)
            for field, value in requested_overrides.items()
            if getattr(production, field) != value
        }
        if deviations and not args.allow_nonproduction_baseline:
            raise RuntimeError(
                f"{shape.label}: requested flags override the dispatch baseline: "
                f"{deviations}. For a production comparison, omit --persistent, "
                "--num-persistent and --persist-decode. To deliberately measure "
                "a diagnostic configuration, add --allow-nonproduction-baseline."
            )
        base_spec = replace(production, **requested_overrides) if deviations else production
        production_baseline = not bool(deviations)
        if not production_baseline:
            diagnostic_shapes.append(shape.label)
            print(f"  *** DIAGNOSTIC NON-PRODUCTION BASELINE: {deviations} ***")

        # The production BF16 D128 default should have none of the three opt-in
        # optimizations on. Refuse to compare if dispatch policy changes later.
        if (base_spec.resolved_use_cfvst(), base_spec.lds_num_buffers,
                base_spec.v_reg_prefetch) != (False, 1, False):
            raise RuntimeError(
                f"{shape.label}: dispatch baseline now has pipeline knobs enabled; "
                "redefine the ablation arms before benchmarking"
            )
        specs = {"baseline": base_spec}  # byte-for-byte production spec, not a copy
        for variant in VARIANTS[1:]:
            specs[variant.name] = replace(
                base_spec,
                use_cfvst=variant.use_cfvst,
                lds_num_buffers=variant.lds_num_buffers,
                v_reg_prefetch=variant.v_reg_prefetch,
            )
        for name, spec in specs.items():
            ok, reason = dense.supports_attention_dense(spec, arch="gfx942")
            if not ok:
                raise RuntimeError(f"{shape.label}/{name} rejected: {reason}")

        # Trace full specs, including policy resolutions that dataclass fields
        # otherwise encode as None. This is the reviewer's baseline proof.
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

        resolved_rows.append({
            "shape": shape.label,
            "request": asdict(req),
            "source": "dispatch.attention.attention_tuning_spec(req, 'gfx942_dense').kernel_spec",
            "production_baseline": production_baseline,
            "diagnostic_overrides": deviations,
            "dispatch_baseline": describe_spec(production),
            "arms": {name: describe_spec(spec) for name, spec in specs.items()},
        })
        print(
            "  dispatch baseline: "
            f"persistent={base_spec.persistent} "
            f"NP={base_spec.num_persistent} "
            f"decode={base_spec.resolved_persist_decode} "
            f"exp2_fast={base_spec.resolved_use_exp2_fast()} "
            f"bm={base_spec.block_m} bn={base_spec.block_n} "
            f"wpe={base_spec.waves_per_eu} "
            f"production={production_baseline}",
            flush=True,
        )
        names = {name: spec.kernel_name() for name, spec in specs.items()}
        if len(set(names.values())) != len(VARIANTS):
            raise RuntimeError(f"Compile-time kernel-name collision: {names}")

        torch.manual_seed(0)
        q = (torch.randn((shape.batch, shape.seqlen, shape.hq, 128), device=device, dtype=torch.bfloat16) * 0.2).contiguous()
        k = (torch.randn((shape.batch, shape.seqlen, shape.hkv, 128), device=device, dtype=torch.bfloat16) * 0.2).contiguous()
        v_tensor = (torch.randn_like(k) * 0.2).contiguous()
        outputs = {v.name: torch.empty_like(q) for v in VARIANTS}
        scale = 1.0 / math.sqrt(128)
        print(f"\n[{shape.label}] benchmark spec derived from production dispatch")

        launchers = {}
        configurations = {}
        values = {}
        compiled_artifacts = {}
        print("Compiling five independent kernels before any timing...")
        for variant in VARIANTS:
            name = variant.name
            spec = specs[name]
            print(f"  {name:20s} cfvst={spec.use_cfvst} K-buffers={spec.lds_num_buffers} V-prefetch={spec.v_reg_prefetch}")
            print(f"    {names[name]} | LDS={dense._lds_bytes(spec)} bytes")
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
                "hsaco_sha256": _artifact_digest(artifact.hsaco),
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
        reference = fp32_reference(torch, q, k, v_tensor, scale=scale, chunk=args.reference_chunk)
        print("Validating all five directly compiled kernels...")
        for variant in VARIANTS:
            name = variant.name
            # Validate through exactly the launcher/config used for timed calls.
            outputs[name].fill_(float("nan"))
            launchers[name](values[name], config=configurations[name])
            torch.cuda.synchronize(device)
            try:
                max_abs, max_rel = validate(
                    torch, outputs[name], reference, atol=args.atol, rtol=args.rtol
                )
            except (AssertionError, RuntimeError) as exc:
                raise RuntimeError(
                    f"ABORT: correctness failed for {shape.label}/{name}; no timing performed"
                ) from exc
            validation_rows.append(dict(
                shape=shape.label, variant=name, stage="pre_timing", round="", passed=1,
                max_abs=f"{max_abs:.9g}", max_rel=f"{max_rel:.9g}",
                atol=args.atol, rtol=args.rtol,
            ))
            print(f"  PASS {name:20s} max_abs={max_abs:.6g} max_rel={max_rel:.6g}")
        # Keep the FP32 reference alive for post-batch validation.
        torch.cuda.synchronize(device)

        print(f"Round-robin warmup: {args.warmup} launches per variant")
        for _ in range(args.warmup):
            for variant in VARIANTS:
                launchers[variant.name](values[variant.name], config=configurations[variant.name])
        torch.cuda.synchronize(device)

        samples = {variant.name: [] for variant in VARIANTS}
        for round_id in range(1, args.rounds + 1):
            shuffled = list(VARIANTS)
            rng.shuffle(shuffled)
            print(f"Round {round_id}: {[variant.name for variant in shuffled]}")
            for position, variant in enumerate(shuffled, 1):
                name = variant.name
                ms = timed_batch(
                    torch, launchers[name], values[name], configurations[name],
                    stream, args.repeat,
                )
                # The GPU timing event has already been synchronized. Check
                # correctness outside the timed interval, after every batch.
                # Only the final launch's output survives the repeated writes.
                try:
                    max_abs, max_rel = validate(
                        torch, outputs[name], reference, atol=args.atol, rtol=args.rtol
                    )
                except (AssertionError, RuntimeError) as exc:
                    raise RuntimeError(
                        f"ABORT: correctness failed after timed batch for "
                        f"{shape.label}/{name}, round={round_id}, latency={ms:.6f} ms"
                    ) from exc
                validation_rows.append(dict(
                    shape=shape.label, variant=name, stage="post_batch",
                    round=round_id, passed=1,
                    max_abs=f"{max_abs:.9g}", max_rel=f"{max_rel:.9g}",
                    atol=args.atol, rtol=args.rtol,
                ))
                samples[name].append(ms)
                raw_rows.append(dict(
                    shape=shape.label,
                    round=round_id,
                    order=position,
                    variant=name,
                    use_cfvst=int(variant.use_cfvst),
                    lds_num_buffers=variant.lds_num_buffers,
                    v_reg_prefetch=int(variant.v_reg_prefetch),
                    repeat=args.repeat,
                    ms=f"{ms:.9f}",
                ))
                print(f"  {name:20s} {ms:.6f} ms | PASS post-batch correctness "
                      f"(max_abs={max_abs:.6g}, max_rel={max_rel:.6g})")

        del reference
        samples_by_shape[shape.label] = samples
        median = {name: statistics.median(times) for name, times in samples.items()}
        print("Medians (ms):")
        for variant in VARIANTS:
            name = variant.name
            x = median[name]
            print(f"  {name:20s} {x:.6f}")
            summary_rows.append(dict(
                shape=shape.label,
                variant=name,
                median_ms=f"{x:.9f}",
                min_ms=f"{min(samples[name]):.9f}",
                max_ms=f"{max(samples[name]):.9f}",
                speedup_vs_baseline=f"{median['baseline']/x:.9f}",
                speedup_vs_cfvst=f"{median['cfvst']/x:.9f}",
                rounds=args.rounds,
                repeat=args.repeat,
            ))

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
            ratio_rows.append(dict(
                shape=shape.label,
                numerator=before, denominator=after,
                ratio_a_over_b=f"{speedup:.9f}",
                latency_reduction_pct=f"{latency_reduction_pct:.6f}",
            ))
            print(f"  {before:20s} / {after:20s} {speedup:.5f}x, {latency_reduction_pct:+.3f}%")

    # All output is written only after every requested shape validates and times.
    (out_dir / "resolved_specs.json").write_text(
        json.dumps(resolved_rows, indent=2, default=str) + "\n"
    )
    write_csv(out_dir / "ablation_runs.csv", (
        "shape", "round", "order", "variant", "use_cfvst", "lds_num_buffers",
        "v_reg_prefetch", "repeat", "ms",
    ), raw_rows)
    write_csv(out_dir / "ablation_summary.csv", (
        "shape", "variant", "median_ms", "min_ms", "max_ms", "speedup_vs_baseline",
        "speedup_vs_cfvst", "rounds", "repeat",
    ), summary_rows)
    write_csv(out_dir / "validation.csv", (
        "shape", "variant", "stage", "round", "passed", "max_abs", "max_rel", "atol", "rtol",
    ), validation_rows)
    write_csv(out_dir / "ablation_ratios.csv", (
        "shape", "numerator", "denominator", "ratio_a_over_b", "latency_reduction_pct",
    ), ratio_rows)
    (out_dir / "measurement_conditions.txt").write_text("\n".join((
        f"rocke_root={rocke}",
        f"git_head={git_head(rocke)}",
        f"attention_dense_file={source_file}",
        f"attention_dense_sha256={hashlib.sha256(source_text.encode()).hexdigest()}",
        f"gpu={environment['gpu']['name']}",
        f"torch={torch.__version__}",
        f"hip={getattr(torch.version, 'hip', None)}",
        f"stream_ptr={stream_ptr}",
        f"ROCKE_BACKEND={os.environ.get('ROCKE_BACKEND')}",
        f"ROCKE_LLVM_FLAVOR={os.environ.get('ROCKE_LLVM_FLAVOR')}",
        f"shapes={[s.label for s in shapes]}",
        f"persistent={args.persistent}",
        f"num_persistent={args.num_persistent}",
        f"persist_decode={args.persist_decode}",
        "baseline_source=dispatch.attention.attention_tuning_spec(req, gfx942_dense).kernel_spec",
        f"production_baseline_all_shapes={not bool(diagnostic_shapes)}",
        f"nonproduction_shapes={diagnostic_shapes}",
        "full_per_shape_specs=resolved_specs.json",
        f"warmup={args.warmup}",
        f"rounds={args.rounds}",
        f"repeat={args.repeat}",
        f"seed={args.seed}",
        f"reference_chunk={args.reference_chunk}",
        f"validation_atol={args.atol}",
        f"validation_rtol={args.rtol}",
        "timing=direct KernelLauncher, one PyTorch/HIP stream, randomized 5-way rounds",
        "validation=pre-timing and post-timed-batch last-output check outside GPU events",
        "variant_kernel_source=CURRENT library/kernels/gfx942/attention_dense.py",
    )) + "\n")
    environment["run"] = {
        "shapes": [asdict(sh) for sh in shapes],
        "variant_count": len(VARIANTS),
        "correctness_atol": args.atol, "correctness_rtol": args.rtol,
        "reference": "chunked independent FP32 causal GQA SDPA",
        "input_dtype": "bf16", "warmup_per_arm": args.warmup,
        "rounds": args.rounds, "launches_per_round_per_arm": args.repeat,
        "randomization_seed": args.seed, "input_seed": 0,
        "current_stream_handle": stream_ptr,
        "validation_scope": "pre-timing and final output after every timed batch",
        "validation_checks_per_shape": len(VARIANTS) * (1 + args.rounds),
        "all_validation_passed": all(v["passed"] == 1 for v in validation_rows),
        "dispatch_baseline_for_all_shapes": not bool(diagnostic_shapes),
        "diagnostic_shape_labels": diagnostic_shapes,
        "exact_python_command": shlex.join([sys.executable, *sys.argv]),
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(environment, indent=2, default=str) + "\n"
    )
    _replay_script(out_dir / "replay_command.sh", environment["environment_variables"])
    report = _report_text(
        environment, args=args, shapes=shapes, resolved_rows=resolved_rows,
        validation_rows=validation_rows, samples_by_shape=samples_by_shape,
        summary_rows=summary_rows, ratio_rows=ratio_rows,
    )
    (out_dir / "ablation_report.md").write_text(report)
    print(f"\nOutputs in {out_dir}")
    print("Reviewer evidence: ablation_report.md, provenance.json, "
          "resolved_specs.json, replay_command.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())