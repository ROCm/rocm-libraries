#!/usr/bin/env python3
"""Eight-arm persistent gfx942 BF16 D128 attention: Step-0 winner A vs seven-stage B ablation.

A uses the seven validated shape-specific Step-0 winners from the
2026-10-28 staged sweep (no CFVST/KDB/VPF). That sweep predates #14367, so A
pins ``use_cfvst=False`` to rebuild the recorded winner. B[1..7] always start
from the CURRENT per-shape production-dispatch spec, independently of A. Since
#14367, production resolves CFVST ON for BF16 D128, so B1 forces it off to give
the CFVST-off reference and B2 IS production dispatch. Thus this is a fair new
same-source A/B measurement, NOT a claim about cross-commit timings.

   A sweep_winner        per-shape best preexisting levers (CFVST/KDB/VPF off)
   1 baseline            production dispatch with CFVST forced off (use_cfvst=False)
   2 cfvst               production dispatch (CFVST on by policy; KDB/VPF off)
   3 cfvst_kdb           CFVST + K LDS double buffering
   4 cfvst_vpf           CFVST + V register prefetch (KDB off)
   5 pipeline            CFVST + KDB + V register prefetch
   6 pipeline_pad        full pipeline + lds_row_pad=4
   7 pipeline_pad_dsplit previous + causal_diag_split=True

Stages 3 and 4 are separate branches from CFVST alone. Stage 5 combines
both branches. A is the ONLY arm receiving lever-sweep geometry, scheduling,
and grid knobs. The seven B configurations do not inherit A's knobs.
The exact single-knob B deltas are checked against production dispatch.
The rebuilt A must compile to the recorded Step-0 winner kernel name, or the
run aborts; resolved specs, HSACO hashes and artifacts are kept.

Methodology: torch-free NumPy/rocKE, one process and explicit HIP stream,
shared Q/K/V and chunked NumPy FP32 causal GQA reference over BF16 inputs,
BF16 correctness before timing and after every timed batch,
round-robin warmup, five randomized rounds, 50 launches per arm per round,
median latency and same-round paired min/max speedups.
Every arm is selected through tuning_spec_with_knobs and replayed from its
recorded (spec_id, tuning_id, knobs) against its dispatch-time spec_hash before
timing. A dirty or untracked checkout is refused unless --allow-dirty is
passed. provenance.json records the source SHA-256 of HASHED_SOURCES;
--verify-provenance OUT_DIR re-checks them and the replay at a later checkout,
and refuses any non-Markdown change since the measured commit.
static_resources.csv records registers, spills, scratch, LDS and static ISA
counts from the exact HSACOs that are timed.
Absolute latency results are kept in local CSVs.
Default cohort: twelve persistent BF16 D128 causal shapes. Seven have recorded
Step-0 winners and run all eight arms. B1 S1024/2048 (Hkv 8 and 16) and B16
S8192 Hkv8 have no Step-0 winner, so they run the seven production-derived B
arms without A: their B ablations against production dispatch are reported,
and no A/B ratio is fabricated for them.
Custom --shape values must be among these twelve shapes and select persistent
dispatch.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
import hashlib
import json
import math
import numpy as np
from pathlib import Path
import random
import re
import shlex
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass

B_VARIANTS = (
    "baseline",
    "cfvst",
    "cfvst_kdb",
    "cfvst_vpf",
    "pipeline",
    "pipeline_pad",
    "pipeline_pad_dsplit",
)
VARIANTS = ("A", *B_VARIANTS)
STAGE_NUMBERS = {"A": 0, **{name: i for i, name in enumerate(B_VARIANTS, start=1)}}

STAGE_DESCRIPTIONS = {
    "A": "Per-shape best existing-lever sweep winner (CFVST/KDB/VPF off)",
    "baseline": "Production dispatch with CFVST forced off (use_cfvst=False)",
    "cfvst": "Production dispatch: CFVST on by policy, KDB/VPF off",
    "cfvst_kdb": "CFVST + K LDS double buffering (no V prefetch)",
    "cfvst_vpf": "CFVST + V register prefetch (no K double buffering)",
    "pipeline": "CFVST + K LDS double buffering + V register prefetch",
    "pipeline_pad": "Full pipeline + lds_row_pad=4",
    "pipeline_pad_dsplit": "Full pipeline + lds_row_pad=4 + causal_diag_split=True",
}

# Dependency graph (not a linear chain): both KDB and VPF extend CFVST.
# The full pipeline is cross-checked against both upstream branches.
PARENT_VARIANT = {
    "A": None,
    "baseline": None,
    "cfvst": "baseline",
    "cfvst_kdb": "cfvst",
    "cfvst_vpf": "cfvst",
    "pipeline": "cfvst_kdb",
    "pipeline_pad": "pipeline",
    "pipeline_pad_dsplit": "pipeline_pad",
}

# Field deltas against the production spec. Production leaves use_cfvst=None,
# which _use_cfvst resolves ON at BF16 D128 (#14367); dispatch canonicalizes a
# redundant use_cfvst=True pin back to None, so only the CFVST-off stage pins it.
STAGE_OVERRIDES = {
    "A": {},  # Separate sweep-winner specification; not a B stage.
    "baseline": {"use_cfvst": False},
    "cfvst": {},
    "cfvst_kdb": {"lds_num_buffers": 2},
    "cfvst_vpf": {"v_reg_prefetch": True},
    "pipeline": {"lds_num_buffers": 2, "v_reg_prefetch": True},
    "pipeline_pad": {
        "lds_num_buffers": 2,
        "v_reg_prefetch": True,
        "lds_row_pad": 4,
    },
    "pipeline_pad_dsplit": {
        "lds_num_buffers": 2,
        "v_reg_prefetch": True,
        "lds_row_pad": 4,
        "causal_diag_split": True,
    },
}


# Exact final A/B-tested Step-0 winners from the 2026-10-28 staged sweep,
# NOT the older 12-row sweep. The sweep's source HEAD is recorded.
SWEEP_SOURCE_COMMIT = "ceeae550e7aa95b291aff5cb75ec4d626a072bf3"

# Every arm is selected through dispatch under this candidate.
SPEC_ID = "gfx942_dense"

# Files whose bytes determine what is benchmarked, relative to the rocke root.
# provenance.json records their SHA-256 so --verify-provenance can prove that a
# later checkout (e.g. a docs-only commit on top) still measures the same code.
HASHED_SOURCES = (
    "library/kernels/gfx942/attention_dense.py",
    "library/kernels/common/attention_dense_spec.py",
    "library/dispatch/attention/axes.py",
    "library/dispatch/attention/dense_rules.py",
    "library/dispatch/attention/gfx942_dense.py",
    "library/builders/gfx942/attention/prefill/attention_dense_prefill.py",
    "library/benchmarks/gfx942/attention/prefill/bench_rocke_7way_specs.py",
)

# Shapes without a Step-0 winner in the newer sweep. B=16, S=8192, Hkv=8 has
# no completed final A/B validation; B=1, S=1024/2048 are not in that sweep's
# cohort. Run the seven production-derived B arms only; do not invent a sweep
# winner or an A arm for these shapes.
MISSING_SWEEP_WINNERS = {
    (1, 1024, 8): "not in newer sweep cohort",
    (1, 1024, 16): "not in newer sweep cohort",
    (1, 2048, 8): "not in newer sweep cohort",
    (1, 2048, 16): "not in newer sweep cohort",
    (16, 8192, 8): "missing_files (no completed final A/B validation in newer sweep)",
}

# Shared fields match the newer report's recorded full winner specifications.
# None and auto values are intentional: they preserve the original policy
# inputs rather than inventing resolved values. The new pipeline is held OFF.
A_COMMON = dict(
    block_m=256,
    block_n=64,
    block_size=0,
    causal=True,
    causal_bottom_right=False,
    chiplet_num_xcds=8,
    dtype="bf16",
    head_size=128,
    persistent=True,
    num_persistent=304,
    persist_decode="auto",
    nonpersist_decode="auto",
    num_kv_blocks=0,
    num_query_heads=32,
    paged=False,
    ragged=False,
    sliding_window=0,
    use_sinks=False,
    varlen=False,
    lds_row_pad=4,
    lds_k_group_pad=8,
    lds_num_buffers=1,
    iglp=True,
    iglp_mode=1,
    pv_loop_order="k_major",
    causal_diag_split=True,
    o_store_width=4,
    lazy_rescale=True,
    interleave=False,
    use_cfvst=False,  # the sweep predates #14367: its winners ran CFVST off.
    use_exp2_fast=None,
    use_v_swizzle=None,
    v_row_pad=None,
    pv_sched_fence_mask=None,
    emit_lse=False,
    v_reg_prefetch=False,  # new field: absent from old report; must be OFF.
)

# Only shape-specific winning knobs differ across the seven completed rows.
# All shapes: Hq=32, D=128, BF16, causal, 304 persistent workgroups.
WINNING_LEVERS = {
    (1, 4096, 8): dict(waves_per_eu=3, pv_priority=1),
    (1, 4096, 16): dict(waves_per_eu=2, pv_priority=1),
    (1, 8192, 8): dict(waves_per_eu=3, pv_priority=1),
    (1, 8192, 16): dict(waves_per_eu=3, pv_priority=2),
    (1, 16384, 8): dict(waves_per_eu=3, pv_priority=3),
    (16, 4096, 8): dict(waves_per_eu=2, pv_priority=3),
    (16, 4096, 16): dict(waves_per_eu=2, pv_priority=3),
}

# The seven HISTORICAL Step-0 winner symbols, exactly as recorded by the sweep
# (pre-#14367, use_cfvst=None resolving off). Exported verbatim for provenance.
_A_PREFIX = "rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_"
_A_SUFFIX = "_krowpad4_iglp1_iglpm1_pvkmaj_prio"
EXPECTED_A_KERNEL = {
    (1, 4096, 8): _A_PREFIX + "bthkvmin_gfx942_wpe3" + _A_SUFFIX + "1_dsplit",
    (1, 4096, 16): _A_PREFIX + "bthkvmin_gfx942_wpe2" + _A_SUFFIX + "1_dsplit",
    (1, 8192, 8): _A_PREFIX + "bthkvmin_gfx942_wpe3" + _A_SUFFIX + "1_dsplit",
    (1, 8192, 16): _A_PREFIX + "bthkvmin_gfx942_wpe3" + _A_SUFFIX + "2_dsplit",
    (1, 16384, 8): _A_PREFIX + "bthkvmin_gfx942_wpe3" + _A_SUFFIX + "3_dsplit",
    (16, 4096, 8): _A_PREFIX + "hkvmaj_gfx942_wpe2" + _A_SUFFIX + "3_dsplit",
    (16, 4096, 16): _A_PREFIX + "gfx942_wpe2" + _A_SUFFIX + "3_dsplit",
}


def _current_a_symbol(historical: str) -> str:
    """The symbol the rebuilt A must compile to at this checkout: the historical
    symbol plus the ``cfvst0`` token kernel_name() emits for the explicit
    use_cfvst=False pin A now needs (CFVST resolves on by default since #14367).
    The guard compares against this; provenance keeps the historical name too."""
    assert historical.count("_krowpad4_") == 1, historical
    return historical.replace("_krowpad4_", "_krowpad4_cfvst0_")


CURRENT_EXPECTED_A_KERNEL = {
    key: _current_a_symbol(name) for key, name in EXPECTED_A_KERNEL.items()
}
assert set(EXPECTED_A_KERNEL) == set(WINNING_LEVERS)
assert not set(MISSING_SWEEP_WINNERS) & set(WINNING_LEVERS)


# speedup = median(numerator)/median(denominator)
# Paired min/max = extrema of same-round numerator/denominator ratios.
# Each effect is measured relative to its correct parent, not list position.
COMPARISON_PAIRS = (
    ("baseline", "cfvst", "cfvst_only"),
    ("cfvst", "cfvst_kdb", "kdb_given_cfvst"),
    ("cfvst", "cfvst_vpf", "vpf_given_cfvst"),
    ("cfvst_kdb", "pipeline", "vpf_given_cfvst_kdb"),
    ("cfvst_vpf", "pipeline", "kdb_given_cfvst_vpf"),
    ("baseline", "cfvst_kdb", "cfvst_plus_kdb_total"),
    ("baseline", "cfvst_vpf", "cfvst_plus_vpf_total"),
    ("baseline", "pipeline", "pipeline_total"),
    ("pipeline", "pipeline_pad", "lds_pad"),
    ("pipeline_pad", "pipeline_pad_dsplit", "causal_split"),
    ("baseline", "pipeline_pad_dsplit", "total"),
    # A is the best earlier Step-0 lever winner; each B stage uses a separate
    # production-derived spec. A/B results are remeasured on the SAME checkout.
    *(("A", name, f"A_vs_{name}") for name in B_VARIANTS),
)
EFFECT_NAMES = tuple(effect for _, _, effect in COMPARISON_PAIRS)


def changed_fields(before, after) -> set[str]:
    """Confirm cumulative stages differ only in their advertised knobs."""
    b, a = asdict(before), asdict(after)
    return {key for key in b if b[key] != a[key]}


def make_request(shape):
    """Exact production request: mirror the gfx942 dense builder's dense_request
    (BF16 causal, square self-attention), rather than hand-building a spec."""
    from dispatch.attention import AttentionRequest

    return AttentionRequest(
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
        spec_id=SPEC_ID,
    )


def tuned_arm(req, production, overrides: dict):
    """Select one arm through dispatch, as a pinned request would.

    The knobs are the arm's field delta against the production spec, passed to
    ``tuning_spec_with_knobs``; the result must be exactly the spec the overrides
    describe, so the ablation measures what dispatch hands out.
    """
    from dispatch.attention import tuning_spec_with_knobs

    expected = replace(production, **overrides)
    want = asdict(expected)
    have = asdict(production)
    knobs = {key: want[key] for key in want if want[key] != have[key]}
    tuned = tuning_spec_with_knobs(req, SPEC_ID, knobs)
    if tuned.kernel_spec != expected:
        got = asdict(tuned.kernel_spec)
        raise RuntimeError(
            f"dispatch built a different spec than {sorted(knobs)} describe: "
            f"{sorted(key for key in got if got[key] != want[key])}"
        )
    return tuned


def replay_record(req, tuned) -> dict:
    """What a tuning store keeps to rebuild an arm: the candidate, ``tuning_id``
    and canonical knobs, checked on replay against the dispatch-time
    ``spec_hash`` (dispatch ARCHITECTURE.md section 11.1)."""
    from rocke.dispatch.core import spec_identity

    return {
        "candidate": tuned.candidate_name,
        "algorithm": req.algorithm,
        "spec_id": SPEC_ID,
        "tuning_id": tuned.tuning_id,
        "config_key": tuned.config_key,
        "knobs": dict(tuned.knobs),
        "spec_hash": spec_identity(tuned),
    }


def replay_spec_hash(req, record: dict) -> str:
    """Dispatch ``req`` pinned to a stored record, round-tripped through JSON as
    a tuning store would keep it, and return the result's ``spec_hash``."""
    from dispatch.attention import dispatch_attention

    stored = json.loads(json.dumps(record))
    pinned = replace(
        req,
        algorithm=stored["algorithm"],
        spec_id=stored["spec_id"],
        tuning_id=stored["tuning_id"],
        tuning_knobs=stored["knobs"],
    )
    return dispatch_attention(pinned).kernel_id.spec_hash


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
    # The pipeline supports BF16 D128.
    if dtype != "bf16" or d != 128:
        raise argparse.ArgumentTypeError("seven-way ablation requires bf16 D128")
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


def fp32_reference(q, k, v, *, scale: float, chunk: int):
    """Chunked causal GQA SDPA in NumPy FP32 over the exact BF16 inputs.

    Batches and KV heads are processed separately. Peak score matrix is
    (Hq/Hkv, chunk, S) so largest shapes do not allocate an S-by-S tensor.
    """
    batch, sq, hq, d = q.shape
    _, sk, hkv, _ = k.shape
    gqa = hq // hkv
    result = np.empty((batch, sq, hq, d), dtype=np.float32)
    key_indices = np.arange(sk)[None, :]
    for b in range(batch):
        for h in range(hkv):
            first, last = h * gqa, (h + 1) * gqa
            kt = k[b, :, h, :].T
            val = v[b, :, h, :]
            for qs in range(0, sq, chunk):
                qe = min(qs + chunk, sq)
                query = q[b, qs:qe, first:last, :].transpose(1, 0, 2)
                scores = np.matmul(query, kt)
                scores *= np.float32(scale)
                # Bottom-left aligned causal mask: key_index <= query_index.
                masked = key_indices > np.arange(qs, qe)[:, None]
                np.copyto(scores, -np.inf, where=masked[None, :, :])
                scores -= np.max(scores, axis=-1, keepdims=True)
                np.exp(scores, out=scores)
                scores /= np.sum(scores, axis=-1, keepdims=True)
                result[b, qs:qe, first:last, :] = np.matmul(scores, val).transpose(
                    1, 0, 2
                )
    return result


def validate(actual_bits, reference, *, atol: float, rtol: float, chunk: int = 256):
    """Compare every BF16 output element to FP32 reference, chunked on host."""
    from rocke.numeric.bf16 import bf16_to_f32

    if actual_bits.shape != reference.shape:
        raise AssertionError(
            f"output shape {actual_bits.shape} != reference {reference.shape}"
        )
    if not np.isfinite(reference).all():
        raise AssertionError("FP32 reference contains nonfinite values")
    max_abs = 0.0
    max_rel = 0.0
    for b in range(reference.shape[0]):
        for qs in range(0, reference.shape[1], chunk):
            sl = slice(qs, qs + chunk)
            actual32 = bf16_to_f32(actual_bits[b, sl])
            expected = reference[b, sl]
            np.testing.assert_allclose(
                actual32, expected, atol=atol, rtol=rtol, equal_nan=False
            )
            diff = np.abs(actual32 - expected)
            max_abs = max(max_abs, float(diff.max()))
            max_rel = max(
                max_rel, float((diff / np.maximum(np.abs(expected), 1e-6)).max())
            )
    return max_abs, max_rel


def _device(array):
    """Torch-free ROCm transfer: NumPy -> rocKE DeviceMem."""
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.launcher import DeviceMem

    array = np.ascontiguousarray(array)
    mem = DeviceMem(array.nbytes)
    Runtime().memcpy_h2d(mem.ptr(), as_u8_buffer(array), array.nbytes)
    return mem


def _download_bf16(device_buffer, shape):
    """Read raw BF16 uint16 bits using rocKE's HIP host-copy utility."""
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.host_buffers import as_u8_buffer

    bits = np.empty(shape, dtype=np.uint16)
    Runtime().memcpy_d2h(as_u8_buffer(bits), device_buffer.ptr(), bits.nbytes)
    return bits


@contextmanager
def _hip_stream():
    """A dedicated HIP stream, avoiding framework-dependent stream lookup."""
    from rocke.runtime import hip_module
    from rocke.runtime.hip_module import Runtime

    hip = hip_module._resolve_hip()  # Same HIP DSO used by rocKE.
    create = hip.hipStreamCreate
    create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    create.restype = ctypes.c_int
    destroy = hip.hipStreamDestroy
    destroy.argtypes = [ctypes.c_void_p]
    destroy.restype = ctypes.c_int
    handle = ctypes.c_void_p()
    hip_module._check(create(ctypes.byref(handle)), "hipStreamCreate")
    if not handle.value:
        raise RuntimeError("hipStreamCreate returned null HIP stream")
    stream = int(handle.value)
    try:
        yield stream
    finally:
        try:
            Runtime().wait_stream(stream)
        finally:
            hip_module._check(destroy(handle), "hipStreamDestroy")


def timed_batch(launcher, values, config, stream: int, repeat: int) -> float:
    """HIP events around exactly N launches; preserves old launch fencing.

    Equivalent to the original torch.cuda.Event harness: no changed
    LaunchConfig fence setting, no compilation or correctness checks timed.
    """
    from rocke.runtime.hip_module import Runtime

    first = Runtime().event()
    last = Runtime().event()
    try:
        first.record(stream=stream)
        for _ in range(repeat):
            launcher(values, config=config)
        last.record(stream=stream)
        last.synchronize()
        return float(first.elapsed_to(last)) / repeat
    finally:
        first.destroy()
        last.destroy()


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


def _capture_git_checkout(root: Path, *, allow_dirty: bool) -> dict:
    """Record Git provenance. A dirty, untracked or unknown checkout is refused
    unless ``allow_dirty``: results from it cannot be tied to a commit."""
    state = _git_provenance(root)
    clean = state["head_sha"] is not None and state["working_tree_dirty"] is False
    print(
        f"Git HEAD={state['head_sha'] or 'unknown'}; "
        f"dirty={state['working_tree_dirty']}"
        + (" (allowed by --allow-dirty)" if not clean and allow_dirty else ""),
        flush=True,
    )
    if not clean and not allow_dirty:
        raise SystemExit(
            "Refusing to benchmark: the checkout is dirty, has untracked files, or "
            "is not a Git work tree. Commit first, or pass --allow-dirty for a "
            "development run whose results cannot be cited."
        )
    return state


def _llvm_tool(name: str) -> str:
    """An LLVM binutil from PATH, else from ``llvm/bin`` beside the loaded COMGR.

    The fallback disassembles with the toolchain that compiled the kernels.
    """
    from rocke.runtime.comgr import loaded_compiler_info

    found = shutil.which(name)
    if found:
        return found
    compiler = loaded_compiler_info()
    if compiler is not None and compiler.comgr_path:
        tool = Path(compiler.comgr_path).resolve().parent / "llvm" / "bin" / name
        if tool.is_file():
            return str(tool)
    raise SystemExit(f"{name} not found on PATH or beside the loaded COMGR library")


def static_resources(hsaco: bytes, hsaco_path: Path) -> dict:
    """Compiler-reported registers, spills, scratch and LDS from the code-object
    notes, plus static ISA instruction counts from the disassembly. Taken from
    the exact HSACO that is timed, so the tables share the timings' provenance."""
    from rocke.analysis.isa import analyze_hsaco
    from rocke.benchmark.perf.occupancy import resources

    notes = resources(hsaco, "gfx942")
    if not notes:
        raise RuntimeError(f"no code-object notes in {hsaco_path}")
    analysis = analyze_hsaco(
        hsaco_path,
        objdump=_llvm_tool("llvm-objdump"),
        readelf=_llvm_tool("llvm-readelf"),
    )
    return {
        "vgpr": notes["vgpr"],
        "agpr": notes["agpr"],
        "sgpr": notes["sgpr"],
        "vgpr_spill": notes["vgpr_spill"],
        "sgpr_spill": notes["sgpr_spill"],
        "scratch_bytes": analysis.resources.scratch_bytes,
        "lds_bytes": notes["lds_bytes"],
        "occupancy_waves_per_simd": notes["occupancy"],
        **analysis.isa.as_dict(),
    }


def _source_hashes(rocke: Path) -> dict[str, str]:
    """SHA-256 of every file in HASHED_SOURCES, keyed by rocke-relative path."""
    return {rel: _file_sha256(rocke / rel) for rel in HASHED_SOURCES}


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


# These are ROCm components that can be loaded by rocKE or the
# compiler. The package version is NOT necessarily the version of every library
# loaded into the process, so capture the actual mapped shared objects too.
_ROCM_SHARED_OBJECT = re.compile(
    r"^lib(?:amdhip64|hsa-runtime64|amd_comgr|hiprtc|hipblas(?:lt)?|"
    r"rocblas|rocsolver|rocsparse|rocfft|hipfft|hiprand|rocrand|"
    r"miopen|rccl|roctracer64|rocprofiler(?:-sdk)?(?:64)?|"
    r"rocprim|hipcub)(?:[.-]|$)",
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
    device_id: int, rocke: Path, script_file: Path, git_state: dict
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
            "numpy": np.__version__,
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
                rocke
                / "library/builders/gfx942/attention/prefill/attention_dense_prefill.py"
            ),
        },
        "source_sha256": _source_hashes(rocke),
    }


def verify_provenance(out_dir: Path, rocke: Path) -> int:
    """Check that a previous run still describes the current checkout.

    Passes only when the run was measured on a clean commit, every file in
    HASHED_SOURCES has the bytes it had then, and every arm's stored replay
    record (spec_id, tuning_id, knobs) still dispatches to its recorded
    spec_hash. No GPU is needed. Returns the process exit status.
    """
    provenance = json.loads((out_dir / "provenance.json").read_text())
    resolved = json.loads((out_dir / "resolved_specs.json").read_text())
    problems: list[str] = []

    measured = provenance["git"]
    current = _git_provenance(rocke)
    print(
        f"Measured at {measured['head_sha']} (dirty={measured['working_tree_dirty']}); "
        f"verifying at {current['head_sha']} (dirty={current['working_tree_dirty']})"
    )
    if measured["working_tree_dirty"] is not False or not measured["head_sha"]:
        problems.append("the run was measured on a dirty or unknown checkout")
    if provenance.get("run", {}).get("dirty_checkout_allowed"):
        problems.append("the run was started with --allow-dirty")

    # Everything the kernels depend on (platform compiler, dispatch, emitters)
    # must be byte-identical; only Markdown may change after the measured commit.
    changed = None
    if measured["head_sha"]:
        changed = _command_output(
            ["git", "diff", "--name-only", measured["head_sha"]], cwd=rocke
        )
    if changed is None:
        problems.append("cannot diff the checkout against the measured commit")
    else:
        for rel in changed.splitlines():
            if not rel.endswith(".md"):
                problems.append(f"{rel}: changed since the measured commit")
        print(
            f"  {len(changed.splitlines())} file(s) changed since the measured commit"
        )

    recorded = provenance.get("source_sha256", {})
    for rel, digest in _source_hashes(rocke).items():
        if rel not in recorded:
            problems.append(f"{rel}: not recorded by the run")
        elif recorded[rel] != digest:
            problems.append(f"{rel}: SHA-256 changed since the run")
        else:
            print(f"  same bytes  {rel}")

    for row in resolved:
        req = make_request(Shape(**row["shape_fields"]))
        for name, record in row["replay"].items():
            replayed = replay_spec_hash(req, record)
            if replayed != record["spec_hash"]:
                problems.append(
                    f"{row['shape']}/{name}: {record['tuning_id']} now replays to "
                    f"spec_hash {replayed}, recorded {record['spec_hash']}"
                )
        print(f"  replayed    {row['shape']}: {len(row['replay'])} arm(s)")

    for problem in problems:
        print(f"MISMATCH: {problem}")
    print("Provenance verified." if not problems else "Provenance NOT verified.")
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rocke-root", help="Directory containing library/kernels/gfx942"
    )
    parser.add_argument(
        "--shape",
        type=parse_shape,
        action="append",
        help="B,S,Hq,Hkv,128,bf16 (repeatable; overrides default twelve persistent shapes)",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=942)
    parser.add_argument("--reference-chunk", type=int, default=128)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--out-dir", default="~/rocke_sevenway_ablation_results")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Run on a dirty/untracked checkout (development only; recorded in "
        "provenance.json, and the results cannot be cited)",
    )
    parser.add_argument(
        "--verify-provenance",
        metavar="OUT_DIR",
        help="Check a previous run's provenance.json and resolved_specs.json "
        "against the current checkout (source hashes and dispatch replay), then exit",
    )
    args = parser.parse_args()
    if args.verify_provenance:
        rocke = locate_rocke(args.rocke_root)
        add_module_paths(rocke)
        return verify_provenance(Path(args.verify_provenance).expanduser(), rocke)
    if args.rounds < 3 or args.repeat <= 0 or args.warmup < 0:
        parser.error("require rounds >= 3, repeat > 0 and warmup >= 0")
    if args.reference_chunk <= 0:
        parser.error("reference-chunk must be positive")
    if args.atol < 0 or args.rtol < 0:
        parser.error("atol and rtol must be nonnegative")

    rocke = locate_rocke(args.rocke_root)
    measured_git = _capture_git_checkout(rocke, allow_dirty=args.allow_dirty)
    add_module_paths(rocke)

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
    needed_knobs = {
        knob for overrides in STAGE_OVERRIDES.values() for knob in overrides
    }
    needed_knobs.update(A_COMMON)
    needed_knobs.update(
        {"persistent", "num_persistent", "persist_decode", "nonpersist_decode"}
    )
    for overrides in WINNING_LEVERS.values():
        needed_knobs.update(overrides)
    for knob in sorted(needed_knobs):
        if knob not in dense.Gfx942AttentionDenseSpec.__dataclass_fields__:
            raise SystemExit(f"Current Gfx942AttentionDenseSpec is missing knob {knob}")
    rng = random.Random(args.seed)
    shapes = args.shape if args.shape else list(DEFAULT_SHAPES)
    # All twelve are runnable: a shape without a Step-0 winner gets no A arm.
    requested_shapes = list(shapes)
    no_winner_shapes = []
    shapes = []
    for shape in requested_shapes:
        key = (shape.batch, shape.seqlen, shape.hkv)
        if shape.hq != 32:
            parser.error(f"No recorded sweep cohort for {shape.label}: Hq must be 32")
        if key in MISSING_SWEEP_WINNERS:
            reason = MISSING_SWEEP_WINNERS[key]
            print(f"NO STEP-0 WINNER {shape.label}: {reason}; B arms only", flush=True)
            no_winner_shapes.append(
                {"shape": shape.label, "reason": reason, "status": "b_arms_only"}
            )
            shapes.append(shape)
        elif key in WINNING_LEVERS:
            shapes.append(shape)
        else:
            parser.error(
                f"Shape is absent from the twelve-workload cohort: {shape.label}"
            )
    if not shapes:
        parser.error("No runnable shapes selected")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device_id = current_device()
    if get_device_arch(device_id) != "gfx942":
        raise SystemExit("This benchmark requires a gfx942 GPU")
    environment = _environment(device_id, rocke, Path(__file__).resolve(), measured_git)
    hip_runtime = environment["software"]["hip_runtime"]
    print(
        f"ROCKE_ROOT={rocke} | GPU={environment['gpu']['name']} "
        f"({environment['gpu']['cu_count_reported_by_hip']} CUs) | "
        f"Git={environment['git']['head_sha']} "
        f"(dirty={environment['git']['working_tree_dirty']}) | "
        f"LLVM={environment['llvm_flavor']}"
    )
    print(
        f"HIP runtime={hip_runtime['version']} | DSO={hip_runtime['loaded_library_path']}"
    )
    # Stream is released even if compilation/numerical validation fails.
    with _hip_stream() as stream_ptr:
        return _run_shapes(
            args,
            rocke,
            environment,
            shapes,
            out_dir,
            rng,
            stream_ptr,
            no_winner_shapes=no_winner_shapes,
            requested_shapes=requested_shapes,
        )


def _run_shapes(
    args,
    rocke,
    environment,
    shapes,
    out_dir,
    rng,
    stream_ptr,
    *,
    no_winner_shapes,
    requested_shapes,
) -> int:
    import kernels.gfx942.attention_dense as dense
    from dispatch.attention import attention_tuning_spec
    from rocke.helpers.compile import compile_kernel
    from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16
    from rocke.runtime import KernelLauncher, LaunchConfig
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import DeviceMem

    rt = Runtime()
    print(
        f"HIP stream={stream_ptr} | full ablation stages={list(VARIANTS)}; "
        "a shape without a Step-0 winner runs the B arms only"
    )

    raw_rows: list[dict] = []
    validation_rows: list[dict] = []
    summary_rows: list[dict] = []
    ratio_rows: list[dict] = []
    comparison_rows: list[dict] = []
    resolved_rows: list[dict] = []
    static_rows: list[dict] = []
    print(f"Running {len(shapes)} shape(s): {[s.label for s in shapes]}")
    for shape_index, shape in enumerate(shapes, start=1):
        print(f"\n=== Shape {shape_index}/{len(shapes)}: {shape.label} ===", flush=True)
        req = make_request(shape)
        production = attention_tuning_spec(req, SPEC_ID).kernel_spec
        if not isinstance(production, dense.Gfx942AttentionDenseSpec):
            raise RuntimeError(
                f"{shape.label}: dispatch returned {type(production).__name__}, "
                "not Gfx942AttentionDenseSpec"
            )
        if not production.persistent:
            raise RuntimeError(
                f"{shape.label}: production dispatch selected persistent=False; "
                "KDB/VPF are gated to persistent=True. This eight-arm benchmark "
                "requires a persistent shape."
            )

        # A is benchmarked only when a validated sweep winner exists; the B
        # stages derive from production dispatch and run on every shape.
        key = (shape.batch, shape.seqlen, shape.hkv)
        no_winner = key in MISSING_SWEEP_WINNERS
        variant_names = B_VARIANTS if no_winner else VARIANTS
        # Every arm, A included, is selected through dispatch from its knob
        # delta and replayed from its stored record before anything is timed.
        tuned = {}
        a_overrides = None
        original_a_name = None
        actual_a_name = None
        if not no_winner:
            a_overrides = A_COMMON | WINNING_LEVERS[key]
            tuned["A"] = tuned_arm(req, production, a_overrides)
        for name in B_VARIANTS:
            tuned[name] = tuned_arm(req, production, STAGE_OVERRIDES[name])
        specs = {name: t.kernel_spec for name, t in tuned.items()}
        replay = {name: replay_record(req, t) for name, t in tuned.items()}
        for name, record in replay.items():
            replayed = replay_spec_hash(req, record)
            if replayed != record["spec_hash"]:
                raise RuntimeError(
                    f"{shape.label}/{name}: replaying {record['tuning_id']} gives "
                    f"spec_hash {replayed}, dispatch-time {record['spec_hash']}"
                )
        if not no_winner:
            original_a_name = EXPECTED_A_KERNEL[key]
            actual_a_name = specs["A"].kernel_name()
            if actual_a_name != CURRENT_EXPECTED_A_KERNEL[key]:
                raise RuntimeError(
                    f"{shape.label}: rebuilt A is not the recorded Step-0 winner; "
                    f"expected={CURRENT_EXPECTED_A_KERNEL[key]} "
                    f"(historical={original_a_name}), rebuilt={actual_a_name}"
                )
            if (
                specs["A"].resolved_use_cfvst()
                or specs["A"].lds_num_buffers != 1
                or specs["A"].v_reg_prefetch
            ):
                raise RuntimeError(
                    f"{shape.label}: A unexpectedly enables CFVST/KDB/VPF"
                )

        if (
            not production.resolved_use_cfvst()
            or production.lds_num_buffers != 1
            or production.v_reg_prefetch
        ):
            raise RuntimeError(
                f"{shape.label}: production dispatch is not CFVST-on with KDB/VPF "
                "off. B2 must equal production dispatch; refusing a misleading "
                "ablation."
            )
        expected_steps = (
            ("baseline", "cfvst", {"use_cfvst"}),
            ("cfvst", "cfvst_kdb", {"lds_num_buffers"}),
            ("cfvst", "cfvst_vpf", {"v_reg_prefetch"}),
            ("cfvst_kdb", "pipeline", {"v_reg_prefetch"}),
            ("cfvst_vpf", "pipeline", {"lds_num_buffers"}),
            ("pipeline", "pipeline_pad", {"lds_row_pad"}),
            ("pipeline_pad", "pipeline_pad_dsplit", {"causal_diag_split"}),
        )
        for before, after, expected in expected_steps:
            actual = changed_fields(specs[before], specs[after])
            if actual != expected:
                raise RuntimeError(
                    f"{shape.label}: {before} -> {after} changed {sorted(actual)}, "
                    f"expected {sorted(expected)}. Current dispatch may already "
                    "enable a lever; refusing a misleading ablation."
                )
        # Check resolved policy too: default/None fields might silently resolve
        # differently than their literal dataclass values.
        for name in B_VARIANTS:
            spec = specs[name]
            expected = STAGE_OVERRIDES[name]
            observed_pipeline = (
                bool(spec.resolved_use_cfvst()),
                int(spec.lds_num_buffers),
                bool(spec.v_reg_prefetch),
            )
            expected_pipeline = (
                # Unpinned use_cfvst resolves ON at BF16 D128 (_use_cfvst).
                bool(expected.get("use_cfvst", True)),
                int(expected.get("lds_num_buffers", 1)),
                bool(expected.get("v_reg_prefetch", False)),
            )
            if observed_pipeline != expected_pipeline:
                raise RuntimeError(
                    f"{shape.label}/{name}: resolved pipeline knobs "
                    f"{observed_pipeline} differ from expected {expected_pipeline}"
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
                "shape_fields": asdict(shape),
                "request": asdict(req),
                "source": "dispatch.attention.tuning_spec_with_knobs(req, 'gfx942_dense', knobs) per arm",
                # Per arm: candidate, spec_id, tuning_id, canonical knobs and the
                # dispatch-time spec_hash; each was replayed before timing.
                "replay": replay,
                "production_baseline": True,
                "production_dispatch_kernel": production.kernel_name(),
                "execution_mode": (
                    "b_arms_only" if no_winner else "eight_arm_ablation"
                ),
                "missing_step0_winner_reason": MISSING_SWEEP_WINNERS.get(key),
                "variant_names": list(variant_names),
                "A_role": (
                    None
                    if no_winner
                    else "per-shape Step-0 lever-sweep winner, independently rebuilt"
                ),
                "A_original_expected_kernel": original_a_name,
                "A_current_expected_kernel": (
                    None if no_winner else CURRENT_EXPECTED_A_KERNEL[key]
                ),
                "A_current_kernel": actual_a_name,
                "A_kernel_name_matches_original": (
                    None if no_winner else actual_a_name == original_a_name
                ),
                "A_kernel_name_matches_current_expected": (
                    None
                    if no_winner
                    else actual_a_name == CURRENT_EXPECTED_A_KERNEL[key]
                ),
                "A_overrides": a_overrides,
                "B_stage_baseline": "current production dispatch",
                "stage_descriptions": {
                    name: STAGE_DESCRIPTIONS[name] for name in variant_names
                },
                "stage_overrides": {
                    name: STAGE_OVERRIDES[name] for name in variant_names
                },
                "parent_variant": {
                    name: PARENT_VARIANT[name] for name in variant_names
                },
                "comparison_pairs": [
                    pair
                    for pair in COMPARISON_PAIRS
                    if pair[0] in variant_names and pair[1] in variant_names
                ],
                "arms": {name: describe_spec(spec) for name, spec in specs.items()},
            }
        )
        for name, spec in specs.items():
            print(f"  {STAGE_NUMBERS[name]} {name:20s}: {spec.kernel_name()}")
        print(
            "  No Step-0 winner: B stages only, from current production dispatch"
            if no_winner
            else "  A source: Step-0 winner; B stages: current production dispatch "
            "with pipeline-only overrides"
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
        if len(set(names.values())) != len(variant_names):
            raise RuntimeError(f"Compile-time kernel-name collision: {names}")

        # Match the original two BF16-rounding points: random BF16, multiply
        # by 0.2, round back to BF16. The NumPy RNG differs from torch's RNG.
        input_rng = np.random.default_rng(0)
        q_shape = (shape.batch, shape.seqlen, shape.hq, 128)
        kv_shape = (shape.batch, shape.seqlen, shape.hkv, 128)

        def bf16_random_scaled(dims):
            original = input_rng.standard_normal(dims, dtype=np.float32)
            initial_bf16 = bf16_to_f32(f32_to_bf16(original))
            return f32_to_bf16(initial_bf16 * np.float32(0.2))

        q_bits = bf16_random_scaled(q_shape)
        k_bits = bf16_random_scaled(kv_shape)
        v_bits = bf16_random_scaled(kv_shape)
        q = _device(q_bits)
        k = _device(k_bits)
        v_tensor = _device(v_bits)
        outputs = {name: DeviceMem(q_bits.nbytes) for name in variant_names}
        scale = 1.0 / math.sqrt(128)
        launchers = {}
        configurations = {}
        values = {}
        compiled_artifacts = {}
        print(f"  Compiling {len(variant_names)} variant(s) and saving HSACOs...")
        for name in variant_names:
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
            static_rows.append(
                {
                    "shape": shape.label,
                    "variant": name,
                    "kernel_name": artifact.kernel_name,
                    "hsaco_sha256": compiled_artifacts[name]["hsaco_sha256"],
                    **static_resources(artifact.hsaco, hsaco_path),
                }
            )
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
        print("Building one chunked FP32 NumPy SDPA reference...")
        reference = fp32_reference(
            bf16_to_f32(q_bits),
            bf16_to_f32(k_bits),
            bf16_to_f32(v_bits),
            scale=scale,
            chunk=args.reference_chunk,
        )
        del q_bits, k_bits, v_bits

        def check_and_record(name: str, stage: str, round_id: int | str = ""):
            """Validate against the same FP32 reference and record the exact gate."""
            try:
                bits = _download_bf16(outputs[name], reference.shape)
                max_abs, max_rel = validate(
                    bits, reference, atol=args.atol, rtol=args.rtol
                )
            except (AssertionError, RuntimeError) as exc:
                raise RuntimeError(
                    f"ABORT: correctness failed: {shape.label}/{name}, "
                    f"stage={stage}, round={round_id}"
                ) from exc
            validation_rows.append(
                dict(
                    shape=shape.label,
                    variant=name,
                    stage=stage,
                    round=round_id,
                    passed=1,
                    max_abs=f"{max_abs:.9g}",
                    max_rel=f"{max_rel:.9g}",
                    atol=args.atol,
                    rtol=args.rtol,
                )
            )
            return max_abs, max_rel

        print(f"  Validating all {len(variant_names)} compiled kernels...")
        for name in variant_names:
            # Validate through exactly the launcher/config used for timed calls.
            # NaN poison catches elements not written by a kernel.
            rt.memset(outputs[name].ptr(), 0xFF, outputs[name].nbytes())
            rt.sync()  # memset runs on default HIP stream
            launchers[name](values[name], config=configurations[name])
            rt.wait_stream(stream_ptr)
            max_abs, max_rel = check_and_record(name, "pre_timing")
            print(f"  PASS {name:20s} max_abs={max_abs:.6g} max_rel={max_rel:.6g}")
        # Keep the FP32 reference alive for post-batch validation.
        rt.wait_stream(stream_ptr)

        print(f"Round-robin warmup: {args.warmup} launches per variant")
        for _ in range(args.warmup):
            for name in variant_names:
                launchers[name](values[name], config=configurations[name])
        rt.wait_stream(stream_ptr)

        samples = {name: [] for name in variant_names}
        for round_id in range(1, args.rounds + 1):
            shuffled = list(variant_names)
            rng.shuffle(shuffled)
            print(f"Round {round_id}: {shuffled}")
            for position, name in enumerate(shuffled, 1):
                ms = timed_batch(
                    launchers[name],
                    values[name],
                    configurations[name],
                    stream_ptr,
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
                        tuning_id=replay[name]["tuning_id"],
                        spec_hash=replay[name]["spec_hash"],
                        use_cfvst=int(specs[name].resolved_use_cfvst()),
                        lds_num_buffers=specs[name].lds_num_buffers,
                        v_reg_prefetch=int(specs[name].v_reg_prefetch),
                        lds_row_pad=specs[name].lds_row_pad,
                        causal_diag_split=int(specs[name].causal_diag_split),
                        repeat=args.repeat,
                        ms=f"{ms:.9f}",
                    )
                )
                print(f"  {name:20s} {ms:.6f} ms | PASS post-batch (abs={max_abs:.4g})")

        del reference
        rt.wait_stream(stream_ptr)
        median = {name: statistics.median(times) for name, times in samples.items()}
        print(
            "  Median ms: " + ", ".join(f"{name}={median[name]:.6f}" for name in median)
        )
        for name in variant_names:
            x = median[name]
            summary_rows.append(
                dict(
                    shape=shape.label,
                    variant=name,
                    kernel_name=names[name],
                    tuning_id=replay[name]["tuning_id"],
                    knobs=json.dumps(replay[name]["knobs"], sort_keys=True),
                    spec_hash=replay[name]["spec_hash"],
                    median_ms=f"{x:.9f}",
                    min_ms=f"{min(samples[name]):.9f}",
                    max_ms=f"{max(samples[name]):.9f}",
                    speedup_vs_baseline=f"{median['baseline']/x:.9f}",
                    parent_variant=PARENT_VARIANT[name] or "",
                    speedup_vs_parent=(
                        f"{median[PARENT_VARIANT[name]]/x:.9f}"
                        if PARENT_VARIANT[name] is not None
                        else "1.000000000"
                    ),
                    stage=STAGE_NUMBERS[name],
                    description=STAGE_DESCRIPTIONS[name],
                    rounds=args.rounds,
                    repeat=args.repeat,
                )
            )

        if any(x <= 0 for x in median.values()):
            raise RuntimeError("GPU timing returned nonpositive latency")
        pair_metrics = {}
        for numerator, denominator, effect in COMPARISON_PAIRS:
            if numerator not in median or denominator not in median:
                continue  # baseline-only shape has no comparable arms
            before, after = median[numerator], median[denominator]
            paired = [x / y for x, y in zip(samples[numerator], samples[denominator])]
            speedup = before / after
            reduction = 100.0 * (before - after) / before
            metrics = dict(
                shape=shape.label,
                effect=effect,
                numerator=numerator,
                denominator=denominator,
                speedup=f"{speedup:.9f}",
                latency_reduction_pct=f"{reduction:.6f}",
                paired_min=f"{min(paired):.9f}",
                paired_max=f"{max(paired):.9f}",
            )
            ratio_rows.append(metrics)
            pair_metrics[effect] = metrics
        comparison_rows.append(
            {
                "batch": shape.batch,
                "seqlen": shape.seqlen,
                "hkv": shape.hkv,
                **{f"{name}_kernel": names[name] for name in variant_names},
                **{
                    f"{name}_median_ms": f"{median[name]:.9f}" for name in variant_names
                },
                **{
                    f"{effect}_speedup": pair_metrics[effect]["speedup"]
                    for effect in pair_metrics
                },
                **{
                    f"{effect}_reduction_pct": pair_metrics[effect][
                        "latency_reduction_pct"
                    ]
                    for effect in pair_metrics
                },
            }
        )
        print(
            "  Incremental, branch-specific and cumulative results"
            + (" (no A: missing Step-0 winner):" if no_winner else ":"),
            flush=True,
        )
        for effect in pair_metrics:
            r = pair_metrics[effect]
            print(
                f"    {effect:28s} {float(r['speedup']):.5f}x "
                f"({float(r['latency_reduction_pct']):+.2f}% latency reduction; "
                f"paired {float(r['paired_min']):.5f}-{float(r['paired_max']):.5f}x)",
                flush=True,
            )
        rt.wait_stream(stream_ptr)
        del outputs, values, q, k, v_tensor, launchers, configurations

    (out_dir / "missing_sweep_winners.json").write_text(
        json.dumps(no_winner_shapes, indent=2) + "\n"
    )
    (out_dir / "resolved_specs.json").write_text(
        json.dumps(resolved_rows, indent=2, default=str) + "\n"
    )
    # Self-contained eight-arm schemas; no external analyzer assumptions.
    exports = (
        (
            "ablation_comparison.csv",
            "batch seqlen hkv "
            + " ".join(f"{name}_kernel" for name in VARIANTS)
            + " "
            + " ".join(f"{name}_median_ms" for name in VARIANTS)
            + " "
            + " ".join(f"{effect}_speedup" for effect in EFFECT_NAMES)
            + " "
            + " ".join(f"{effect}_reduction_pct" for effect in EFFECT_NAMES),
            comparison_rows,
        ),
        (
            "ablation_runs.csv",
            "shape round order variant tuning_id spec_hash use_cfvst lds_num_buffers v_reg_prefetch "
            "lds_row_pad causal_diag_split repeat ms",
            raw_rows,
        ),
        (
            "ablation_summary.csv",
            "shape variant kernel_name tuning_id knobs spec_hash median_ms min_ms max_ms "
            "speedup_vs_baseline parent_variant speedup_vs_parent stage description rounds repeat",
            summary_rows,
        ),
        (
            "validation.csv",
            "shape variant stage round passed max_abs max_rel atol rtol",
            validation_rows,
        ),
        (
            "ablation_ratios.csv",
            "shape effect numerator denominator speedup latency_reduction_pct paired_min paired_max",
            ratio_rows,
        ),
    )
    for filename, schema, rows in exports:
        write_csv(out_dir / filename, tuple(schema.split()), rows)
    if static_rows:
        write_csv(out_dir / "static_resources.csv", tuple(static_rows[0]), static_rows)
    # Speedup-only Markdown; raw absolute timings remain in local CSV files.
    ratios_by_shape = {}
    for r in ratio_rows:
        ratios_by_shape.setdefault(r["shape"], {})[r["effect"]] = r

    def fmt(r, effect):
        if effect not in r:
            return "—"  # no A arm for a shape without a Step-0 winner
        x = r[effect]
        return (
            f"{float(x['speedup']):.5f}x "
            f"[{float(x['paired_min']):.5f}, "
            f"{float(x['paired_max']):.5f}]"
        )

    lines = [
        "# Step-0 versus B1–B7 ablation",
        "",
        "## Variant definitions",
        "A is the recorded shape-specific existing-lever sweep winner (when available), rebuilt on the current emitter. ",
        "B1-B7 are derived separately from the CURRENT production dispatch. ",
        "A's optimization knobs are **not** added to any B stage. ",
        "Original A symbols and current compiled A symbols are recorded in resolved_specs.json.",
        "Shapes without a Step-0 winner run B1-B7 only, and their A columns are —.",
        "",
        "| # | Variant | Knobs enabled | Parent comparison |",
        "|---:|---|---|---|",
    ]
    for name in VARIANTS:
        overrides = STAGE_OVERRIDES[name]
        knobs = (
            "Shape-specific existing-lever winner (see resolved_specs.json)"
            if name == "A"
            else ", ".join(f"`{k}={v}`" for k, v in overrides.items())
            or "Production dispatch"
        )
        parent = PARENT_VARIANT[name] or "—"
        lines.append(f"| {STAGE_NUMBERS[name]} | `{name}` | {knobs} | `{parent}` |")
    lines += [
        "",
        "## A versus the production-derived B stages (new same-source A/B)",
        "| B | S | Hkv | A / B1 baseline | A / B2 CFVST | A / B3 CFVST+KDB | A / B4 CFVST+VPF | A / B5 full | A / B6 full+pad | A / B7 full+pad+split |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sh in shapes:
        r = ratios_by_shape.get(sh.label, {})
        effects = tuple(f"A_vs_{name}" for name in B_VARIANTS)
        lines.append(
            f"| {sh.batch} | {sh.seqlen} | {sh.hkv} | "
            + " | ".join(fmt(r, effect) for effect in effects)
            + " |"
        )
    # A shape without a Step-0 winner appears as a row of — in the A/B table.
    lines += [
        "",
        "## Direct B ablations (production baseline / CFVST anchor)",
        "| B | S | Hkv | CFVST / baseline | CFVST+KDB / CFVST | CFVST+VPF / CFVST | Full / CFVST+KDB | Full / CFVST+VPF |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sh in shapes:
        r = ratios_by_shape.get(sh.label, {})
        effects = (
            "cfvst_only",
            "kdb_given_cfvst",
            "vpf_given_cfvst",
            "vpf_given_cfvst_kdb",
            "kdb_given_cfvst_vpf",
        )
        lines.append(
            f"| {sh.batch} | {sh.seqlen} | {sh.hkv} | "
            + " | ".join(fmt(r, effect) for effect in effects)
            + " |"
        )
    lines += [
        "",
        "## Cumulative, padding and causal-split ablation",
        "| B | S | Hkv | Full / baseline | + LDS pad (6/5) | + Causal split (7/6) | Total (7/1) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sh in shapes:
        r = ratios_by_shape.get(sh.label, {})
        effects = ("pipeline_total", "lds_pad", "causal_split", "total")
        lines.append(
            f"| {sh.batch} | {sh.seqlen} | {sh.hkv} | "
            + " | ".join(fmt(r, effect) for effect in effects)
            + " |"
        )
    lines += [
        "",
        "Speedup = numerator-arm median latency divided by denominator-arm median "
        "latency. Bracketed range = min/max of same-round paired latency ratios. "
        "A value >1x means the denominator arm is faster.",
        "",
        "## Compiled kernels",
        "| B | S | Hkv | "
        + " | ".join(f"{STAGE_NUMBERS[name]}: {name}" for name in VARIANTS)
        + " |",
        "|---:|---:|---:|" + "---|" * len(VARIANTS),
    ]
    for r in comparison_rows:
        lines.append(
            f"| {r['batch']} | {r['seqlen']} | {r['hkv']} | "
            + " | ".join(
                f"`{r[f'{name}_kernel']}`" if f"{name}_kernel" in r else "—"
                for name in VARIANTS
            )
            + " |"
        )
    if no_winner_shapes:
        lines += [
            "",
            "## Shapes without a Step-0 winner (B1-B7 only, no A)",
            "| Shape | Reason |",
            "|---|---|",
        ]
        lines += [f"| {row['shape']} | {row['reason']} |" for row in no_winner_shapes]
    (out_dir / "ablation_comparison.md").write_text("\n".join(lines) + "\n")
    # A-focused compact report retains the expected symbol, current symbol,
    # paired A/B min/max ratios and medians without mixing old sweep timings.
    a_vs_b_rows = []
    for sh in shapes:
        key = (sh.batch, sh.seqlen, sh.hkv)
        if key not in EXPECTED_A_KERNEL:
            continue  # no A arm, so no A/B7 comparison
        r = next(
            row
            for row in comparison_rows
            if (row["batch"], row["seqlen"], row["hkv"]) == key
        )
        p = ratios_by_shape[sh.label]
        a_vs_b_rows.append(
            {
                "batch": sh.batch,
                "seqlen": sh.seqlen,
                "hkv": sh.hkv,
                "A_original_kernel": EXPECTED_A_KERNEL[key],
                "A_current_expected_kernel": CURRENT_EXPECTED_A_KERNEL[key],
                "A_compiled_kernel": r["A_kernel"],
                "A_kernel_name_matches_original": r["A_kernel"]
                == EXPECTED_A_KERNEL[key],
                "A_kernel_name_matches_current_expected": r["A_kernel"]
                == CURRENT_EXPECTED_A_KERNEL[key],
                "B7_kernel": r["pipeline_pad_dsplit_kernel"],
                "A_median_ms": r["A_median_ms"],
                "B7_median_ms": r["pipeline_pad_dsplit_median_ms"],
                "speedup_A_over_B7": p["A_vs_pipeline_pad_dsplit"]["speedup"],
                "latency_reduction_pct": p["A_vs_pipeline_pad_dsplit"][
                    "latency_reduction_pct"
                ],
                "paired_min": p["A_vs_pipeline_pad_dsplit"]["paired_min"],
                "paired_max": p["A_vs_pipeline_pad_dsplit"]["paired_max"],
            }
        )
    a_vs_b_fields = (
        "batch",
        "seqlen",
        "hkv",
        "A_original_kernel",
        "A_current_expected_kernel",
        "A_compiled_kernel",
        "A_kernel_name_matches_original",
        "A_kernel_name_matches_current_expected",
        "B7_kernel",
        "A_median_ms",
        "B7_median_ms",
        "speedup_A_over_B7",
        "latency_reduction_pct",
        "paired_min",
        "paired_max",
    )
    write_csv(out_dir / "A_vs_B7.csv", a_vs_b_fields, a_vs_b_rows)
    environment["run"] = {
        "shapes": [asdict(sh) for sh in shapes],
        "requested_shapes": [asdict(sh) for sh in requested_shapes],
        "no_step0_winner_shapes": no_winner_shapes,
        "variant_counts_by_shape": {
            sh.label: (
                len(B_VARIANTS)
                if (sh.batch, sh.seqlen, sh.hkv) in MISSING_SWEEP_WINNERS
                else len(VARIANTS)
            )
            for sh in shapes
        },
        "sweep_winner_source_commit": SWEEP_SOURCE_COMMIT,
        "full_ablation_variant_count": len(VARIANTS),
        "correctness_atol": args.atol,
        "correctness_rtol": args.rtol,
        "reference": "chunked NumPy FP32 causal GQA SDPA over BF16-rounded inputs",
        "input_dtype": "bf16",
        "warmup_per_arm": args.warmup,
        "rounds": args.rounds,
        "launches_per_round_per_arm": args.repeat,
        "randomization_seed": args.seed,
        "input_seed": 0,
        "input_rng": "numpy.random.default_rng(0)",
        "input_encoding": "NumPy FP32, BF16 RNE, multiply 0.2, BF16 RNE",
        "hip_stream_handle": stream_ptr,
        "timing": (
            f"rocKE Runtime HIP events around {args.repeat} direct KernelLauncher "
            "calls, unchanged launcher fencing"
        ),
        "validation_scope": "pre-timing and final output after every timed batch",
        "validation_checks_by_shape": {
            sh.label: (
                len(B_VARIANTS)
                if (sh.batch, sh.seqlen, sh.hkv) in MISSING_SWEEP_WINNERS
                else len(VARIANTS)
            )
            * (1 + args.rounds)
            for sh in shapes
        },
        "all_validation_passed": all(v["passed"] == 1 for v in validation_rows),
        "dispatch_baseline_for_all_shapes": True,
        "A_is_per_shape_step0_winner_where_available": True,
        "A_config_matches_newer_seven_winner_report": True,
        "no_A_comparison_for_shapes_without_step0_winner": True,
        "A_expected_kernels": {str(k): v for k, v in EXPECTED_A_KERNEL.items()},
        "A_current_expected_kernels": {
            str(k): v for k, v in CURRENT_EXPECTED_A_KERNEL.items()
        },
        "B_stages_derive_only_from_production": True,
        "stage_descriptions": STAGE_DESCRIPTIONS,
        "stage_overrides": STAGE_OVERRIDES,
        "parent_variant": PARENT_VARIANT,
        "comparison_pairs": COMPARISON_PAIRS,
        "dirty_checkout_allowed": args.allow_dirty,
        "exact_python_command": shlex.join([sys.executable, *sys.argv]),
        "working_directory": str(Path.cwd()),
    }
    # Snapshot after compilation/launch so lazily loaded ROCm DSOs appear.
    environment["software"][
        "rocm_libraries_loaded_at_end"
    ] = _loaded_rocm_shared_objects()
    (out_dir / "provenance.json").write_text(
        json.dumps(environment, indent=2, default=str) + "\n"
    )
    print(f"\nOutputs in {out_dir}")
    print(
        "Reproducibility: provenance.json, resolved_specs.json, validation.csv, "
        "ablation_ratios.csv, ablation_comparison.csv, ablation_comparison.md, A_vs_B7.csv, "
        "static_resources.csv, missing_sweep_winners.json; "
        "keep latency CSVs internal"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
