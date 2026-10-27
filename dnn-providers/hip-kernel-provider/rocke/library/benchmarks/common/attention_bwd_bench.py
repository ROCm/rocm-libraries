# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Paired timing harness for the attention backward (dQ, dK, dV).

Measures the complete backward of each competitor arm (PyTorch SDPA paths) and
of each rocKE configuration on the same device tensors, interleaving the arms
inside every attempt, one fresh process per attempt. It also owns the backward
census helpers: loading the committed census, the performance class of a cell,
and the deterministic class-representative generator.

Per cell and job: one reference process computes the fp32 chunked reference
(and, for the largest cells, fp64 samples of dQ rows and dK / dV keys, which
also check the reference itself), compiles every rocKE configuration's kernels
into the on-disk compile cache (``--compile-cache``, else
``$ROCKE_BWD_HSACO_CACHE``, else a job-local directory) and prepares the
pruning-set oracles; it stores them in a job-local cache. Attempt processes
only load code objects from that cache (a missing one is an arm error, never
a compile inside a timed process). Each attempt process regenerates the inputs
(checked against the stored digest), then for every arm runs the correctness
gate before timing it: the pruning set (rocKE arms, ``--gate`` hook, on the
code objects that are timed), the timed shape against the cached reference
(and samples), and the rocKE canaries. A cell whose spread exceeds
``NOISY_SPREAD`` gets ``ATTEMPTS_NOISY`` attempts; ``--rerun-ambiguous`` adds a
block of attempts to ambiguous cells (pooled, first attempt of every block
discarded).

Output discipline: stdout and stderr carry only verdict tokens (``pass``,
``fail``, ``ambiguous``, ``correctness_fail``, ...) and counts. Absolute times
are written only as JSONL into the results directory given by ``--out-dir``
(or the ``ROCKE_BWD_RESULTS_DIR`` environment variable), which must lie
outside the source tree. Gate constants are read by name from the file given
by ``--gate-constants``; their values are never printed.

Usage (repository-relative)::

    python library/benchmarks/common/attention_bwd_bench.py --count
    python library/benchmarks/common/attention_bwd_bench.py --list-arms --core-only
    python library/benchmarks/common/attention_bwd_bench.py --core-only \\
        --ids <cell-id>,<cell-id> --out-dir <results-dir> --arch <gfx> \\
        --rocke-configs shipped --compile-cache <cache-dir> \\
        --gate tests._attention_bwd_perf_gate:gate \\
        --gate-constants <file> --expect-flavor llvm22 --deadline-s 1000
    python library/benchmarks/common/attention_bwd_bench.py --plan --core-only \\
        --wall-times <cells.jsonl>
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import importlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

CENSUS_SCHEMA = "sdpa_bwd_census/v1"
RESULTS_ENV = "ROCKE_BWD_RESULTS_DIR"
GATE_CONSTANT_NAMES = (
    "BAND_K",
    "TIE_EPS",
    "RERUN_MAX",
    "NOISY_SPREAD",
    "ATTEMPTS_NOISY",
    "CLIFF_BAND",
)
VERDICTS = ("pass", "fail", "ambiguous", "correctness_fail", "no_valid_competitor")

# Hygiene floors (fresh-process attempts per arm per cell, first discarded).
MIN_KEPT_ATTEMPTS = 5
MIN_WARMUP = 5
MIN_ITERS = 20

# Norm-relative backward tolerance; kept equal to the case table's
# TOLERANCE_BWD / ROW_FLOOR_BWD (a test asserts the equality).
TOLERANCE_BWD = {
    "fp16": {"rtol": 4e-3, "atol": 1e-4},
    "bf16": {"rtol": 2.5e-2, "atol": 1e-3},
}
ROW_FLOOR_BWD = 0.05

# Declared matrix of the general route (the class axes of a performance class).
CLASS_ARCHS = ("gfx942", "gfx950", "gfx1151")
CLASS_D = (32, 64, 128)
CLASS_DTYPES = ("bf16", "fp16")
CLASS_MASKS = ("none", "causal_tl", "causal_br", "window")
CLASS_HEAD_MODES = ("mha", "gqa", "hk_ne_hv")
CLASS_LENGTH_RELATIONS = ("square", "unequal", "s_q_1")
CLASS_LENGTH_MODES = ("fixed", "padded", "thd")
CLASS_STAGE_VECS = (8, 1)
_CLASS_AXES = (
    "d",
    "dtype",
    "mask",
    "head_mode",
    "length_relation",
    "length_mode",
    "stage_vec",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_census_path() -> Path:
    return Path(__file__).resolve().with_name("attention_bwd_census.json")


# --------------------------------------------------------------------------- census
@dataclass(frozen=True)
class Cell:
    """One backward problem. ``mask`` uses the census names (``swa`` = window)."""

    id: str
    d: int
    hq: int
    hkv: int
    b: int
    sq: int
    skv: int
    mask: str
    dtype: str
    layout: str
    window: int | None = None
    hv: int | None = None  # value heads when h_k != h_v
    seqlens: tuple[int, ...] | None = None  # THD q lengths (kv equal unless seqlens_kv)
    seqlens_kv: tuple[int, ...] | None = None
    padded_q: tuple[int, ...] | None = None  # padded batched lengths
    padded_kv: tuple[int, ...] | None = None
    stage_vec: int = 8
    weight: float = 0.0
    tag: str = ""
    core: bool = False
    model: str = ""
    parent: str | None = None

    @property
    def h_v(self) -> int:
        return self.hv if self.hv is not None else self.hkv

    @property
    def is_thd(self) -> bool:
        return self.layout == "thd"

    @property
    def kv_lengths(self) -> tuple[int, ...] | None:
        if self.seqlens is None:
            return None
        return self.seqlens_kv if self.seqlens_kv is not None else self.seqlens


def cell_from_dict(d: Mapping[str, Any]) -> Cell:
    def tup(key):
        v = d.get(key)
        return tuple(int(x) for x in v) if v is not None else None

    return Cell(
        id=d["id"],
        d=int(d["d"]),
        hq=int(d["hq"]),
        hkv=int(d["hkv"]),
        b=int(d["b"]),
        sq=int(d["sq"]),
        skv=int(d["skv"]),
        mask=d["mask"],
        dtype=d["dtype"],
        layout=d["layout"],
        window=d.get("window"),
        hv=d.get("hv"),
        seqlens=tup("seqlens"),
        seqlens_kv=tup("seqlens_kv"),
        padded_q=tup("padded_q"),
        padded_kv=tup("padded_kv"),
        stage_vec=int(d.get("stage_vec", 8)),
        weight=float(d.get("weight", 0.0)),
        tag=d.get("tag", ""),
        core=bool(d.get("core", False)),
        model=d.get("model", ""),
        parent=d.get("parent"),
    )


def cell_to_dict(c: Cell) -> dict:
    out = asdict(c)
    for k, v in list(out.items()):
        if isinstance(v, tuple):
            out[k] = list(v)
    return out


def load_census(path: os.PathLike | None = None) -> dict:
    p = Path(path) if path else default_census_path()
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("schema") != CENSUS_SCHEMA:
        raise ValueError(f"unexpected census schema {data.get('schema')!r}")
    return data


def census_cells(census: Mapping[str, Any]) -> list[Cell]:
    return [cell_from_dict(c) for c in census["configs"]]


# --------------------------------------------------------------------------- classes
@dataclass(frozen=True, order=True)
class PerfClass:
    """Request-only performance class (route is supplied by the caller)."""

    route: str
    d: int
    dtype: str
    mask: str
    head_mode: str
    length_relation: str
    length_mode: str
    stage_vec: int

    def key(self) -> str:
        return (
            f"{self.route}_d{self.d}_{self.dtype}_{self.mask}_{self.head_mode}_"
            f"{self.length_relation}_{self.length_mode}_sv{self.stage_vec}"
        )

    def axes(self) -> tuple:
        return tuple(getattr(self, a) for a in _CLASS_AXES)


def _mask_class(mask: str) -> str:
    return "window" if mask in ("swa", "window") else mask


def performance_class(cell: Cell, *, route: str = "general") -> PerfClass:
    if cell.hq == cell.hkv and cell.h_v == cell.hkv:
        head = "mha"
    elif cell.h_v != cell.hkv:
        head = "hk_ne_hv"
    else:
        head = "gqa"
    if cell.is_thd:
        q, kv = cell.seqlens, cell.kv_lengths
        if all(x == 1 for x in q):
            rel = "s_q_1"
        else:
            rel = "square" if tuple(q) == tuple(kv) else "unequal"
        mode = "thd"
    else:
        rel = (
            "s_q_1"
            if cell.sq == 1
            else ("square" if cell.sq == cell.skv else "unequal")
        )
        mode = "padded" if cell.padded_q is not None else "fixed"
    return PerfClass(
        route,
        cell.d,
        cell.dtype,
        _mask_class(cell.mask),
        head,
        rel,
        mode,
        cell.stage_vec,
    )


def declared_classes(route: str = "general") -> list[PerfClass]:
    out = []
    for d in CLASS_D:
        for dt in CLASS_DTYPES:
            for m in CLASS_MASKS:
                for h in CLASS_HEAD_MODES:
                    for r in CLASS_LENGTH_RELATIONS:
                        for lm in CLASS_LENGTH_MODES:
                            for sv in CLASS_STAGE_VECS:
                                out.append(PerfClass(route, d, dt, m, h, r, lm, sv))
    return out


def _seed_for(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def _lengths(rng: random.Random, n: int, s_max: int) -> list[int]:
    """Deterministic lengths in ``[max(1, s_max // 4), s_max]``; one equals ``s_max``."""
    lo = max(1, s_max // 4)
    out = [rng.randint(lo, s_max) for _ in range(n)]
    out[rng.randrange(n)] = s_max
    return out


def representative_for(cls: PerfClass, parent: Cell) -> Cell:
    """Derive a cell of ``cls`` from ``parent`` changing only class-defining axes."""
    rng = random.Random(_seed_for(cls.key() + "|" + parent.id))
    hq, hkv, hv = parent.hq, parent.hkv, None
    if cls.head_mode == "mha":
        hkv = hq
    elif cls.head_mode == "gqa":
        if hkv == hq:
            hkv = max(1, hq // 4)
    else:  # hk_ne_hv
        if hkv == hq:
            hkv = max(2, hq // 4)
        hv = max(1, hkv // 2)
    skv = max(parent.sq, parent.skv)
    if cls.length_relation == "square":
        sq = skv
    elif cls.length_relation == "unequal":
        sq = (
            parent.sq if parent.sq != parent.skv and parent.sq > 1 else max(2, skv // 4)
        )
    else:
        sq = 1
    mask = {"window": "swa"}.get(cls.mask, cls.mask)
    window = min(4096, max(1, skv // 2)) if mask == "swa" else None
    b = parent.b
    seqlens = seqlens_kv = padded_q = padded_kv = None
    layout = parent.layout if parent.layout != "thd" else "bshd"
    if cls.length_mode == "thd":
        layout = "thd"
        n = max(2, b)
        kv = _lengths(rng, n, skv)
        if cls.length_relation == "square":
            q = list(kv)
        elif cls.length_relation == "unequal":
            q = [max(1, x // 4) for x in kv]
        else:
            q = [1] * n
        seqlens = tuple(q)
        seqlens_kv = None if q == kv else tuple(kv)
        b, sq, skv = n, max(q), max(kv)
    elif cls.length_mode == "padded":
        b = max(2, b)
        padded_kv = tuple(_lengths(rng, b, skv))
        if cls.length_relation == "square":
            padded_q = padded_kv
        elif cls.length_relation == "unequal":
            padded_q = tuple(_lengths(rng, b, sq))
        else:
            padded_q = tuple([1] * b)
    return Cell(
        id="rep_" + cls.key(),
        d=cls.d,
        hq=hq,
        hkv=hkv,
        b=b,
        sq=sq,
        skv=skv,
        mask=mask,
        dtype=cls.dtype,
        layout=layout,
        window=window,
        hv=hv,
        seqlens=seqlens,
        seqlens_kv=seqlens_kv,
        padded_q=padded_q,
        padded_kv=padded_kv,
        stage_vec=cls.stage_vec,
        weight=0.0,
        tag="class_representative",
        core=False,
        model=parent.model,
        parent=parent.id,
    )


def class_representatives(
    cells: Sequence[Cell], *, route: str = "general"
) -> list[Cell]:
    """One generated cell per declared class without a census cell.

    The parent is the nearest core cell (fewest differing class axes; ties go
    to census order); only class-defining axes change. Deterministic.
    """
    covered = {performance_class(c, route=route) for c in cells}
    cores = [c for c in cells if c.core]
    core_axes = [(c, performance_class(c, route=route).axes()) for c in cores]
    reps = []
    for cls in declared_classes(route):
        if cls in covered:
            continue
        target = cls.axes()
        parent = min(
            core_axes,
            key=lambda ca: sum(1 for x, y in zip(ca[1], target) if x != y),
        )[0]
        reps.append(representative_for(cls, parent))
    return reps


def class_counts(cells: Sequence[Cell], *, route: str = "general") -> dict:
    declared = declared_classes(route)
    covered = {performance_class(c, route=route) for c in cells} & set(declared)
    return {
        "declared_classes": len(declared),
        "classes_with_census_cell": len(covered),
        "class_representatives": len(declared) - len(covered),
    }


def cliff_probes(core: Cell) -> list[Cell]:
    """One-axis moves of a core cell (heads ratio, S +- 1, mask, layout, dtype)."""
    out = []

    def add(tag, **kw):
        out.append(
            replace(
                core,
                id=f"{core.id}__{tag}",
                tag="cliff_probe",
                core=False,
                parent=core.id,
                weight=0.0,
                **kw,
            )
        )

    if core.is_thd:
        return out
    if core.hkv != core.hq:
        add("ratio_half", hkv=min(core.hq, core.hkv * 2))
    add("s_plus1", sq=core.sq + 1, skv=core.skv + 1)
    add("s_minus1", sq=core.sq - 1, skv=core.skv - 1)
    for m in ("none", "causal_tl", "causal_br", "swa"):
        if m != core.mask:
            add(
                f"mask_{m}",
                mask=m,
                window=(min(4096, core.skv // 2) if m == "swa" else None),
            )
    add(
        "layout_" + ("bhsd" if core.layout == "bshd" else "bshd"),
        layout="bhsd" if core.layout == "bshd" else "bshd",
    )
    add(
        "dtype_" + ("fp16" if core.dtype == "bf16" else "bf16"),
        dtype="fp16" if core.dtype == "bf16" else "bf16",
    )
    return out


# --------------------------------------------------------------------------- arms
def default_arm(cell: Cell) -> str:
    """The PyTorch default call for the cell (the bar)."""
    if cell.is_thd:
        return "varlen_flash|-"
    if cell.h_v != cell.hkv:
        return "default|expand"
    return "default|native" if cell.hq != cell.hkv else "default|mha"


def arms_for_cell(cell: Cell, *, math_ok: bool = True) -> list[str]:
    """Competitor arms: backend x GQA mode (THD: varlen flash and nested jagged)."""
    if cell.is_thd:
        return ["varlen_flash|-", "njt|default"]
    if cell.h_v != cell.hkv:
        modes = ["expand"]
    elif cell.hq != cell.hkv:
        modes = ["native", "expand"]
    else:
        modes = ["mha"]
    arms = []
    for gm in modes:
        for be in ("default", "flash", "efficient"):
            arms.append(f"{be}|{gm}")
        if math_ok:
            arms.append(f"math|{gm}")
    return arms


def math_arm_allowed(cell: Cell, max_bytes: float) -> bool:
    if cell.is_thd:
        return False
    return cell.b * cell.hq * cell.sq * cell.skv * 4 * 3 <= max_bytes


# --------------------------------------------------------------------------- verdicts
def load_gate_constants(path: os.PathLike) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    consts = data.get("constants", data)
    missing = [n for n in GATE_CONSTANT_NAMES if n not in consts]
    if missing:
        raise ValueError(f"gate constants missing: {missing}")
    return {n: consts[n] for n in GATE_CONSTANT_NAMES}


def _iqr(xs: Sequence[float]) -> float:
    q = statistics.quantiles(xs, n=4, method="inclusive")
    return q[2] - q[0]


def relative_spread(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return _iqr(xs) / statistics.median(xs)


def paired_verdict(
    rocke: Sequence[float],
    competitor: Sequence[float],
    constants: Mapping[str, float],
    *,
    reruns_done: int = 0,
) -> str:
    """Per-cell verdict from attempt medians (both arms of the same jobs).

    ``rho = M_a / M_b``; ``rho <= 1`` pass; ``rho > 1 + BAND_K * max(s_a, s_b)``
    fail; otherwise ambiguous until ``RERUN_MAX`` reruns were pooled, then pass
    iff ``rho <= 1 + TIE_EPS``.
    """
    if len(rocke) < MIN_KEPT_ATTEMPTS or len(competitor) < MIN_KEPT_ATTEMPTS:
        raise ValueError("at least five kept attempts per arm are required")
    rho = statistics.median(rocke) / statistics.median(competitor)
    if rho <= 1.0:
        return "pass"
    delta = constants["BAND_K"] * max(
        relative_spread(rocke), relative_spread(competitor)
    )
    if rho > 1.0 + delta:
        return "fail"
    if reruns_done >= int(constants["RERUN_MAX"]):
        return "pass" if rho <= 1.0 + constants["TIE_EPS"] else "fail"
    return "ambiguous"


def is_cliff(
    kappa_probe: float, kappa_parent: float, constants: Mapping[str, float]
) -> bool:
    """Cliff rule: ``kappa(p) / kappa(c) > 1 + CLIFF_BAND``."""
    return kappa_probe / kappa_parent > 1.0 + constants["CLIFF_BAND"]


def needs_noisy_rerun(medians: Sequence[float], constants: Mapping[str, float]) -> bool:
    return relative_spread(medians) > constants["NOISY_SPREAD"]


# --------------------------------------------------------------------------- planning
def plan_shards(
    cell_seconds: Sequence[float], *, job_cap_s: float = 1200.0, margin_s: float = 180.0
) -> int:
    """Shards needed (first-fit decreasing) for per-cell wall times."""
    cap = job_cap_s - margin_s
    bins: list[float] = []
    for t in sorted(cell_seconds, reverse=True):
        for i, used in enumerate(bins):
            if used + t <= cap:
                bins[i] += t
                break
        else:
            bins.append(t)
    return len(bins)


def cell_wall_estimate(
    arm_seconds: Mapping[str, float], *, attempts: int, reference_s: float
) -> float:
    """``attempts x (sum over arms of per-attempt arm time + reference)``."""
    return attempts * (sum(arm_seconds.values()) + reference_s)


def _read_wall_times(path: os.PathLike) -> dict[str, dict]:
    """Per-cell ``{arm: seconds}`` and reference seconds from private records."""
    out: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid = rec.get("id") or rec.get("cell")
        if not cid:
            continue
        entry = out.setdefault(cid, {"arms": {}, "reference_s": 0.0})
        for name, arm in (rec.get("arms") or {}).items():
            short = "|".join(name.split("|")[:2])
            wall = arm.get("wall_s")
            if wall is not None:
                entry["arms"][short] = max(entry["arms"].get(short, 0.0), float(wall))
        if rec.get("reference_s") is not None:
            entry["reference_s"] = max(entry["reference_s"], float(rec["reference_s"]))
        if rec.get("cell_wall_s") is not None:
            # a cell summary of this harness: measured wall of the whole cell
            # (attempt processes included) for ``attempts`` attempts
            entry["cell_wall_s"] = max(
                entry.get("cell_wall_s", 0.0), float(rec["cell_wall_s"])
            )
            entry["attempts"] = int(rec.get("attempts") or 1)
    return out


# --------------------------------------------------------------------------- results dir
def resolve_results_dir(cli_value: str | None) -> Path:
    """The private results directory; refuses any path inside the source tree."""
    raw = cli_value or os.environ.get(RESULTS_ENV)
    if not raw:
        raise SystemExit(f"results directory required (--out-dir or {RESULTS_ENV})")
    p = Path(raw).expanduser().resolve()
    root = _repo_root().resolve()
    if p == root or root in p.parents:
        raise SystemExit("results directory must be outside the source tree")
    p.mkdir(parents=True, exist_ok=True)
    return p


def completed_cells(path: Path) -> set[str]:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("complete"):
                done.add(rec["id"])
    return done


def append_jsonl(path: Path, rec: Mapping[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


# --------------------------------------------------------------------------- torch side
def _torch():
    import torch

    return torch


def allowed_mask(kind: str, sq: int, skv: int, window: int | None, device):
    """Boolean ``[sq, skv]`` (True = attend), ``None`` for ``none``."""
    torch = _torch()
    if kind == "none":
        return None
    i = torch.arange(sq, device=device).view(-1, 1)
    j = torch.arange(skv, device=device).view(1, -1)
    if kind == "causal_tl":
        return j <= i
    off = skv - sq
    if kind == "causal_br":
        return j <= i + off
    if kind == "swa":
        dd = (i + off) - j
        return (dd >= 0) & (dd < window)
    raise ValueError(kind)


@dataclass
class Problem:
    """Device tensors of one cell in logical ``[B, H, S, D]`` (THD: ``[T, H, D]``)."""

    cell: Cell
    q: Any
    k: Any
    v: Any
    do: Any
    cu_q: Any = None
    cu_kv: Any = None
    scale: float = 1.0


def make_problem(cell: Cell, *, seed: int = 42, device: str = "cuda") -> Problem:
    torch = _torch()
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[cell.dtype]
    gen = torch.Generator(device=device).manual_seed(seed)
    pad = 1 if cell.stage_vec == 1 else 0

    def mk(h, s):
        if cell.layout == "bhsd":
            t = torch.randn(cell.b, h, s, cell.d + pad, generator=gen, device=device)
            return t.to(dt)[..., : cell.d]
        t = torch.randn(cell.b, s, h, cell.d + pad, generator=gen, device=device)
        return t.to(dt)[..., : cell.d].transpose(1, 2)

    def mk_thd(h, total):
        t = torch.randn(total, h, cell.d + pad, generator=gen, device=device)
        return t.to(dt)[..., : cell.d]

    scale = 1.0 / math.sqrt(cell.d)
    if cell.is_thd:
        tq, tk = sum(cell.seqlens), sum(cell.kv_lengths)
        q = mk_thd(cell.hq, tq).requires_grad_(True)
        k = mk_thd(cell.hkv, tk).requires_grad_(True)
        v = mk_thd(cell.h_v, tk).requires_grad_(True)
        do = mk_thd(cell.hq, tq)

        def cu(lens):
            c = torch.zeros(len(lens) + 1, dtype=torch.int32, device=device)
            c[1:] = torch.tensor(lens, device=device).cumsum(0)
            return c

        return Problem(
            cell, q, k, v, do, cu(cell.seqlens), cu(cell.kv_lengths), scale=scale
        )
    q = mk(cell.hq, cell.sq).requires_grad_(True)
    k = mk(cell.hkv, cell.skv).requires_grad_(True)
    v = mk(cell.h_v, cell.skv).requires_grad_(True)
    do = mk(cell.hq, cell.sq)
    if cell.padded_q is not None:
        # dO = 0 on padded query rows: those rows then contribute nothing to dK/dV
        # in any arm, so every arm computes the same work as the length-aware one.
        live = torch.arange(cell.sq, device=device).view(1, 1, -1, 1) < torch.tensor(
            cell.padded_q, device=device
        ).view(-1, 1, 1, 1)
        do.masked_fill_(~live, 0)  # in place: keeps the storage layout
    return Problem(cell, q, k, v, do, scale=scale)


def _expand_heads(t, h_to: int):
    h = t.shape[1]
    return t if h == h_to else t.repeat_interleave(h_to // h, dim=1)


def _dense_attn_mask(problem: Problem):
    """Boolean attend mask, ``[sq, skv]`` or ``[B, 1, sq, skv]`` for padded cells.

    Padded cells use the per-batch lengths for the mask geometry (bottom-right
    and window offsets come from ``len_kv - len_q`` of each batch, as in the
    length-aware kernels); keys past ``len_kv`` are masked; padded query rows
    attend every key (their dO is zero, so they contribute nothing).
    """
    torch = _torch()
    c = problem.cell
    dev = problem.q.device
    if c.padded_kv is None and c.padded_q is None:
        return allowed_mask(c.mask, c.sq, c.skv, c.window, dev)
    lq_all = c.padded_q or (c.sq,) * c.b
    lk_all = c.padded_kv or (c.skv,) * c.b
    out = torch.zeros(c.b, 1, c.sq, c.skv, dtype=torch.bool, device=dev)
    for b, (lq, lk) in enumerate(zip(lq_all, lk_all)):
        m = allowed_mask(c.mask, lq, lk, c.window, dev)
        out[b, 0, :lq, :lk] = True if m is None else m
        out[b, 0, lq:, :] = True
    return out


def arm_forward(arm: str, problem: Problem) -> Callable[[], Any]:
    """Closure producing the arm's attention output (forward, autograd-tracked)."""
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    c = problem.cell
    backend, mode = arm.split("|")
    if c.is_thd:
        return _thd_forward(backend, mode, problem)
    backends = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
    }

    def fwd():
        k, v, kw = problem.k, problem.v, {}
        if mode == "expand":
            k, v = _expand_heads(k, c.hq), _expand_heads(v, c.hq)
        elif mode == "native":
            kw["enable_gqa"] = True
        ctx = (
            sdpa_kernel([backends[backend]])
            if backend in backends
            else contextlib.nullcontext()
        )
        with ctx:
            if c.padded_q is not None or c.padded_kv is not None or c.mask == "swa":
                return F.scaled_dot_product_attention(
                    problem.q, k, v, attn_mask=_dense_attn_mask(problem), **kw
                )
            if c.mask == "none":
                return F.scaled_dot_product_attention(problem.q, k, v, **kw)
            if c.mask == "causal_tl":
                return F.scaled_dot_product_attention(
                    problem.q, k, v, is_causal=True, **kw
                )
            from torch.nn.attention.bias import causal_lower_right

            return F.scaled_dot_product_attention(
                problem.q, k, v, attn_mask=causal_lower_right(c.sq, c.skv), **kw
            )

    return fwd


def _thd_forward(backend: str, mode: str, problem: Problem) -> Callable[[], Any]:
    torch = _torch()
    c = problem.cell
    if c.mask not in ("none", "causal_tl"):
        raise NotImplementedError("THD competitor arms cover none / causal_tl only")
    mq, mk = max(c.seqlens), max(c.kv_lengths)
    if backend == "varlen_flash":
        from torch.nn.attention.varlen import varlen_attn

        ws = (-1, 0) if c.mask == "causal_tl" else (-1, -1)

        def fwd_v():
            kk = _expand_heads(problem.k.unsqueeze(0).transpose(1, 2), c.hq).transpose(
                1, 2
            )[0]
            vv = _expand_heads(problem.v.unsqueeze(0).transpose(1, 2), c.hq).transpose(
                1, 2
            )[0]
            return varlen_attn(
                problem.q, kk, vv, problem.cu_q, problem.cu_kv, mq, mk, window_size=ws
            )

        return fwd_v
    import torch.nn.functional as F

    offs_q, offs_k = problem.cu_q.to(torch.int64), problem.cu_kv.to(torch.int64)

    def fwd_n():
        kk = _expand_heads(problem.k.unsqueeze(0).transpose(1, 2), c.hq).transpose(
            1, 2
        )[0]
        vv = _expand_heads(problem.v.unsqueeze(0).transpose(1, 2), c.hq).transpose(
            1, 2
        )[0]
        nq = torch.nested.nested_tensor_from_jagged(problem.q, offs_q).transpose(1, 2)
        nk = torch.nested.nested_tensor_from_jagged(kk, offs_k).transpose(1, 2)
        nv = torch.nested.nested_tensor_from_jagged(vv, offs_k).transpose(1, 2)
        o = F.scaled_dot_product_attention(
            nq, nk, nv, is_causal=(c.mask == "causal_tl")
        )
        return o.transpose(1, 2).values()

    return fwd_n


# --------------------------------------------------------------------------- reference
@dataclass
class Reference:
    o: Any
    lse: Any  # natural log, fp32
    dq: Any
    dk: Any
    dv: Any
    dead_q: Any  # boolean rows with no attended key
    seconds: float = 0.0


def _segments(problem: Problem):
    """Yield ``(q_slice, kv_slice, b, sq, skv)`` per independent attention problem."""
    c = problem.cell
    if c.is_thd:
        oq = ok = 0
        for i, (lq, lk) in enumerate(zip(c.seqlens, c.kv_lengths)):
            yield slice(oq, oq + lq), slice(ok, ok + lk), i, lq, lk
            oq, ok = oq + lq, ok + lk
    else:
        for b in range(c.b):
            lq = c.padded_q[b] if c.padded_q is not None else c.sq
            lk = c.padded_kv[b] if c.padded_kv is not None else c.skv
            yield slice(0, lq), slice(0, lk), b, lq, lk


def reference_backward(
    problem: Problem, *, q_block: int = 1024, head_block: int = 8
) -> Reference:
    """fp32 chunked forward (O, natural-log LSE) and backward on the same inputs.

    Chunked over query-head groups and query blocks, so no full score matrix
    is materialised. Masked-out keys contribute nothing; a query row with no
    attended key has O = 0, LSE = -inf and zero gradients. Padded batched rows
    outside the lengths get zero gradients (THD rows are all inside segments).

    ``Dsum = rowsum(dO * O)`` uses O rounded to the I/O dtype: that is the O a
    backward consumes (the rocKE arm reads ``Reference.o`` cast to the I/O
    dtype), and the convention of the float64 test oracle that calibrated
    ``TOLERANCE_BWD`` (it evaluates on exactly the stored inputs). With the
    unrounded O, rows that attend two or three keys (exact dQ nearly cancels)
    would carry the O rounding error as a reference error.
    """
    torch = _torch()
    t0 = time.perf_counter()
    c = problem.cell
    f32 = torch.float32
    q, k, v, do = (
        t.detach().to(f32) for t in (problem.q, problem.k, problem.v, problem.do)
    )
    if c.is_thd:  # -> [1, H, T, D]
        q, k, v, do = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v, do))
    o = torch.zeros_like(q)
    lse = torch.full(q.shape[:-1], float("-inf"), device=q.device, dtype=f32)
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    gk, gv = c.hq // c.hkv, c.hq // c.h_v
    for qs, ks, b, lq, lk in _segments(problem):
        bi = 0 if c.is_thd else b
        mfull = allowed_mask(c.mask, lq, lk, c.window, q.device)
        for h0 in range(0, c.hq, head_block):
            hs = slice(h0, min(c.hq, h0 + head_block))
            heads = torch.arange(hs.start, hs.stop, device=q.device)
            kh, vh = heads // gk, heads // gv
            kk = k[bi, kh][:, ks]  # [hb, lk, D]
            vv = v[bi, vh][:, ks]
            for r0 in range(0, lq, q_block):
                r1 = min(lq, r0 + q_block)
                rows = slice(qs.start + r0, qs.start + r1)
                qq, dd = q[bi, hs, rows], do[bi, hs, rows]
                s = torch.matmul(qq, kk.transpose(-1, -2)) * problem.scale
                if mfull is not None:
                    s = s.masked_fill(~mfull[r0:r1].unsqueeze(0), float("-inf"))
                l_ = torch.logsumexp(s, dim=-1)
                p = torch.exp(s - l_.unsqueeze(-1)).nan_to_num(0.0)
                oo = torch.matmul(p, vv)
                dp = torch.matmul(dd, vv.transpose(-1, -2))
                o_io = oo.to(problem.q.dtype).to(f32)
                dsum = (dd * o_io).sum(-1, keepdim=True)
                ds = p * (dp - dsum)
                o[bi, hs, rows] = oo
                lse[bi, hs, rows] = l_
                dq[bi, hs, rows] = torch.matmul(ds, kk) * problem.scale
                dkp = torch.matmul(ds.transpose(-1, -2), qq) * problem.scale
                dvp = torch.matmul(p.transpose(-1, -2), dd)
                for j, (a, bb) in enumerate(zip(kh.tolist(), vh.tolist())):
                    dk[bi, a, ks] += dkp[j]
                    dv[bi, bb, ks] += dvp[j]
    dead = ~torch.isfinite(lse)
    if c.is_thd:
        o, dq, dk, dv = (t[0].transpose(0, 1) for t in (o, dq, dk, dv))
        lse, dead = lse[0].transpose(0, 1), dead[0].transpose(0, 1)
    if q.is_cuda:
        torch.cuda.synchronize()
    return Reference(o, lse, dq, dk, dv, dead, time.perf_counter() - t0)


def kv_growth(cell: Cell) -> float:
    s_kv = max(cell.kv_lengths) if cell.is_thd else cell.skv
    return max(1.0, math.sqrt(s_kv / 1024.0))


def _valid_rows(problem: Problem, which: str):
    """Boolean mask over the leading dims of a gradient (rows inside the lengths)."""
    torch = _torch()
    c = problem.cell
    if c.is_thd:
        return None
    lens = c.padded_q if which == "dq" else c.padded_kv
    if lens is None:
        return None
    s = c.sq if which == "dq" else c.skv
    h = c.hq if which == "dq" else (c.hkv if which == "dk" else c.h_v)
    m = torch.arange(s, device=problem.q.device).view(1, -1) < torch.tensor(
        lens, device=problem.q.device
    ).view(-1, 1)
    return m.view(c.b, 1, s).expand(c.b, h, s)


def compare_to_reference(
    problem: Problem,
    grads: Sequence[Any],
    ref: Reference,
    tol: Mapping | None = None,
    *,
    row_check: bool = True,
) -> list[str]:
    """Norm-relative check (global and per-row) of dq, dk, dv; [] when correct.

    ``row_check=False`` keeps the global bound and the finiteness check only
    (recorded per attempt when used for competitor arms).
    """
    torch = _torch()
    t = tol or TOLERANCE_BWD[problem.cell.dtype]
    rtol, atol = t["rtol"], t["atol"]
    g = kv_growth(problem.cell)
    probs = []
    for name, got, want in zip(("dq", "dk", "dv"), grads, (ref.dq, ref.dk, ref.dv)):
        if got is None:
            probs.append(f"{name}: missing")
            continue
        got = got.detach().to(torch.float32)
        if tuple(got.shape) != tuple(want.shape):
            probs.append(f"{name}: shape")
            continue
        rows = _valid_rows(problem, name)
        gv, wv = (got, want) if rows is None else (got[rows], want[rows])
        if gv.numel() == 0:
            continue
        if not torch.isfinite(gv).all():
            probs.append(f"{name}: non-finite")
            continue
        diff = (gv - wv).abs()
        mx = float(wv.abs().max())
        if float(diff.max()) > atol + rtol * g * mx:
            probs.append(f"{name}: global")
            continue
        if not row_check:
            continue
        row_err = diff.amax(dim=-1)
        row_bound = atol + rtol * g * torch.clamp(
            wv.abs().amax(dim=-1), min=ROW_FLOOR_BWD * mx
        )
        if bool((row_err > row_bound).any()):
            probs.append(f"{name}: row")
    return probs


# --------------------------------------------------------------------------- fp64 samples
# Cells with at least this many score elements (sum over batches / sequences
# of h_q * len_q * len_kv) also get the fp64 sampled oracle of the timed-shape
# gate: sampled query rows of dQ and sampled keys of dK / dV.
FP64_SAMPLE_MIN_SCORES = 1 << 30
FP64_ROWS = 16
FP64_KEYS = 16
# The fp32 chunked reference must agree with the fp64 samples to this
# fraction of the gradient's largest magnitude (a reference self-check).
REFERENCE_SELF_RTOL = 1e-4


def score_elements(cell: Cell) -> int:
    if cell.is_thd:
        return cell.hq * sum(a * b for a, b in zip(cell.seqlens, cell.kv_lengths))
    lq = cell.padded_q or (cell.sq,) * cell.b
    lk = cell.padded_kv or (cell.skv,) * cell.b
    return cell.hq * sum(a * b for a, b in zip(lq, lk))


def needs_fp64_samples(cell: Cell, threshold: int = FP64_SAMPLE_MIN_SCORES) -> bool:
    return score_elements(cell) >= threshold


def _spread(n: int, k: int) -> list[int]:
    """At most ``k`` (>= 3) distinct indices in ``[0, n)``: 0, 1, ``n - 1`` and
    an even spread between them (every index when ``n <= k``)."""
    if n <= k:
        return list(range(n))
    pick = {1} | {round(i * (n - 1) / (k - 2)) for i in range(k - 1)}
    return sorted(pick)


@dataclass
class Samples:
    """fp64 values of sampled gradient rows, indexed into the gradient tensors.

    ``*_idx`` are lists of leading-dim index tuples (``(b, h, s)`` batched,
    ``(t, h)`` THD); ``dq`` / ``dk`` / ``dv`` are ``[n, D]`` fp64 tensors.
    """

    dq_idx: list
    dq: Any
    dk_idx: list
    dk: Any
    dv_idx: list
    dv: Any

    def to(self, device) -> "Samples":
        return Samples(
            self.dq_idx,
            self.dq.to(device),
            self.dk_idx,
            self.dk.to(device),
            self.dv_idx,
            self.dv.to(device),
        )


def fp64_samples(
    problem: Problem,
    o_io=None,
    *,
    n_rows: int = FP64_ROWS,
    n_keys: int = FP64_KEYS,
    row_block: int = 256,
) -> Samples:
    """fp64 dQ rows and dK / dV keys from the same inputs (no stored state).

    Sampled (batch or sequence, K head) pairs: the first non-empty sequence
    with K head 0 and the last one with the last K head. For each pair, every
    query head of the K head's group (and of the matching V head's group) is
    recomputed in fp64 over all its query rows, chunked by ``row_block``, so
    the sampled keys get their complete dK / dV; the sampled query rows of the
    group's first head get dQ. ``o_io``: the I/O-dtype O the backward
    consumes (``Reference.o`` cast to the I/O dtype; logical layout of the
    gradients); ``Dsum`` is taken from it, as in :func:`reference_backward`.
    Without it, O is the fp64 forward rounded to the I/O dtype.
    """
    torch = _torch()
    c = problem.cell
    f64 = torch.float64
    q, k, v, do = (
        t.detach().to(f64) for t in (problem.q, problem.k, problem.v, problem.do)
    )
    oio = None if o_io is None else o_io.detach().to(f64)
    if c.is_thd:
        q, k, v, do = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v, do))
        if oio is not None:
            oio = oio.transpose(0, 1).unsqueeze(0)
    gk, gv = c.hq // c.hkv, c.hq // c.h_v
    segs = [s for s in _segments(problem) if s[3] > 0 and s[4] > 0]
    pairs = []
    if segs:
        pairs.append((segs[0], 0))
        if (segs[-1][2], c.hkv - 1) != (segs[0][2], 0):
            pairs.append((segs[-1], c.hkv - 1))
    out = {n: ([], []) for n in ("dq", "dk", "dv")}
    for (qs, ks, b, lq, lk), hk in pairs:
        bi = 0 if c.is_thd else b
        hv = min(hk * c.h_v // c.hkv, c.h_v - 1)
        heads_k = [h for h in range(c.hq) if h // gk == hk]
        heads_v = [h for h in range(c.hq) if h // gv == hv]
        h_dq = heads_k[0]
        keys = _spread(lk, n_keys)
        rows = _spread(lq, n_rows)
        mask = allowed_mask(c.mask, lq, lk, c.window, q.device)
        kidx = torch.tensor(keys, device=q.device)
        dk_acc = torch.zeros(len(keys), c.d, dtype=f64, device=q.device)
        dv_acc = torch.zeros_like(dk_acc)
        dq_rows = {}
        for h in sorted(set(heads_k) | set(heads_v)):
            kk = k[bi, h // gk, ks]
            vv = v[bi, h // gv, ks]
            for r0 in range(0, lq, row_block):
                r1 = min(lq, r0 + row_block)
                rr = slice(qs.start + r0, qs.start + r1)
                qq, dd = q[bi, h, rr], do[bi, h, rr]
                s = torch.matmul(qq, kk.transpose(0, 1)) * problem.scale
                if mask is not None:
                    s = s.masked_fill(~mask[r0:r1], float("-inf"))
                lse = torch.logsumexp(s, dim=-1, keepdim=True)
                p = torch.exp(s - lse).nan_to_num(0.0)
                if oio is None:
                    o = torch.matmul(p, vv).to(problem.q.dtype).to(f64)
                else:
                    o = oio[bi, h, rr]
                dsum = (dd * o).sum(-1, keepdim=True)
                ds = p * (torch.matmul(dd, vv.transpose(0, 1)) - dsum)
                if h in heads_k:
                    dk_acc += torch.matmul(ds[:, kidx].transpose(0, 1), qq)
                if h in heads_v:
                    dv_acc += torch.matmul(p[:, kidx].transpose(0, 1), dd)
                if h == h_dq:
                    for r in rows:
                        if r0 <= r < r1:
                            dq_rows[r] = torch.matmul(ds[r - r0], kk) * problem.scale
        dk_acc *= problem.scale
        for r in rows:
            out["dq"][0].append((qs.start + r, h_dq) if c.is_thd else (b, h_dq, r))
            out["dq"][1].append(dq_rows[r])
        for j, key in enumerate(keys):
            tok = ks.start + key
            out["dk"][0].append((tok, hk) if c.is_thd else (b, hk, key))
            out["dk"][1].append(dk_acc[j])
            out["dv"][0].append((tok, hv) if c.is_thd else (b, hv, key))
            out["dv"][1].append(dv_acc[j])

    def stack(xs):
        if not xs:
            return torch.zeros(0, c.d, dtype=f64, device=q.device)
        return torch.stack(xs)

    return Samples(
        out["dq"][0],
        stack(out["dq"][1]),
        out["dk"][0],
        stack(out["dk"][1]),
        out["dv"][0],
        stack(out["dv"][1]),
    )


def _gather(t, idx: list):
    torch = _torch()
    if not idx:
        return t.new_zeros((0, t.shape[-1]))
    cols = list(zip(*idx))
    return t[tuple(torch.tensor(cl, device=t.device) for cl in cols)]


def compare_to_samples(
    problem: Problem,
    grads: Sequence[Any],
    samples: Samples,
    ref: "Reference",
    tol: Mapping | None = None,
    *,
    row_check: bool = True,
) -> list[str]:
    """The fp64 sampled check: ``TOLERANCE_BWD`` with the tensor-wide maximum
    of the reference gradient as the global scale; ``[]`` when correct."""
    torch = _torch()
    t = tol or TOLERANCE_BWD[problem.cell.dtype]
    rtol, atol = t["rtol"], t["atol"]
    g = kv_growth(problem.cell)
    probs = []
    for name, got_t, full in zip(("dq", "dk", "dv"), grads, (ref.dq, ref.dk, ref.dv)):
        idx, want = getattr(samples, f"{name}_idx"), getattr(samples, name)
        if not idx:
            continue
        got = _gather(got_t.detach(), idx).to(torch.float64)
        want = want.to(got.device)
        if not torch.isfinite(got).all():
            probs.append(f"{name}: fp64_non_finite")
            continue
        mx = float(full.abs().max())
        diff = (got - want).abs()
        if float(diff.max()) > atol + rtol * g * mx:
            probs.append(f"{name}: fp64_global")
            continue
        if row_check:
            bound = atol + rtol * g * torch.clamp(
                want.abs().amax(dim=-1), min=ROW_FLOOR_BWD * mx
            )
            if bool((diff.amax(dim=-1) > bound).any()):
                probs.append(f"{name}: fp64_row")
    return probs


def reference_self_check(ref: "Reference", samples: Samples) -> list[str]:
    """The fp32 reference against the fp64 samples (must agree far inside
    the tolerance; a failure means the reference cannot gate anything)."""
    torch = _torch()
    probs = []
    for name in ("dq", "dk", "dv"):
        idx, want = getattr(samples, f"{name}_idx"), getattr(samples, name)
        if not idx:
            continue
        full = getattr(ref, name)
        got = _gather(full, idx).to(torch.float64)
        mx = max(float(full.abs().max()), 1e-30)
        if float((got - want.to(got.device)).abs().max()) > REFERENCE_SELF_RTOL * mx:
            probs.append(f"reference: {name}")
    return probs


# --------------------------------------------------------------------------- reference cache
def _tensor_digest(t) -> str:
    flat = t.detach().reshape(-1)
    step = max(1, flat.numel() // 65536)
    sample = flat[::step].float().cpu().numpy().tobytes()
    return hashlib.sha256(sample + str(tuple(t.shape)).encode()).hexdigest()[:16]


def inputs_digest(problem: Problem) -> str:
    h = hashlib.sha256()
    for t in (problem.q, problem.k, problem.v, problem.do):
        h.update(_tensor_digest(t).encode())
    return h.hexdigest()[:16]


def _ref_path(ref_dir: Path, cell: Cell) -> Path:
    return Path(ref_dir) / f"ref-{hashlib.sha256(cell.id.encode()).hexdigest()[:16]}.pt"


def save_reference(
    ref_dir: Path, cell: Cell, problem: Problem, ref: "Reference", samples, extra
) -> Path:
    torch = _torch()
    p = _ref_path(ref_dir, cell)
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "id": cell.id,
        "digest": inputs_digest(problem),
        "seconds": ref.seconds,
        **{n: getattr(ref, n).cpu() for n in ("o", "lse", "dq", "dk", "dv", "dead_q")},
        "samples": (
            None
            if samples is None
            else {
                "dq_idx": samples.dq_idx,
                "dq": samples.dq.cpu(),
                "dk_idx": samples.dk_idx,
                "dk": samples.dk.cpu(),
                "dv_idx": samples.dv_idx,
                "dv": samples.dv.cpu(),
            }
        ),
        "extra": dict(extra),
    }
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    torch.save(blob, tmp)
    os.replace(tmp, p)
    return p


def load_reference(ref_dir: Path, cell: Cell, problem: Problem):
    """``(Reference, Samples | None, extra)`` saved for this cell and job.

    Raises ``RuntimeError`` when the inputs regenerated in this process differ
    from the ones the reference was computed on.
    """
    torch = _torch()
    p = _ref_path(ref_dir, cell)
    blob = torch.load(p, map_location="cpu", weights_only=False)
    if blob["id"] != cell.id or blob["digest"] != inputs_digest(problem):
        raise RuntimeError("reference inputs differ from this attempt's inputs")
    dev = problem.q.device
    ref = Reference(
        *(blob[n].to(dev) for n in ("o", "lse", "dq", "dk", "dv", "dead_q")),
        seconds=blob["seconds"],
    )
    s = blob["samples"]
    samples = None
    if s is not None:
        samples = Samples(
            [tuple(x) for x in s["dq_idx"]],
            s["dq"].to(dev),
            [tuple(x) for x in s["dk_idx"]],
            s["dk"].to(dev),
            [tuple(x) for x in s["dv_idx"]],
            s["dv"].to(dev),
        )
    return ref, samples, blob.get("extra", {})


# --------------------------------------------------------------------------- timing
ROCKE_PREFIX = "rocke|"


def rocke_arm_name(label: str) -> str:
    return ROCKE_PREFIX + label


AA_SUFFIX = "#aa"


def is_rocke_arm(name: str) -> bool:
    return name.startswith(ROCKE_PREFIX)


def is_aa_arm(name: str) -> bool:
    """The same-kernel control copy of a rocKE arm (never a verdict arm)."""
    return name.endswith(AA_SUFFIX)


def rocke_arms(configs: Sequence, aa_mode: str = "on") -> tuple[list, dict]:
    """``(arm names, {arm: configuration})`` of the rocKE configurations and
    their same-kernel controls: each control is the same configuration, built
    and timed as its own arm; the first configuration's (``"on"``), one per
    configuration (``"all"``) or none (``"off"``)."""
    arms = [rocke_arm_name(c.label) for c in configs]
    by_label = {rocke_arm_name(c.label): c for c in configs}
    if aa_mode not in ("on", "off", "all"):
        raise ValueError(f"aa_control={aa_mode!r} not in ('on', 'off', 'all')")
    if configs and aa_mode != "off":
        for cfg in configs if aa_mode == "all" else configs[:1]:
            aa = rocke_arm_name(cfg.label) + AA_SUFFIX
            arms.append(aa)
            by_label[aa] = cfg
    return arms, by_label


def balanced_orders(n: int) -> list[list[int]]:
    """Arm orders of a balanced (Williams) Latin square on ``n`` arms.

    Every arm takes every position equally often, and every ordered pair of
    arms is adjacent equally often, over one cycle of the returned orders
    (``n`` orders for even ``n``, ``2n`` for odd ``n``: the square plus its
    mirror). A plain rotation keeps neighbouring arms in one fixed order.
    """
    if n <= 1:
        return [list(range(n))]
    first, lo, hi = [0], 1, n - 1
    while len(first) < n:
        first.append(lo)
        lo += 1
        if len(first) < n:
            first.append(hi)
            hi -= 1
    rows = [[(x + r) % n for x in first] for r in range(n)]
    if n % 2:
        rows += [list(reversed(row)) for row in rows]
    return rows


def arm_order(arms: Sequence[str], attempt: int) -> list[str]:
    """The timing order of ``arms`` in ``attempt`` (cycles through
    :func:`balanced_orders`)."""
    rows = balanced_orders(len(arms))
    return [arms[i] for i in rows[attempt % len(rows)]]


_BACKEND_PATTERNS = (
    ("aotriton", "flash"),
    ("attn_bwd", "flash"),
    ("bwd_kernel", "flash"),
    ("fmha", "efficient"),
    ("mem_eff", "efficient"),
    ("ck_tile", "ck"),
)


def selected_backend(kernel_names: Sequence[str]) -> str:
    """A label for the attention backend a competitor call launched, from its
    kernel names (``math`` when only GEMM / softmax / elementwise kernels ran,
    ``unknown`` when nothing ran)."""
    low = [k.lower() for k in kernel_names]
    for needle, label in _BACKEND_PATTERNS:
        if any(needle in k for k in low):
            return label
    return "math" if low else "unknown"


def capture_kernels(fn: Callable[[], Any]) -> list[str]:
    """Device kernel names launched by one untimed call of ``fn``."""
    torch = _torch()
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    dev = torch.autograd.DeviceType.CUDA
    return sorted({e.key for e in prof.key_averages() if e.device_type == dev})


def _backward_closure(out, leaves, do):
    torch = _torch()

    def bwd():
        return torch.autograd.grad(out, leaves, do, retain_graph=True)

    return bwd


def time_closure(
    fn: Callable[[], Any], *, warmup: int, iters: int, windows: int, window_ms: float
) -> dict:
    """Event timing on the current stream: warmup, adaptive iters, median of windows."""
    torch = _torch()
    for _ in range(max(warmup, MIN_WARMUP)):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(3):
        fn()
    e.record()
    e.synchronize()
    est = s.elapsed_time(e) / 3
    n = int(max(max(iters, MIN_ITERS), min(200, window_ms / est if est > 0 else 0)))
    per = []
    for _ in range(windows):
        s.record()
        for _ in range(n):
            fn()
        e.record()
        e.synchronize()
        per.append(s.elapsed_time(e) / n)
    return {"median_ms": statistics.median(per), "windows_ms": per, "iters": n}


# Gate hook (the pruning set): ``module:attr`` naming an object (or a
# factory of one) with ``prepare([(config, plan)], cache_dir)`` (oracles
# computed once per job, in the reference process) and
# ``check(arch, config, plan, run_cache, cache_dir) -> [problem, ...]`` (run
# in every attempt process on the code objects of ``run_cache``, the ones
# that are timed). The committed provider lives with the backward tests
# (``tests._attention_bwd_perf_gate:gate``); it is injected by name because
# the benchmarks layer does not import the tests. It is named on the command
# line only (``--gate``), never taken from the environment, so a run's gate is
# visible in its command and in every worker's.
# Shared on-disk code-object cache (``kernels.common.attention_bwd_cache``).
HSACO_CACHE_ENV = "ROCKE_BWD_HSACO_CACHE"


def load_gate(spec: str | None):
    if not spec:
        return None
    mod, _, fn = spec.partition(":")
    obj = getattr(importlib.import_module(mod), fn)
    return obj() if callable(obj) and not hasattr(obj, "check") else obj


RockeArmFactory = Callable[[Cell, Problem, str], Callable[[], Sequence[Any]] | None]


def load_rocke_arm(spec: str | None) -> RockeArmFactory | None:
    """``module:callable`` returning ``factory(cell, problem, arch) -> run() | None``."""
    if not spec:
        return None
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)


def _rocke_configs(args: argparse.Namespace) -> list:
    from benchmarks.common import attention_bwd_rocke_arm as ra

    if not args.rocke_configs:
        return []
    if args.rocke_configs == ra.SHIPPED_LABEL:
        return [ra.SHIPPED]
    return ra.load_configs(args.rocke_configs)


def _compile_cache(args: argparse.Namespace):
    if not args.compile_cache:
        return None
    from benchmarks.common import attention_bwd_compile as bc

    tc = bc.toolchain_identity()
    return bc.CompileCache(
        Path(args.compile_cache).expanduser(),
        toolchain=tc,
        source_hash=bc.backward_source_hash(),
    )


def competitor_arms(cell: Cell, args: argparse.Namespace) -> list[str]:
    if args.competitor_arms == "none":
        return []
    arms = arms_for_cell(cell, math_ok=math_arm_allowed(cell, args.math_max_bytes))
    if args.competitor_arms == "default":
        return [a for a in arms if a == default_arm(cell)] + [
            a for a in bar_fallbacks(cell) if a in arms and a != default_arm(cell)
        ][:1]
    return arms


def bar_fallbacks(cell: Cell) -> list[str]:
    """Default-path arms in bar order (the default call first)."""
    if cell.is_thd:
        return ["varlen_flash|-", "njt|default"]
    first = default_arm(cell)
    rest = [f"default|{m}" for m in ("native", "expand", "mha")]
    out = [first] + [a for a in rest if a != first]
    return [a for a in out if a in arms_for_cell(cell)]


def _rocke_arm(cell, problem, ref, args, cfg, cache, *, load_only=False):
    from benchmarks.common import attention_bwd_rocke_arm as ra

    return ra.RockeArm(
        cell,
        problem,
        ref,
        arch=args.arch,
        config=cfg,
        cache=cache,
        load_only=load_only,
    )


def run_attempt(
    cell: Cell,
    *,
    attempt: int,
    block: int,
    arch: str,
    args: argparse.Namespace,
    rocke_factory: RockeArmFactory | None = None,
) -> dict:
    """One fresh-process attempt: per-arm gate (4.4), then interleaved timing.

    The reference (and the fp64 samples) come from the per-(cell, job) cache
    written by the reference process; every arm is gated against it in this
    process before it is timed. rocKE arms additionally pass the pruning set
    (the ``--gate`` hook) in this process, on the very code objects they time,
    and must leave their canaries intact.
    """
    torch = _torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    problem = make_problem(cell, seed=args.seed)
    ref, samples, ref_extra = load_reference(Path(args.ref_dir), cell, problem)
    configs = _rocke_configs(args)
    cache = _compile_cache(args)
    gate = load_gate(args.gate)
    arms, by_label = rocke_arms(configs, getattr(args, "aa_control", "on"))
    if rocke_factory is not None:
        arms.append("rocke|-")
    arms += competitor_arms(cell, args)
    order = arm_order(arms, attempt)
    leaves = (problem.q, problem.k, problem.v)
    rec: dict[str, Any] = {
        "id": cell.id,
        "attempt": attempt,
        "block": block,
        "arms": {},
        "competitor_row_check": args.competitor_row_check,
        "fp64_samples": samples is not None,
        "prune_gate": "on" if gate is not None else "off",
        "arm_order": order,
    }
    for arm in order:
        a: dict[str, Any] = {}
        rec["arms"][arm] = a
        t0 = time.perf_counter()
        holder = None
        try:
            row_check = is_rocke_arm(arm) or args.competitor_row_check == "on"
            if arm in by_label:
                from benchmarks.common.attention_bwd_rocke_arm import Unsupported

                try:
                    # the reference process compiled this cell's kernels; an
                    # attempt only loads them (a miss is an arm error)
                    holder = _rocke_arm(
                        cell,
                        problem,
                        ref,
                        args,
                        by_label[arm],
                        cache,
                        load_only=cache is not None,
                    )
                except Unsupported as ex:
                    a.update(status="unsupported", reason=str(ex)[:300])
                    continue
                a["kernel"] = holder.kernel_record()
                if gate is not None:
                    gp = gate.check(
                        arch, by_label[arm], holder.plan, holder.run_cache, args.ref_dir
                    )
                    if gp:
                        a.update(status="correctness_fail", problems=gp, stage="prune")
                        continue
                elif not args.allow_ungated:
                    a.update(status="ungated")
                    continue
                bwd = holder.run
            elif arm == "rocke|-":
                bwd = rocke_factory(cell, problem, arch)
                if bwd is None:
                    a["status"] = "unsupported"
                    continue
            else:
                out = arm_forward(arm, problem)()
                bwd = _backward_closure(out, leaves, problem.do)
                if attempt == 1:  # first kept attempt: what the call launches
                    try:
                        names = capture_kernels(bwd)
                        a["kernel_list"] = names[:64]
                        a["selected_backend"] = selected_backend(names)
                    except Exception as ex:  # noqa: BLE001 - recorded, not fatal
                        a["kernel_capture_error"] = type(ex).__name__
            grads = bwd()
            torch.cuda.synchronize()
            probs = compare_to_reference(problem, grads, ref, row_check=row_check)
            if samples is not None:
                probs += compare_to_samples(
                    problem, grads, samples, ref, row_check=row_check
                )
            if holder is not None:
                probs += holder.canary_problems()
            if not row_check and not probs:
                info = compare_to_reference(problem, grads, ref, row_check=True)
                a["row_check_info"] = info  # recorded, not a verdict
            del grads
            if probs:
                a.update(status="correctness_fail", problems=probs, stage="timed_shape")
                continue
            a.update(
                time_closure(
                    bwd,
                    warmup=args.warmup,
                    iters=args.iters,
                    windows=args.windows,
                    window_ms=args.window_ms,
                )
            )
            if holder is not None:
                holder.drain()
                after = holder.canary_problems()
                if after:
                    a.update(status="correctness_fail", problems=after, stage="canary")
                    continue
            a["status"] = "ok"
        except Exception as ex:  # noqa: BLE001 - recorded per arm
            a.update(status="error", error=(type(ex).__name__ + ": " + str(ex))[:400])
        finally:
            a["wall_s"] = time.perf_counter() - t0
            if holder is not None:
                try:
                    holder.drain()
                except Exception:  # noqa: BLE001
                    pass
            del holder
            torch.cuda.empty_cache()
    rec["reference_s"] = ref.seconds
    rec["reference_extra"] = ref_extra
    return rec


def compute_reference(cell: Cell, args: argparse.Namespace) -> dict:
    """The reference process of a (cell, job): fp32 chunked forward and
    backward, the fp64 samples for large cells, the self-check, the compile of
    every rocKE configuration's kernels for this cell (cache), and the
    pruning-set oracles (``--gate``). Writes the per-(cell, job) cache."""
    torch = _torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.perf_counter()
    problem = make_problem(cell, seed=args.seed)
    reused = _ref_path(Path(args.ref_dir), cell).exists()
    if reused:  # computed earlier in this job (another timing group)
        ref, samples, prev = load_reference(Path(args.ref_dir), cell, problem)
        out: dict[str, Any] = {
            "id": cell.id,
            "reference_s": 0.0,
            "reused": True,
            "fp64_samples": samples is not None,
            "fp64_s": 0.0,
            "reference_check": list(prev.get("reference_check") or []),
            "configs": {},
        }
    else:
        ref = reference_backward(problem)
        samples = None
        t1 = time.perf_counter()
        if needs_fp64_samples(cell, args.fp64_min_scores):
            samples = fp64_samples(problem, ref.o.to(problem.q.dtype))
        t2 = time.perf_counter()
        out = {
            "id": cell.id,
            "reference_s": ref.seconds,
            "reused": False,
            "fp64_samples": samples is not None,
            "fp64_s": t2 - t1,
            "reference_check": (
                reference_self_check(ref, samples) if samples is not None else []
            ),
            "configs": {},
        }
    configs = _rocke_configs(args)
    cache = _compile_cache(args)
    gate = load_gate(args.gate)
    specs = []
    for cfg in configs:
        name = rocke_arm_name(cfg.label)
        try:
            from benchmarks.common.attention_bwd_rocke_arm import Unsupported

            holder = _rocke_arm(cell, problem, ref, args, cfg, cache)
        except Unsupported as ex:
            out["configs"][name] = {"status": "unsupported", "reason": str(ex)[:300]}
            continue
        except Exception as ex:  # noqa: BLE001 - recorded
            out["configs"][name] = {
                "status": "error",
                "error": (type(ex).__name__ + ": " + str(ex))[:300],
            }
            continue
        out["configs"][name] = {"status": "built", **holder.kernel_record()}
        specs.append((cfg, holder.plan))
        del holder
    if gate is not None and specs:
        gate.prepare(specs, args.ref_dir)
    if not reused:
        save_reference(Path(args.ref_dir), cell, problem, ref, samples, out)
    out["reference_total_s"] = time.perf_counter() - t0
    return out


def environment_record() -> dict:
    import platform
    import socket

    info: dict[str, Any] = {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }
    # torch first (versions only, no device query): with torch imported, the
    # rocKE runtime binds torch's bundled HIP library, so one process never
    # loads two HIP runtimes
    try:
        torch = _torch()
        info.update(torch=torch.__version__, hip=torch.version.hip)
    except Exception as ex:  # noqa: BLE001
        info["torch_error"] = str(ex)[:200]
    info.update(_device_record())
    try:
        info["rocm_system"] = Path("/opt/rocm/.info/version").read_text().strip()
    except OSError:
        info["rocm_system"] = None
    info.update(_toolchain_record())
    info["clocks"] = _clocks()
    return info


def _device_record(device: int = 0) -> dict:
    """The device fields of the environment record, from the rocKE HIP
    runtime (``rocke.runtime.hip_module``; no torch device query): the target
    id (``gcn_arch``, features kept), the base arch (``device_arch``; the
    run's requested ``arch`` is recorded separately) and the CU count. The
    device is named by its gfx id only. A field the runtime cannot read is
    ``None``."""
    try:
        from rocke.runtime import hip_module as hm

        return {
            "gcn_arch": hm.get_device_target_id(device),
            "device_arch": hm.get_device_arch(device),
            "cu_count": hm.get_device_num_cus(device),
        }
    except Exception as ex:  # noqa: BLE001
        return {"device_error": str(ex)[:200]}


def _toolchain_record() -> dict:
    try:
        from benchmarks.common.attention_bwd_compile import toolchain_identity

        return toolchain_identity().record()
    except Exception as ex:  # noqa: BLE001
        return {"rocke_llvm_flavor": _flavor(), "toolchain_error": str(ex)[:200]}


def _flavor() -> str:
    try:
        from rocke.core.lower_llvm import _detect_llvm_flavor

        return os.environ.get("ROCKE_LLVM_FLAVOR") or _detect_llvm_flavor()
    except Exception:  # noqa: BLE001
        return os.environ.get("ROCKE_LLVM_FLAVOR", "unknown")


def _clocks() -> str | None:
    for cmd in (["rocm-smi", "--showclocks"], ["amd-smi", "metric", "-c"]):
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=8, check=False
            )
        except (OSError, subprocess.SubprocessError):
            r = None
        if r is not None and r.returncode == 0:
            return r.stdout[-2000:]
    return None


def toolchain_problems(args: argparse.Namespace) -> list[str]:
    """``--expect-flavor`` / ``--expect-comgr`` checks (tokens only)."""
    rec = _toolchain_record()
    out = []
    if args.expect_flavor and rec.get("rocke_llvm_flavor") != args.expect_flavor:
        out.append("flavor_mismatch")
    if args.expect_comgr and rec.get("comgr") != args.expect_comgr:
        out.append("comgr_mismatch")
    return out


# --------------------------------------------------------------------------- driver
def select_cells(args: argparse.Namespace) -> list[Cell]:
    census = load_census(args.census)
    cells = census_cells(census)
    if args.representatives:
        cells = cells + class_representatives(cells)
    if args.core_only:
        cells = [c for c in cells if c.core]
    if args.ids:
        want = args.ids.split(",")
        by_id = {c.id: c for c in cells}
        cells = [by_id[i] for i in want if i in by_id]
    si, sn = (int(x) for x in args.shard.split("/"))
    return [c for i, c in enumerate(cells) if i % sn == si]


def summarize_cell(
    cell: Cell,
    attempts: Sequence[dict],
    constants: Mapping | None,
    *,
    reruns_done: int = 0,
) -> dict:
    """Pooled medians / spreads per arm (the first attempt of every block is
    discarded) and, with constants, a verdict per rocKE arm."""
    kept = [a for a in attempts if a["attempt"] > 0]
    arms: dict[str, dict] = {}
    for a in kept:
        for name, r in a["arms"].items():
            e = arms.setdefault(name, {"medians": [], "statuses": []})
            e["statuses"].append(r.get("status"))
            if r.get("status") == "ok":
                e["medians"].append(r["median_ms"])
    out: dict[str, Any] = {
        "id": cell.id,
        "default_arm": default_arm(cell),
        "arms": {},
        "blocks": sorted({a.get("block", 0) for a in attempts}),
        "reruns_done": reruns_done,
    }
    for name, e in arms.items():
        ok = len(e["medians"]) == len(e["statuses"]) and len(e["medians"]) >= 1
        entry = {
            "status": "ok" if ok else _worst(e["statuses"]),
            "n": len(e["medians"]),
        }
        if ok:
            entry.update(
                median_ms=statistics.median(e["medians"]),
                spread=relative_spread(e["medians"]),
                attempt_medians_ms=e["medians"],
            )
        out["arms"][name] = entry
    bar = next(
        (
            a
            for a in bar_fallbacks(cell)
            if out["arms"].get(a, {}).get("status") == "ok"
        ),
        None,
    )
    out["bar_arm"] = bar
    out["bar_substituted"] = bar is not None and bar != default_arm(cell)
    out["order_imbalance"] = order_imbalance(kept)
    if constants is not None:
        out["verdicts"] = {
            name: cell_verdict(out, constants, name, reruns_done)
            for name in out["arms"]
            if is_rocke_arm(name) and not is_aa_arm(name)
        }
        if len(out["verdicts"]) == 1:
            out["verdict"] = next(iter(out["verdicts"].values()))
        per_arm = aa_controls(out, constants, reruns_done)
        aa = aa_control(out, constants, reruns_done)
        if aa is not None:
            out["aa_control"] = aa
            out["aa_controls"] = per_arm
            out["flagged"] = aa == "fail"
    return out


def aa_controls(
    summary: Mapping, constants: Mapping, reruns_done: int = 0
) -> dict[str, str]:
    """Per rocKE arm with a same-kernel twin (``--aa-control``): ``fail`` when
    either copy is judged slower than the other, ``incomplete`` when a copy
    has too few kept attempts, ``ok`` otherwise."""
    out = {}
    for name, e in summary["arms"].items():
        if not is_aa_arm(name):
            continue
        base_name = name[: -len(AA_SUFFIX)]
        base = summary["arms"].get(base_name, {})
        if e.get("status") != "ok" or base.get("status") != "ok":
            out[base_name] = "incomplete"
            continue
        a, b = e["attempt_medians_ms"], base["attempt_medians_ms"]
        if min(len(a), len(b)) < MIN_KEPT_ATTEMPTS:
            out[base_name] = "incomplete"
            continue
        both = (
            paired_verdict(a, b, constants, reruns_done=reruns_done),
            paired_verdict(b, a, constants, reruns_done=reruns_done),
        )
        out[base_name] = "fail" if "fail" in both else "ok"
    return out


def aa_control(
    summary: Mapping, constants: Mapping, reruns_done: int = 0
) -> str | None:
    """Verdict of the same-kernel controls of a cell: ``fail`` when a copy of
    a configuration is judged slower than the other (the job's verdicts are
    then flagged), ``incomplete`` when a copy has too few kept attempts (and
    none fails), ``ok`` otherwise; ``None`` without a control arm. One twin
    under ``--aa-control on``, one per configuration under ``all``
    (:func:`aa_controls` has the verdict of each)."""
    per = aa_controls(summary, constants, reruns_done)
    if not per:
        return None
    if "fail" in per.values():
        return "fail"
    if "incomplete" in per.values():
        return "incomplete"
    return "ok"


def order_imbalance(attempts: Sequence[dict]) -> int:
    """Largest difference, over arms, between how often the arm was timed in
    its most and in its least frequent position (0 = perfectly balanced)."""
    width = max((len(a.get("arm_order") or []) for a in attempts), default=0)
    counts: dict[str, collections.Counter] = {}
    for a in attempts:
        for pos, name in enumerate(a.get("arm_order") or []):
            counts.setdefault(name, collections.Counter())[pos] += 1
    worst = 0
    for c in counts.values():
        per = [c.get(p, 0) for p in range(width)]
        worst = max(worst, max(per) - min(per))
    return worst


def _worst(statuses: Iterable[str | None]) -> str:
    for s in ("correctness_fail", "error", "ungated", "unsupported"):
        if s in statuses:
            return s
    return "missing"


def cell_verdict(
    summary: Mapping,
    constants: Mapping,
    arm: str = "rocke|-",
    reruns_done: int = 0,
) -> str:
    r = summary["arms"].get(arm, {})
    if r.get("status") != "ok":
        if r.get("status") in ("correctness_fail", "unsupported", "ungated"):
            return r["status"]
        return "fail"
    bar_name = summary.get("bar_arm", summary["default_arm"])
    bar = summary["arms"].get(bar_name or "", {})
    if bar.get("status") != "ok" or len(bar.get("attempt_medians_ms", ())) < (
        MIN_KEPT_ATTEMPTS
    ):
        return "no_valid_competitor"
    if len(r["attempt_medians_ms"]) < MIN_KEPT_ATTEMPTS:
        return "fail"
    return paired_verdict(
        r["attempt_medians_ms"],
        bar["attempt_medians_ms"],
        constants,
        reruns_done=reruns_done,
    )


def _spawn(cmd_extra: Sequence[str], args: argparse.Namespace, out_dir: Path) -> int:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--out-dir", str(out_dir)]
    cmd += ["--census", str(args.census)]
    for flag in (
        "arch",
        "rocke_arm",
        "rocke_configs",
        "compile_cache",
        "gate",
        "ref_dir",
        "seed",
        "warmup",
        "iters",
        "windows",
        "window_ms",
        "math_max_bytes",
        "competitor_row_check",
        "competitor_arms",
        "aa_control",
        "fp64_min_scores",
        "expect_flavor",
        "expect_comgr",
    ):
        val = getattr(args, flag)
        if val is not None:
            cmd += ["--" + flag.replace("_", "-"), str(val)]
    for flag in ("representatives", "allow_ungated"):
        if getattr(args, flag):
            cmd.append("--" + flag.replace("_", "-"))
    return subprocess.run(cmd + list(cmd_extra), check=False).returncode


def _records(path: Path, cell_id: str) -> list[dict]:
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("id") == cell_id:
                out.append(rec)
    return out


def latest_summaries(path: Path) -> dict[str, dict]:
    """The last complete summary per cell id."""
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("complete"):
                out[rec["id"]] = rec
    return out


def verdict_token(summary: Mapping) -> str:
    """One stdout token for a cell summary (counts for several rocKE arms);
    ``aa_fail:`` prefixes it when the same-kernel control failed."""
    if summary.get("flagged"):
        return "aa_fail:" + verdict_token({**summary, "flagged": False})
    if "verdict" in summary:
        return summary["verdict"]
    vs = summary.get("verdicts")
    if not vs:
        rocke = any(is_rocke_arm(n) for n in summary.get("arms", {}))
        return "unjudged" if rocke else "competitor_only"  # no gate constants
    counts = collections.Counter(vs.values())
    return "verdicts=" + ",".join(f"{k}:{counts[k]}" for k in sorted(counts))


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--census", default=str(default_census_path()))
    ap.add_argument("--ids", default=None)
    ap.add_argument("--core-only", action="store_true")
    ap.add_argument(
        "--representatives",
        action="store_true",
        help="include the generated class representatives",
    )
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--gate-constants", default=None)
    ap.add_argument("--attempts", type=int, default=MIN_KEPT_ATTEMPTS + 1)
    ap.add_argument("--deadline-s", type=float, default=1e9)
    ap.add_argument("--warmup", type=int, default=MIN_WARMUP)
    ap.add_argument("--iters", type=int, default=MIN_ITERS)
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--window-ms", type=float, default=40.0)
    ap.add_argument("--math-max-bytes", type=float, default=6e9)
    ap.add_argument(
        "--competitor-row-check",
        choices=("on", "off"),
        default="off",
        help="per-row bound for competitor arms (default off: the global bound "
        "and finiteness gate the competitor, the per-row result is recorded; "
        "the rocKE arms always use the per-row bound)",
    )
    ap.add_argument(
        "--competitor-arms",
        choices=("all", "default", "none"),
        default="all",
        help="competitor arms per cell (default: the bar and its fallback)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--rocke-arm", default=None, help="module:callable for a custom rocKE arm"
    )
    ap.add_argument(
        "--rocke-configs",
        default=None,
        help="rocKE configurations (JSON file of labelled knobs) or 'shipped'",
    )
    ap.add_argument(
        "--compile-cache",
        default=os.environ.get(HSACO_CACHE_ENV) or None,
        help="on-disk compile cache directory (default: $ROCKE_BWD_HSACO_CACHE, "
        "else a job-local directory beside the reference cache)",
    )
    ap.add_argument(
        "--gate",
        default=None,
        help="module:callable of the pruning-set gate provider (explicit only: "
        "no environment default; the sweep driver passes it, and every worker "
        "process gets it on its command line)",
    )
    ap.add_argument(
        "--allow-ungated",
        action="store_true",
        help="time rocKE arms without the pruning-set gate (local development)",
    )
    ap.add_argument(
        "--aa-control",
        choices=("on", "off", "all"),
        default="on",
        help="time a second copy of the first rocKE configuration ('on') or of "
        "every rocKE configuration ('all': each arm with its own noise control, "
        "e.g. arms with different runtime values) as a same-kernel control; a "
        "cell whose copies differ is flagged",
    )
    ap.add_argument("--fp64-min-scores", type=int, default=FP64_SAMPLE_MIN_SCORES)
    ap.add_argument("--ref-dir", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--expect-flavor", default=None)
    ap.add_argument("--expect-comgr", default=None)
    ap.add_argument(
        "--rerun-ambiguous",
        action="store_true",
        help="add a block of attempts to every ambiguous cell already recorded",
    )
    ap.add_argument(
        "--count", action="store_true", help="print census and class counts"
    )
    ap.add_argument("--list-arms", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--wall-times", default=None, help="private records for --plan")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--worker-ref", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--attempt", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--block", type=int, default=0, help=argparse.SUPPRESS)
    return ap


def _plan(args: argparse.Namespace, cells: Sequence[Cell]) -> int:
    walls = _read_wall_times(args.wall_times)
    secs, missing = [], 0
    for c in cells:
        w = walls.get(c.id)
        if not w or (not w["arms"] and not w.get("cell_wall_s")):
            missing += 1
            continue
        if w.get("cell_wall_s"):
            secs.append(
                w["cell_wall_s"] * args.attempts / max(1, w.get("attempts", 1))
                + w["reference_s"]
            )
        else:
            secs.append(
                cell_wall_estimate(w["arms"], attempts=args.attempts, reference_s=0.0)
                + w["reference_s"]
            )
    print(
        f"plan cells={len(cells)} with_wall_times={len(secs)} missing={missing} "
        f"shards={plan_shards(secs)}"
    )
    return 0


def _run_cell(
    c: Cell,
    args: argparse.Namespace,
    out_dir: Path,
    constants: Mapping | None,
    *,
    block: int,
    reruns_done: int,
) -> dict | None:
    attempts_path = out_dir / "attempts.jsonl"
    refs_path = out_dir / "references.jsonl"
    t0 = time.time()
    # always: computes (or reuses) the reference, compiles this run's rocKE
    # configurations and prepares their pruning-set oracles
    rc = _spawn(["--worker-ref", "--ids", c.id], args, out_dir)
    if rc != 0 or not _ref_path(Path(args.ref_dir), c).exists():
        return {"id": c.id, "complete": True, "status": "reference_error"}
    ref_rec = (_records(refs_path, c.id) or [{}])[-1]
    if ref_rec.get("reference_check"):
        return {"id": c.id, "complete": True, "status": "reference_fail"}
    have = {
        r["attempt"]
        for r in _records(attempts_path, c.id)
        if r.get("block", 0) == block
    }
    target = args.attempts
    att = 0
    while att < target:
        if att not in have:
            _spawn(
                [
                    "--worker",
                    "--ids",
                    c.id,
                    "--attempt",
                    str(att),
                    "--block",
                    str(block),
                ],
                args,
                out_dir,
            )
        att += 1
        if att == target and constants is not None:
            summary = summarize_cell(c, _records(attempts_path, c.id), None)
            noisy = [
                e["attempt_medians_ms"]
                for name, e in summary["arms"].items()
                if (name == summary["bar_arm"] or is_rocke_arm(name))
                and e.get("status") == "ok"
            ]
            if any(needs_noisy_rerun(m, constants) for m in noisy):
                target = max(target, int(constants["ATTEMPTS_NOISY"]) + 1)
    summary = summarize_cell(
        c, _records(attempts_path, c.id), constants, reruns_done=reruns_done
    )
    summary.update(
        complete=True,
        arch=args.arch,
        reference_s=ref_rec.get("reference_s"),
        fp64_samples=ref_rec.get("fp64_samples"),
        cell_wall_s=time.time() - t0,
        attempts=target,
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.worker or args.worker_ref:
        import warnings

        # PyTorch's backend-selection notices name source files and lines;
        # they carry no result, and job logs keep tokens only.
        warnings.filterwarnings("ignore", category=UserWarning)
    if args.count:
        cells = census_cells(load_census(args.census))
        print(f"census cells={len(cells)} core={sum(c.core for c in cells)}")
        for arch in CLASS_ARCHS:
            cc = class_counts(cells)
            print(f"{arch} general " + " ".join(f"{k}={v}" for k, v in cc.items()))
        return 0
    cells = select_cells(args)
    if args.list_arms:
        for c in cells:
            arms = arms_for_cell(c, math_ok=math_arm_allowed(c, args.math_max_bytes))
            print(f"{c.id} default={default_arm(c)} arms={','.join(arms)}")
        return 0
    if args.plan:
        if not args.wall_times:
            raise SystemExit("--plan needs --wall-times")
        return _plan(args, cells)

    out_dir = resolve_results_dir(args.out_dir)
    if args.worker or args.worker_ref:
        bad = toolchain_problems(args)
        if bad:
            print("abort: " + ",".join(bad), flush=True)
            return 3
        if args.worker_ref:
            rec = compute_reference(cells[0], args)
            append_jsonl(out_dir / "references.jsonl", rec)
            print(
                "reference "
                + ("ok" if not rec["reference_check"] else "reference_fail"),
                flush=True,
            )
            return 0
        rec = run_attempt(
            cells[0],
            attempt=args.attempt,
            block=args.block,
            arch=args.arch or "unknown",
            args=args,
            rocke_factory=load_rocke_arm(args.rocke_arm),
        )
        rec["env"] = environment_record()
        append_jsonl(out_dir / "attempts.jsonl", rec)
        print(
            " ".join(f"{n}={a.get('status')}" for n, a in rec["arms"].items()),
            flush=True,
        )
        return 0

    bad = toolchain_problems(args)
    if bad:
        print("abort: " + ",".join(bad), flush=True)
        return 3
    if args.attempts < MIN_KEPT_ATTEMPTS + 1:
        raise SystemExit("at least six attempts (first discarded) are required")
    if _rocke_configs(args) and not args.gate and not args.allow_ungated:
        raise SystemExit("rocKE arms need the pruning-set gate (--gate)")
    constants = (
        load_gate_constants(args.gate_constants) if args.gate_constants else None
    )
    own_ref_dir = args.ref_dir is None
    if own_ref_dir:
        base = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp"
        args.ref_dir = str(Path(base) / f"rocke-bwd-ref-{os.getpid()}")
    Path(args.ref_dir).mkdir(parents=True, exist_ok=True)
    if args.compile_cache is None and _rocke_configs(args):
        # The reference process of each cell compiles into this job-local
        # cache and every attempt process only loads from it (one compile per
        # kernel and job, none inside the timed processes).
        args.compile_cache = str(Path(args.ref_dir) / "hsaco")
    cells_path = out_dir / "cells.jsonl"
    prior = latest_summaries(cells_path)
    append_jsonl(
        out_dir / "env.jsonl",
        {"t": time.time(), "arch": args.arch, **environment_record()},
    )
    t_start = time.time()
    counts: collections.Counter = collections.Counter()
    n_cells = 0
    longest = 0.0
    for c in cells:
        block, reruns = 0, 0
        if c.id in prior:
            p = prior[c.id]
            if (
                not args.rerun_ambiguous
                or "ambiguous" not in (p.get("verdicts") or {}).values()
            ):
                continue
            reruns = int(p.get("reruns_done", 0)) + 1
            if constants is not None and reruns > int(constants["RERUN_MAX"]):
                continue
            block = max(p.get("blocks") or [0]) + 1
        elapsed = time.time() - t_start
        if elapsed > args.deadline_s or elapsed + longest > args.deadline_s:
            print(f"deadline: stopped cells_done={n_cells}", flush=True)
            break
        summary = _run_cell(
            c, args, out_dir, constants, block=block, reruns_done=reruns
        )
        append_jsonl(cells_path, summary)
        if own_ref_dir:  # a caller-given cache is the caller's (one per job)
            try:
                os.remove(_ref_path(Path(args.ref_dir), c))
            except OSError:
                pass
        n_cells += 1
        longest = max(longest, summary.get("cell_wall_s") or 0.0)
        if summary.get("status"):
            tok = summary["status"]
            print(f"cell {c.id}: {tok}", flush=True)
            counts[tok] += 1
            continue
        n_ok = sum(1 for a in summary["arms"].values() if a["status"] == "ok")
        n_bad = len(summary["arms"]) - n_ok
        tok = verdict_token(summary)
        for v in (summary.get("verdicts") or {}).values():
            counts[v] += 1
        if not summary.get("verdicts"):
            counts[tok] += 1
        print(f"cell {c.id}: {tok} arms_ok={n_ok} arms_not_ok={n_bad}", flush=True)
    append_jsonl(
        out_dir / "env.jsonl", {"t": time.time(), "end": True, "clocks": _clocks()}
    )
    if own_ref_dir:
        import shutil

        shutil.rmtree(args.ref_dir, ignore_errors=True)
    print(
        f"summary cells={n_cells} "
        + " ".join(f"{k}={counts[k]}" for k in sorted(counts)),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
