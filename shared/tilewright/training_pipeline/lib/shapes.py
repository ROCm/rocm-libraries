# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Benchable-shape filter and per-cell shape synthesis for stage01.

`ShapeFilter.accepts(m, n, k, b)` keeps a shape hipblaslt-bench can run for
the configured dtypes without exceeding the device-memory budget or int32
indexing, within the cost caps, and not matched by any `seed.exclude` rule.
`synthesize_cell_shapes` draws distinct shapes inside one grid cell (or a
sub-cell box) that the filter accepts.

Config keys read here:
  bench.device_memory_gib  device memory in GiB; the footprint budget is
                           MEMORY_SAFE_FRACTION of it. When absent, the
                           smallest GPU memory reported by the KFD sysfs
                           topology is used, and without that a conservative
                           FALLBACK_DEVICE_MEMORY_GIB.
  bench.rotating_mb        rotating-buffer cap, counted on top of the
                           working set (default 512).
  seed.exclude             list of exclusion rules, e.g.
                           [{max_mn: 8}, {min_k: 65536}]; see
                           `parse_exclude_rules`.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import yaml

from .dat import mx_block_size_for_scale_mode
from .fs import atomic_write_text
from .grid import BANY_VALUES, TIER_RANGES_K, TIER_RANGES_MN

ShapeTuple = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Shape:
    m: int
    n: int
    k: int
    batch: int

    def as_tuple(self) -> ShapeTuple:
        return (self.m, self.n, self.k, self.batch)


# ── device footprint ─────────────────────────────────────────────────────────

# Storage bytes per element by hipblaslt-bench dtype name. Unknown names count
# as 4 bytes so a footprint is over- rather than under-estimated.
BYTES_PER_ELEMENT: Dict[str, float] = {
    "f64_r": 8,
    "f32_r": 4,
    "xf32_r": 4,
    "i32_r": 4,
    "f16_r": 2,
    "bf16_r": 2,
    "f8_r": 1,
    "bf8_r": 1,
    "f8_fnuz_r": 1,
    "bf8_fnuz_r": 1,
    "i8_r": 1,
    "f6_r": 0.75,
    "bf6_r": 0.75,
    "f4_r": 0.5,
}
_UNKNOWN_BPE = 4.0


def bytes_per_element(dtype: Optional[str]) -> float:
    if not dtype:
        return _UNKNOWN_BPE
    return float(BYTES_PER_ELEMENT.get(str(dtype), _UNKNOWN_BPE))


def footprint_bytes(
    m: int,
    n: int,
    k: int,
    b: int,
    *,
    a_type: Optional[str],
    b_type: Optional[str],
    c_type: Optional[str],
    d_type: Optional[str],
    scale_a: int = 0,
    scale_b: int = 0,
) -> int:
    """Device bytes of one problem instance: A, B, C and D (the bench
    allocates C even when beta is 0), plus the MX block-scale tensors of A and
    B, one byte per block of the reduction dimension."""
    b = max(1, int(b))
    total = b * (
        m * k * bytes_per_element(a_type)
        + n * k * bytes_per_element(b_type)
        + m * n * bytes_per_element(c_type)
        + m * n * bytes_per_element(d_type)
    )
    block_a = mx_block_size_for_scale_mode(scale_a)
    block_b = mx_block_size_for_scale_mode(scale_b)
    if block_a > 0:
        total += b * math.ceil(m * k / block_a)
    if block_b > 0:
        total += b * math.ceil(n * k / block_b)
    return int(math.ceil(total))


# ── device-memory budget ─────────────────────────────────────────────────────

MEMORY_SAFE_FRACTION = 0.70
FALLBACK_DEVICE_MEMORY_GIB = 32
KFD_TOPOLOGY_NODES = Path("/sys/class/kfd/kfd/topology/nodes")
# KFD memory-bank heap types of device-local memory (frame buffer, public and
# private).
_DEVICE_HEAP_TYPES = (1, 2)
_GIB = 1024**3


def _read_kfd_properties(path: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    try:
        text = path.read_text()
    except OSError:
        return out
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                out[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return out


def read_kfd_device_memory_bytes(
    nodes_dir: Path = KFD_TOPOLOGY_NODES,
) -> Optional[int]:
    """Smallest device-local memory over the GPU nodes of the KFD topology, or
    None when sysfs lists none. Only sysfs files are read; no device is
    opened. The minimum is used because a HIP device index cannot be mapped to
    a topology node reliably, and the smallest device is the safe bound for
    all of them."""
    try:
        nodes = sorted(Path(nodes_dir).iterdir())
    except OSError:
        return None
    sizes: List[int] = []
    for node in nodes:
        props = _read_kfd_properties(node / "properties")
        if props.get("simd_count", 0) <= 0:
            continue
        total = 0
        try:
            banks = sorted((node / "mem_banks").iterdir())
        except OSError:
            banks = []
        for bank in banks:
            bank_props = _read_kfd_properties(bank / "properties")
            if bank_props.get("heap_type", -1) in _DEVICE_HEAP_TYPES:
                total += bank_props.get("size_in_bytes", 0)
        if total <= 0:
            total = props.get("local_mem_size", 0)
        if total > 0:
            sizes.append(total)
    return min(sizes) if sizes else None


@dataclass(frozen=True)
class MemoryBudget:
    device_bytes: int
    source: str  # "config", "kfd" or "fallback"

    @property
    def budget_bytes(self) -> int:
        return int(self.device_bytes * MEMORY_SAFE_FRACTION)


def device_memory_budget(
    cfg: Optional[Mapping[str, Any]],
    *,
    nodes_dir: Path = KFD_TOPOLOGY_NODES,
) -> MemoryBudget:
    """Device memory for the footprint filter: `bench.device_memory_gib` when
    set, else the KFD sysfs topology, else FALLBACK_DEVICE_MEMORY_GIB. Set the
    config key to make shape generation independent of the host."""
    gib = ((cfg or {}).get("bench") or {}).get("device_memory_gib")
    if gib is not None:
        if isinstance(gib, bool) or not isinstance(gib, (int, float)):
            raise ValueError(f"bench.device_memory_gib must be a number, got {gib!r}")
        if not math.isfinite(gib) or gib <= 0:
            raise ValueError(f"bench.device_memory_gib must be positive, got {gib!r}")
        return MemoryBudget(int(gib * _GIB), "config")
    kfd = read_kfd_device_memory_bytes(nodes_dir)
    if kfd:
        return MemoryBudget(kfd, "kfd")
    return MemoryBudget(FALLBACK_DEVICE_MEMORY_GIB * _GIB, "fallback")


# ── exclusion rules ──────────────────────────────────────────────────────────

# `<bound>_<axis>`: min_* is an inclusive lower bound, max_* an inclusive upper
# bound; axis `mn` is min(M, N).
EXCLUDE_KEYS = (
    "min_m",
    "max_m",
    "min_n",
    "max_n",
    "min_k",
    "max_k",
    "min_batch",
    "max_batch",
    "min_mn",
    "max_mn",
)
ExcludeRule = Tuple[Tuple[str, int], ...]


def parse_exclude_rules(rules: Any) -> Tuple[ExcludeRule, ...]:
    """Validate `seed.exclude`: a list of non-empty mappings from EXCLUDE_KEYS
    to integers. A shape is excluded when it satisfies every bound of at least
    one rule, so `[{max_mn: 8}, {min_k: 65536}]` drops shapes with
    min(M, N) <= 8 and shapes with K >= 65536. Raises ValueError on anything
    else, since a mistyped rule would otherwise exclude nothing."""
    if rules is None:
        return ()
    if not isinstance(rules, (list, tuple)):
        raise ValueError(f"seed.exclude must be a list of rules, got {rules!r}")
    out: List[ExcludeRule] = []
    for i, rule in enumerate(rules):
        if not isinstance(rule, Mapping) or not rule:
            raise ValueError(f"seed.exclude[{i}] must be a non-empty mapping")
        unknown = sorted(set(map(str, rule)) - set(EXCLUDE_KEYS))
        if unknown:
            raise ValueError(
                f"seed.exclude[{i}]: unknown key(s) {unknown}; "
                f"allowed: {list(EXCLUDE_KEYS)}"
            )
        bounds: List[Tuple[str, int]] = []
        for key in EXCLUDE_KEYS:
            if key not in rule:
                continue
            v = rule[key]
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"seed.exclude[{i}].{key} must be an integer")
            bounds.append((key, int(v)))
        out.append(tuple(bounds))
    return tuple(out)


def is_excluded(m: int, n: int, k: int, b: int, rules: Iterable[ExcludeRule]) -> bool:
    values = {"m": m, "n": n, "k": k, "batch": b, "mn": min(m, n)}
    for rule in rules:
        if all(
            (values[key[4:]] >= v) if key.startswith("min_") else (values[key[4:]] <= v)
            for key, v in rule
        ):
            return True
    return False


# ── benchable-shape filter ───────────────────────────────────────────────────

# Kernels compute per-matrix element offsets in int32, so no 2D operand may
# hold more elements than this.
_INT32_MAX = 2**31 - 1
# Cost cap rather than a memory limit: shapes above it take long to bench
# across every candidate kernel, and the large-square regime they cover is
# the saturated one where a few large tiles always win.
_MAX_FLOPS = 1e13
# Batched shapes are held to tighter working-set, per-dimension and FLOP caps
# than the device-memory limit so that per-candidate bench time stays bounded.
_BATCHED_MAX_BYTES = 16 * _GIB
_BATCHED_MAX_DIM = 8192
_BATCHED_MAX_FLOPS = 1e12
_MIB = 1024**2


@dataclass(frozen=True)
class ShapeFilter:
    memory_budget_bytes: int
    a_type: str = "bf16_r"
    b_type: str = "bf16_r"
    c_type: str = "bf16_r"
    d_type: str = "bf16_r"
    scale_a: int = 0
    scale_b: int = 0
    rotating_bytes: int = 512 * _MIB
    exclude: Tuple[ExcludeRule, ...] = ()

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any],
        *,
        budget: Optional[MemoryBudget] = None,
        nodes_dir: Path = KFD_TOPOLOGY_NODES,
    ) -> "ShapeFilter":
        hl = cfg.get("hipblaslt") or {}
        bench = cfg.get("bench") or {}
        if budget is None:
            budget = device_memory_budget(cfg, nodes_dir=nodes_dir)
        return cls(
            memory_budget_bytes=budget.budget_bytes,
            a_type=str(hl.get("a_type", "bf16_r")),
            b_type=str(hl.get("b_type", hl.get("a_type", "bf16_r"))),
            c_type=str(hl.get("c_type", "bf16_r")),
            d_type=str(hl.get("d_type", hl.get("c_type", "bf16_r"))),
            scale_a=int(hl.get("scaleA", 0) or 0),
            scale_b=int(hl.get("scaleB", 0) or 0),
            rotating_bytes=max(0, int(bench.get("rotating_mb", 512))) * _MIB,
            exclude=parse_exclude_rules((cfg.get("seed") or {}).get("exclude")),
        )

    @property
    def mx_block(self) -> int:
        """Elements per scale when either operand is block scaled, else 0."""
        return max(
            mx_block_size_for_scale_mode(self.scale_a),
            mx_block_size_for_scale_mode(self.scale_b),
        )

    def footprint_bytes(self, m: int, n: int, k: int, b: int) -> int:
        return footprint_bytes(
            m,
            n,
            k,
            b,
            a_type=self.a_type,
            b_type=self.b_type,
            c_type=self.c_type,
            d_type=self.d_type,
            scale_a=self.scale_a,
            scale_b=self.scale_b,
        )

    def is_benchable(self, m: int, n: int, k: int, b: int) -> bool:
        if m <= 0 or n <= 0 or k <= 0 or b <= 0:
            return False
        if max(m * n, m * k, n * k) > _INT32_MAX:
            return False
        flops = 2.0 * m * n * k * b
        if flops > _MAX_FLOPS:
            return False
        working_set = self.footprint_bytes(m, n, k, b)
        if working_set + self.rotating_bytes > self.memory_budget_bytes:
            return False
        if b > 1 and (
            working_set > _BATCHED_MAX_BYTES
            or max(m, n, k) > _BATCHED_MAX_DIM
            or flops > _BATCHED_MAX_FLOPS
        ):
            return False
        return True

    def accepts(self, m: int, n: int, k: int, b: int) -> bool:
        return self.is_benchable(m, n, k, b) and not is_excluded(
            m, n, k, b, self.exclude
        )


# ── per-cell shape synthesis ─────────────────────────────────────────────────


def _log_unif(rng: random.Random, lo: int, hi: int) -> int:
    if lo == hi:
        return lo
    lg = math.log(max(lo, 1)) + rng.random() * (math.log(hi) - math.log(max(lo, 1)))
    return max(lo, min(hi, int(round(math.exp(lg)))))


def _mt_align(v: int, lo: int, hi: int) -> int:
    """Snap down to the largest macro-tile multiple (256 .. 16) that stays in
    [lo, hi]; `v` unchanged when none does."""
    for mult in (256, 128, 64, 32, 16):
        snapped = (v // mult) * mult
        if lo <= snapped <= hi:
            return snapped
    return v


def _log_sub_range(lo: int, hi: int, bin_idx: int, n_bins: int) -> Tuple[int, int]:
    """Bin `bin_idx` of `[lo, hi]` cut into `n_bins` log-equal, inclusive
    sub-ranges."""
    if lo >= hi or n_bins <= 1:
        return lo, hi
    log_lo = math.log(max(lo, 1))
    log_hi = math.log(max(hi, 1))
    step = (log_hi - log_lo) / n_bins
    sub_lo = max(lo, int(math.floor(math.exp(log_lo + bin_idx * step))))
    sub_hi = min(hi, int(math.ceil(math.exp(log_lo + (bin_idx + 1) * step))))
    return sub_lo, max(sub_lo, sub_hi)


def _mx_align_k(K: int, klo: int, khi: int, block: int) -> Optional[int]:
    """Snap `K` onto a multiple of `block` inside [klo, khi], or None when the
    range holds no multiple. A block-scaled GEMM has one scale per `block`
    elements of K and no partial trailing block, so an unaligned K matches no
    kernel at all."""
    if block <= 1:
        return K
    lo = ((klo + block - 1) // block) * block
    hi = (khi // block) * block
    if lo > hi:
        return None
    return max(lo, min(hi, int(round(K / block)) * block))


def _split_evenly(total: int, n_parts: int, rng: random.Random) -> List[int]:
    base, extra = divmod(total, n_parts)
    quotas = [base] * n_parts
    for i in rng.sample(range(n_parts), extra):
        quotas[i] += 1
    return quotas


Box = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]


def _draw_in_box(
    rng: random.Random,
    box: Box,
    quota: int,
    batched: bool,
    shape_filter: ShapeFilter,
    mt_align_fraction: float,
    seen: Set[ShapeTuple],
    out: List[ShapeTuple],
) -> int:
    """Append up to `quota` new distinct accepted shapes from `box` to `out`;
    return how many were added (20 attempts per requested shape at most)."""
    (mlo, mhi), (nlo, nhi), (klo, khi) = box
    mx_block = shape_filter.mx_block
    if mx_block > 1 and _mx_align_k(klo, klo, khi, mx_block) is None:
        return 0
    added = 0
    attempts = 0
    while added < quota and attempts < quota * 20:
        attempts += 1
        M = _log_unif(rng, mlo, mhi)
        # Cap N, then K, so that M*N, M*K and N*K stay within int32 by
        # construction instead of rejecting most samples of huge boxes.
        nhi_eff = min(nhi, _INT32_MAX // max(1, M))
        if nhi_eff < nlo:
            continue
        N = _log_unif(rng, nlo, nhi_eff)
        khi_eff = min(khi, _INT32_MAX // max(1, M, N))
        if khi_eff < klo:
            continue
        K = _log_unif(rng, klo, khi_eff)
        if rng.random() < mt_align_fraction:
            M = _mt_align(M, mlo, mhi)
        if rng.random() < mt_align_fraction:
            N = _mt_align(N, nlo, nhi_eff)
        if rng.random() < mt_align_fraction:
            K = _mt_align(K, klo, khi_eff)
        if mx_block > 1:
            K_mx = _mx_align_k(K, klo, khi_eff, mx_block)
            if K_mx is None:
                continue
            K = K_mx
        if not batched:
            if not shape_filter.accepts(M, N, K, 1):
                continue
            shape = (M, N, K, 1)
        else:
            feasible_b = [b for b in BANY_VALUES if shape_filter.accepts(M, N, K, b)]
            if not feasible_b:
                continue
            # Weighted by log2(b) so the large batch counts that small
            # matrices admit are not drowned out by the many small ones.
            weights = [math.log2(b) for b in feasible_b]
            shape = (M, N, K, rng.choices(feasible_b, weights=weights, k=1)[0])
        if shape in seen:
            continue
        seen.add(shape)
        out.append(shape)
        added += 1
    return added


def synthesize_cell_shapes(
    cell: str,
    n: int,
    rng: random.Random,
    shape_filter: ShapeFilter,
    *,
    axis_bounds: Optional[Dict[str, Tuple[int, int]]] = None,
    n_strata: int = 1,
    mt_align_fraction: float = 0.70,
) -> List[ShapeTuple]:
    """Up to `n` distinct shapes in `cell` that `shape_filter` accepts.

    `cell` is a base grid label `"<Mt>|<Nt>|<Kt>|<Bt>"`; `axis_bounds` narrows
    the M / N / K ranges for a sub-cell. Each axis is sampled log-uniformly,
    and with probability `mt_align_fraction` snapped down to a macro-tile
    multiple. For block-scaled (MX) configs K is always a multiple of the
    block size. Bany shapes draw B from the BANY_VALUES the sampled (M, N, K)
    admits.

    `n_strata > 1` cuts each of M, N and K into that many log-equal ranges
    and spreads the `n` shapes evenly over the resulting sub-boxes. The quota
    of a sub-box that cannot fill it (no accepted shape there, or too few
    distinct ones) is handed to the sub-boxes that can, so the cell reaches
    `n` whenever the box holds enough accepted shapes. Fewer than `n` come
    back only when it does not. The result is shuffled and depends only on
    the arguments and the state of `rng`."""
    if n <= 0:
        return []
    mt, nt, kt, bt = cell.split("|")
    m_lo, m_hi = TIER_RANGES_MN[mt]
    n_lo, n_hi = TIER_RANGES_MN[nt]
    k_lo, k_hi = TIER_RANGES_K[kt]
    if axis_bounds:
        if "M" in axis_bounds:
            m_lo = max(m_lo, axis_bounds["M"][0])
            m_hi = min(m_hi, axis_bounds["M"][1])
        if "N" in axis_bounds:
            n_lo = max(n_lo, axis_bounds["N"][0])
            n_hi = min(n_hi, axis_bounds["N"][1])
        if "K" in axis_bounds:
            k_lo = max(k_lo, axis_bounds["K"][0])
            k_hi = min(k_hi, axis_bounds["K"][1])
    if m_lo > m_hi or n_lo > n_hi or k_lo > k_hi:
        return []
    strata = max(1, int(n_strata))
    boxes: List[Box] = [
        (
            _log_sub_range(m_lo, m_hi, i, strata),
            _log_sub_range(n_lo, n_hi, j, strata),
            _log_sub_range(k_lo, k_hi, kk, strata),
        )
        for i in range(strata)
        for j in range(strata)
        for kk in range(strata)
    ]
    batched = bt != "Bnone"
    seen: Set[ShapeTuple] = set()
    out: List[ShapeTuple] = []
    exhausted = [False] * len(boxes)
    quotas = _split_evenly(n, len(boxes), rng)
    while True:
        for i, quota in enumerate(quotas):
            if quota <= 0 or exhausted[i]:
                continue
            got = _draw_in_box(
                rng,
                boxes[i],
                quota,
                batched,
                shape_filter,
                mt_align_fraction,
                seen,
                out,
            )
            if got < quota:
                exhausted[i] = True
        deficit = n - len(out)
        live = [i for i, done in enumerate(exhausted) if not done]
        if deficit <= 0 or not live:
            break
        quotas = [0] * len(boxes)
        for i, quota in zip(live, _split_evenly(deficit, len(live), rng)):
            quotas[i] = quota
    rng.shuffle(out)
    return out[:n]


# ── re-used history shapes ───────────────────────────────────────────────────

# Written by stage01 into its output directory: the shapes of this round that
# were re-benched from earlier rounds to reach a leaf's minimum shape count.
# They are training data for this round but not held-out data.
REUSED_SHAPES_FILE = "reused_shapes.yaml"


def write_reused_shapes(
    path: Path, shapes: Iterable[ShapeTuple], round_index: int
) -> None:
    rows = sorted({tuple(int(v) for v in s) for s in shapes})
    lines = [f"round_index: {int(round_index)}"]
    if rows:
        lines.append("shapes:")
        lines.extend(f"- [{m}, {n}, {k}, {b}]" for m, n, k, b in rows)
    else:
        lines.append("shapes: []")
    atomic_write_text(Path(path), "\n".join(lines) + "\n")


def read_reused_shapes(path: Path) -> Set[ShapeTuple]:
    """(M, N, K, B) of the re-used shapes recorded by stage01. `path` is the
    file or the stage01 directory holding it; a missing file means none."""
    p = Path(path)
    if p.is_dir():
        p = p / REUSED_SHAPES_FILE
    if not p.exists():
        return set()
    with p.open() as f:
        data = yaml.safe_load(f) or {}
    return {
        (int(s[0]), int(s[1]), int(s[2]), int(s[3])) for s in (data.get("shapes") or [])
    }
