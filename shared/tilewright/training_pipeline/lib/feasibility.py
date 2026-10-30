# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Python port of the engine's candidate gates: the LDS-capacity gate and the
kernel-feasibility rule applied before scoring.

Must stay identical to the engine. Non-temporal availability is per operand
and derived from the caller's whole kernel pool (the library), never from the
subset benched for one GEMM: a candidate set without any `cache_hints == 4`
variant of an operand must not require one. Hints 1..3 are temporal hints and
do not count as availability.

The LDS fit always applies. The leading-free-dimension validity rule reads
the global-read vector widths, which MLREC_v3 models do not read: v2 models
apply it, v3 models leave it to the integration's kernel predicates. The
small-batched one-tile, Dot2 and non-temporal rules are soft: MLREC_v2 models
apply the first two and switch the third, MLREC_v3 models switch each by
their serving flags (`soft_rules`).

An MLREC_v3 model may refine the non-temporal rule family-wise: within
`in_family_region`, a kernel family (the pool's kernels with equal mt and mi,
`family_key`) whose every member the rule rejects keeps one member, its
hint-free member if any, else its lowest (cache_hints_a, cache_hints_b). Like
non-temporal availability, a family is the caller's whole pool
(`non_temporal_families`).

An MLREC_v3 model served without the non-temporal rule may declare an
alignment veto (`alignment_vetoed`), a serving rule of its rankings alone: it
is never one of `passes_gates`' rules, which the planner applies, and
stage06 adds it to the model file it stages (`deployed_alignment_veto`). It
counts the bits of an operand at the non-temporal rule's widths, which are
the element widths of the data types that rule lists by name
(`ALIGNMENT_VETO_DTYPES`); a configuration whose K-contiguous operand has
another data type cannot set it.

Kernel guards (lib/correctness.py `Guard`) are no rule of the engine's: an
integration serving a model trained with them leaves a guarded kernel out of
a problem its guards deny, before the engine ranks and in the analytical
ranking that fills the remaining slots. `guard_bounds` and `guard_denies`
mirror TensileLite's TilewrightRanker (its GuardBounds and guardDenies), so
that picks derived offline leave such a kernel out where it would.

Dtypes are catalog names (`lib.features.normalize_dtype_name`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .features import dtype_from_hipblaslt, normalize_dtype_name

_RULE_BITS = {
    "float": 32,
    "xfloat32": 32,
    "half": 16,
    "bfloat16": 16,
    "float8": 8,
    "float8_fnuz": 8,
    "bfloat8": 8,
    "bfloat8_fnuz": 8,
}
_RULE_BITS_DEFAULT = 16
# The data types whose element width is the non-temporal rule's width for
# them: the alignment veto counts whole lines right only for these.
ALIGNMENT_VETO_DTYPES = tuple(sorted(_RULE_BITS))

_LDS_BITS = {
    "float": 32,
    "double": 64,
    "complexfloat": 64,
    "complexdouble": 128,
    "half": 16,
    "int8x4": 32,
    "int32": 32,
    "bfloat16": 16,
    "int8": 8,
    "int4": 4,
    "int64": 64,
    "xfloat32": 32,
    "float8_fnuz": 8,
    "bfloat8_fnuz": 8,
    "float8bfloat8_fnuz": 8,
    "bfloat8float8_fnuz": 8,
    "float8": 8,
    "bfloat8": 8,
    "float8bfloat8": 8,
    "bfloat8float8": 8,
    "float6": 6,
    "bfloat6": 6,
    "float4": 4,
}

NON_TEMPORAL = 4


# The free sizes a kernel guard reads, by index: M (of A) and N (of B).
GUARD_FREE_SIZES: Tuple[str, ...] = ("M", "N")


def guard_bounds(guards: Iterable[Any]) -> Tuple[int, int]:
    """(least M, least N) that the guards of one kernel admit, 0 for a free
    size none of them reads: per free size the largest minimum, as
    TilewrightRanker merges a kernel's guards. A guard is an object with
    `free` and `minimum` or a mapping of them."""
    bounds = [0, 0]
    for g in guards or ():
        free = g.get("free") if isinstance(g, Mapping) else getattr(g, "free")
        minimum = g.get("minimum") if isinstance(g, Mapping) else getattr(g, "minimum")
        i = GUARD_FREE_SIZES.index(str(free))
        bounds[i] = max(bounds[i], int(minimum))
    return bounds[0], bounds[1]


def guard_denies(m: int, n: int, bounds: Tuple[int, int]) -> bool:
    """Whether guards of bounds `bounds` (`guard_bounds`) deny a problem
    whose free sizes are `m` (of A) and `n` (of B): TilewrightRanker's
    guardDenies."""
    return int(m) < int(bounds[0]) or int(n) < int(bounds[1])


def rule_bits(dtype: Any) -> int:
    """Element bits the feasibility rule uses (16 for unlisted dtypes)."""
    return _RULE_BITS.get(normalize_dtype_name(dtype), _RULE_BITS_DEFAULT)


def lds_bits(dtype: Any) -> int:
    """Element bits the LDS gate uses; -1 for a dtype it does not know."""
    return _LDS_BITS.get(normalize_dtype_name(dtype), -1)


def _is_t(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().upper() == "T"


def _field(c: Any, name: str) -> int:
    v = c.get(name, 0) if isinstance(c, Mapping) else getattr(c, name, 0)
    return int(v or 0)


def non_temporal_availability(pool: Iterable[Any]) -> Tuple[bool, bool]:
    """(A, B) availability of non-temporal variants in a kernel pool of
    mappings or objects carrying `cache_hints_a` / `cache_hints_b`."""
    nt_a = nt_b = False
    for c in pool:
        nt_a = nt_a or _field(c, "cache_hints_a") == NON_TEMPORAL
        nt_b = nt_b or _field(c, "cache_hints_b") == NON_TEMPORAL
    return nt_a, nt_b


FAMILY_FIELDS = ("mt_m", "mt_n", "mt_k", "mi_m", "mi_n", "mi_k")


def in_family_region(m: int, n: int, k: int, batch: int) -> bool:
    """Whether the family-wise non-temporal rule applies to a problem:
    batched, K <= 512, and thin along exactly one of M and N (M <= 32 < N,
    or 32 < M <= 128 with N <= 32)."""
    m, n = int(m), int(n)
    if int(batch) <= 1 or int(k) > 512:
        return False
    return (m <= 32 < n) or (32 < m <= 128 and n <= 32)


def family_key(config: Any) -> Tuple[int, ...]:
    """A kernel's family: (mt_m, mt_n, mt_k, mi_m, mi_n, mi_k)."""
    return tuple(_field(config, f) for f in FAMILY_FIELDS)


@dataclass(frozen=True)
class NtFamily:
    """What the family-wise non-temporal rule reads of one family of a pool:
    whether a member has no cache hint, cache_hints_a == 4 or
    cache_hints_b == 4, and the hints of the member it keeps."""

    plain: bool
    nt_a: bool
    nt_b: bool
    kept: Tuple[int, int]


def non_temporal_families(pool: Iterable[Any]) -> Dict[Tuple[int, ...], NtFamily]:
    """`NtFamily` of every family of a kernel pool (mappings or objects
    carrying the FAMILY_FIELDS and cache hints)."""
    members: Dict[Tuple[int, ...], list] = {}
    for c in pool:
        hints = (_field(c, "cache_hints_a"), _field(c, "cache_hints_b"))
        members.setdefault(family_key(c), []).append(hints)
    out: Dict[Tuple[int, ...], NtFamily] = {}
    for key, hints in members.items():
        plain = (0, 0) in hints
        out[key] = NtFamily(
            plain=plain,
            nt_a=any(a == NON_TEMPORAL for a, _ in hints),
            nt_b=any(b == NON_TEMPORAL for _, b in hints),
            kept=(0, 0) if plain else min(hints),
        )
    return out


def family_keeps(
    family: NtFamily,
    aligned: bool,
    by_b: bool,
    by_a: bool,
    nt_a_available: bool,
    nt_b_available: bool,
) -> bool:
    """Whether the non-temporal rule passes no member of `family`, given the
    tests its members share: K and DepthU aligned, the B and A branches."""
    if aligned and by_b:
        return nt_b_available and not family.nt_b
    if aligned and by_a:
        return nt_a_available and not family.nt_a
    return not family.plain


def lds_fits(
    mt_m: int, mt_n: int, mt_k: int, a_dtype: Any, b_dtype: Any, lds_bytes: int
) -> bool:
    """A and B macro-tile footprint fits the LDS budget."""
    a_bits, b_bits = lds_bits(a_dtype), lds_bits(b_dtype)
    if a_bits <= 0 or b_bits <= 0:
        return False
    usage = (int(mt_m) * int(mt_k)) * (a_bits / 8.0) + (int(mt_n) * int(mt_k)) * (
        b_bits / 8.0
    )
    return usage <= float(lds_bytes)


def is_kernel_feasible(
    m: int,
    n: int,
    k: int,
    batch: int,
    a_dtype: Any,
    b_dtype: Any,
    trans_a: Any,
    trans_b: Any,
    mt_m: int,
    mt_n: int,
    mt_k: int,
    mi_m: int,
    mi_n: int,
    mi_k: int,
    cache_hints_a: int,
    cache_hints_b: int,
    *,
    grvw_a: int,
    grvw_b: int,
    nt_a_available: bool,
    nt_b_available: bool,
    non_temporal_rule: bool = True,
    small_batched_rule: bool = True,
    dot2_rule: bool = True,
    leading_free_rule: bool = True,
    non_temporal_family_rule: bool = False,
    nt_family: Optional[NtFamily] = None,
) -> bool:
    """The engine's feasibility rules beyond the LDS fit. With
    `non_temporal_family_rule` (and `non_temporal_rule`), `nt_family` is the
    kernel's family in the pool (`non_temporal_families`); ValueError when a
    problem of the rule's region needs it and it is None."""
    M, N, K, B = int(m), int(n), int(k), int(batch)
    MT_M, MT_N, MT_K = int(mt_m), int(mt_n), int(mt_k)
    MI_M, MI_N, MI_K = int(mi_m), int(mi_n), int(mi_k)
    cha, chb = int(cache_hints_a), int(cache_hints_b)
    a_trans, b_trans = _is_t(trans_a), _is_t(trans_b)
    a_bits, b_bits = rule_bits(a_dtype), rule_bits(b_dtype)

    small_batched = M <= 256 and N <= 256 and K < 1024 and B != 1
    if small_batched_rule and small_batched and (MT_M < M or MT_N < N):
        return False
    if dot2_rule and MI_M == 1 and MI_N == 1 and MI_K == 64 and M > 2:
        return False
    # A contiguous free dimension holds one global-read vector (TensileLite's
    # LeadingFree0/1SizesGreaterOrEqual; not emitted for Int8x4).
    if leading_free_rule and normalize_dtype_name(a_dtype) != "int8x4":
        if not a_trans and M < int(grvw_a):
            return False
        if b_trans and N < int(grvw_b):
            return False
    if not non_temporal_rule:
        return True
    aligned = (K * a_bits) % 1024 == 0 and (MT_K * a_bits) % 1024 == 0
    by_b = M <= MT_M * 2 and not b_trans and (N * b_bits) // max(M * a_bits, 1) > 5
    by_a = (
        not by_b
        and N <= MT_N * 2
        and a_trans
        and (M * a_bits) // max(N * b_bits, 1) > 5
    )
    if aligned and by_b:
        passes = not (nt_b_available and chb != NON_TEMPORAL)
    elif aligned and by_a:
        passes = not (nt_a_available and cha != NON_TEMPORAL)
    else:
        passes = not (cha or chb)
    if passes or not non_temporal_family_rule or not in_family_region(M, N, K, B):
        return passes
    if nt_family is None:
        raise ValueError("the family-wise non-temporal rule needs the kernel's family")
    return (cha, chb) == nt_family.kept and family_keeps(
        nt_family, aligned, by_b, by_a, nt_a_available, nt_b_available
    )


def passes_gates(
    problem: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    lds_bytes: int,
    nt_a_available: bool,
    nt_b_available: bool,
    non_temporal_rule: bool = True,
    small_batched_rule: bool = True,
    dot2_rule: bool = True,
    leading_free_rule: bool = True,
    non_temporal_family_rule: bool = False,
    nt_families: Optional[Mapping[Tuple[int, ...], NtFamily]] = None,
) -> bool:
    """LDS gate then feasibility rule, on `lib.features` problem / config
    kwargs dicts (`problem_kwargs_from_row` / `config_kwargs_from_row`).
    `nt_families`: `non_temporal_families` of the pool, which the
    family-wise rule needs."""
    if not lds_fits(
        config["mt_m"],
        config["mt_n"],
        config["mt_k"],
        problem["a_dtype"],
        problem["b_dtype"],
        lds_bytes,
    ):
        return False
    return is_kernel_feasible(
        problem["m"],
        problem["n"],
        problem["k"],
        problem["batch"],
        problem["a_dtype"],
        problem["b_dtype"],
        problem["a_transpose"],
        problem["b_transpose"],
        config["mt_m"],
        config["mt_n"],
        config["mt_k"],
        config["mi_m"],
        config["mi_n"],
        config["mi_k"],
        config["cache_hints_a"],
        config["cache_hints_b"],
        grvw_a=config["grvw_a"],
        grvw_b=config["grvw_b"],
        nt_a_available=nt_a_available,
        nt_b_available=nt_b_available,
        non_temporal_rule=non_temporal_rule,
        small_batched_rule=small_batched_rule,
        dot2_rule=dot2_rule,
        leading_free_rule=leading_free_rule,
        non_temporal_family_rule=non_temporal_family_rule,
        nt_family=None if nt_families is None else nt_families.get(family_key(config)),
    )


def soft_rules(serving: Mapping[str, Any], version: int) -> Dict[str, bool]:
    """`passes_gates` rule keywords of a model's serving options: v2 applies
    small-batched and Dot2 and the non-temporal rule unless switched off; v3
    applies only those its serving options switch on, and not the
    leading-free rule. The family-wise rule's keyword is there only when the
    serving options switch it on, with the non-temporal rule."""
    if version == 3:
        rules = {
            "leading_free_rule": False,
            "small_batched_rule": bool(serving.get("small_batched_rule", False)),
            "dot2_rule": bool(serving.get("dot2_rule", False)),
            "non_temporal_rule": bool(serving.get("non_temporal_rule", False)),
        }
        if rules["non_temporal_rule"] and serving.get("non_temporal_family_rule"):
            rules["non_temporal_family_rule"] = True
        return rules
    return {
        "small_batched_rule": True,
        "dot2_rule": True,
        "non_temporal_rule": bool(serving.get("non_temporal_rule", True)),
    }


def alignment_veto_operand(
    problem: Mapping[str, Any], veto: Optional[Mapping[str, Any]]
) -> Optional[str]:
    """The operand, "a" or "b", whose non-temporal cache hint the alignment
    veto `veto` (a model's serving `alignment_veto`: `k_min` and
    `align_bits`; None for a model without one) rejects for `problem`, or
    None where it does not apply. It applies to a layout with exactly one
    K-contiguous operand (A transposed with B transposed: A; neither
    transposed: B) when K exceeds `k_min` and K elements of that operand, at
    the widths of the non-temporal rule (`rule_bits`), are not a whole number
    of `align_bits`-bit lines; as the engine's Feasibility."""
    if not veto:
        return None
    k = int(problem["k"])
    if k <= int(veto["k_min"]):
        return None
    a = _is_t(problem["a_transpose"])
    b = not _is_t(problem["b_transpose"])
    if a == b:
        return None
    bits = rule_bits(problem["a_dtype"] if a else problem["b_dtype"])
    if (k * bits) % int(veto["align_bits"]) == 0:
        return None
    return "a" if a else "b"


def alignment_vetoed(
    problem: Mapping[str, Any], config: Any, veto: Optional[Mapping[str, Any]]
) -> bool:
    """Whether the alignment veto `veto` rejects `config` (a mapping or object
    carrying cache_hints_a / cache_hints_b) for `problem`: its cache hint on
    the operand of `alignment_veto_operand` is the non-temporal one."""
    op = alignment_veto_operand(problem, veto)
    return op is not None and _field(config, f"cache_hints_{op}") == NON_TEMPORAL


def alignment_veto_served(
    problem: Mapping[str, Any],
    ranked: Sequence[Any],
    veto: Optional[Mapping[str, Any]],
) -> List[int]:
    """The positions of `ranked` (configs best first, as a model without the
    veto scores them) that remain for `problem` once the alignment veto
    `veto` removes the configs it rejects, in order: those it does not
    reject, or all of them when it rejects every one (the veto then does not
    apply). The first is the pick.

    This is what a model with the veto serves where its ranking has no
    whitelist tier, trained-key gate or cascade that the veto changes: under
    the scorer's serving, or where the tier keeps a config the veto does not
    reject and no cascade cuts the ranking short. The engine fails a
    rejected config in the rules its tiers are built with, so a whitelist
    tier or trained-key gate whose every feasible key the veto rejects falls
    back to the other feasible (or untrained) keys, and a cascade keeps the
    best of the keys left; rank the model file with the veto in the engine to
    read what it serves there."""
    kept = [i for i, c in enumerate(ranked) if not alignment_vetoed(problem, c, veto)]
    return kept if kept else list(range(len(ranked)))


def alignment_veto_bits(veto: Mapping[str, Any]) -> Dict[str, int]:
    """`deploy.alignment_veto` ({k_min, align_bytes}) as a model's serving
    `alignment_veto` ({k_min, align_bits}). ValueError unless k_min is a
    whole number from 1 to 2**32 - 1 and align_bytes a power of two whose
    bits fit in 32 bits."""
    k_min, align = veto.get("k_min"), veto.get("align_bytes")
    for name, value in (("k_min", k_min), ("align_bytes", align)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"deploy.alignment_veto.{name} must be a whole number, got {value!r}"
            )
    if not 1 <= k_min < 2**32:
        raise ValueError(
            f"deploy.alignment_veto.k_min {k_min} is not from 1 to 2**32 - 1"
        )
    if align < 1 or align & (align - 1) or 8 * align >= 2**32:
        raise ValueError(
            f"deploy.alignment_veto.align_bytes {align} is not a power of two below 2**29"
        )
    return {"k_min": int(k_min), "align_bits": 8 * int(align)}


def deployed_alignment_veto(
    cfg: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, int]]:
    """The alignment veto stage06 packs into every MLREC_v3 model it stages
    under config `cfg` (`deploy.alignment_veto`, absent or null for none), as
    a serving `alignment_veto`; ValueError for an invalid one
    (`alignment_veto_bits`) and for a config whose K-contiguous operand
    (`hipblaslt.a_type` with transA T and transB T, `hipblaslt.b_type` with
    both N) has a data type outside `ALIGNMENT_VETO_DTYPES`, whose lines the
    veto would count at another width than its elements'."""
    veto = ((cfg or {}).get("deploy") or {}).get("alignment_veto")
    if veto is None:
        return None
    if not isinstance(veto, Mapping):
        raise ValueError(f"deploy.alignment_veto must be a mapping, got {veto!r}")
    out = alignment_veto_bits(veto)
    h = (cfg or {}).get("hipblaslt") or {}
    a = _is_t(h.get("transA", "N"))
    b = not _is_t(h.get("transB", "T"))
    if a != b:
        side = "a" if a else "b"
        name = h.get(f"{side}_type")
        if name is not None:
            try:
                dtype = dtype_from_hipblaslt(name)
            except ValueError:
                dtype = normalize_dtype_name(name)
            if dtype not in _RULE_BITS:
                raise ValueError(
                    f"deploy.alignment_veto: hipblaslt.{side}_type {name} is the "
                    f"K-contiguous operand's data type, whose element width the "
                    f"veto does not count (it counts "
                    f"{', '.join(ALIGNMENT_VETO_DTYPES)} only)"
                )
    return out
