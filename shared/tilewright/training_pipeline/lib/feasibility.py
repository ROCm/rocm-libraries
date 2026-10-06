# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Python port of the engine's candidate gates: the LDS-capacity gate and the
kernel-feasibility rule applied before scoring.

Must stay identical to the engine. Non-temporal availability is per operand
and derived from the caller's whole kernel pool (the library), never from the
subset benched for one GEMM: a candidate set without any `cache_hints == 4`
variant of an operand must not require one. Hints 1..3 are temporal hints and
do not count as availability.

Dtypes are catalog names (`lib.features.normalize_dtype_name`).
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Tuple

from .features import normalize_dtype_name

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


def non_temporal_availability(pool: Iterable[Any]) -> Tuple[bool, bool]:
    """(A, B) availability of non-temporal variants in a kernel pool of
    mappings or objects carrying `cache_hints_a` / `cache_hints_b`."""
    nt_a = nt_b = False
    for c in pool:
        if isinstance(c, Mapping):
            cha, chb = c.get("cache_hints_a", 0), c.get("cache_hints_b", 0)
        else:
            cha, chb = getattr(c, "cache_hints_a", 0), getattr(c, "cache_hints_b", 0)
        nt_a = nt_a or int(cha or 0) == NON_TEMPORAL
        nt_b = nt_b or int(chb or 0) == NON_TEMPORAL
    return nt_a, nt_b


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
    nt_a_available: bool,
    nt_b_available: bool,
) -> bool:
    M, N, K, B = int(m), int(n), int(k), int(batch)
    MT_M, MT_N, MT_K = int(mt_m), int(mt_n), int(mt_k)
    MI_M, MI_N, MI_K = int(mi_m), int(mi_n), int(mi_k)
    cha, chb = int(cache_hints_a), int(cache_hints_b)
    a_trans, b_trans = _is_t(trans_a), _is_t(trans_b)
    a_bits, b_bits = rule_bits(a_dtype), rule_bits(b_dtype)

    # Small batched problems must fit in one tile.
    if M <= 256 and N <= 256 and K < 1024 and B != 1 and (MT_M < M or MT_N < N):
        return False
    # Dot2 kernels are only correct for M < 3.
    if MI_M == 1 and MI_N == 1 and MI_K == 64 and M > 2:
        return False
    if (K * a_bits) % 1024 == 0 and (MT_K * a_bits) % 1024 == 0:
        if M <= MT_M * 2 and not b_trans and (N * b_bits) // max(M * a_bits, 1) > 5:
            if nt_b_available and chb != NON_TEMPORAL:
                return False
        elif N <= MT_N * 2 and a_trans and (M * a_bits) // max(N * b_bits, 1) > 5:
            if nt_a_available and cha != NON_TEMPORAL:
                return False
        elif cha or chb:
            return False
    elif cha or chb:
        return False
    return True


def passes_gates(
    problem: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    lds_bytes: int,
    nt_a_available: bool,
    nt_b_available: bool,
) -> bool:
    """LDS gate then feasibility rule, on `lib.features` problem / config
    kwargs dicts (`problem_kwargs_from_row` / `config_kwargs_from_row`)."""
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
        nt_a_available=nt_a_available,
        nt_b_available=nt_b_available,
    )
