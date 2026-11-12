# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free closed forms and raw-bit policy checks for the uniform-probability probe.

There is intentionally no general attention implementation in this module.
Zero Q forces uniform legal logits, so every expectation follows directly from
legal-key membership and exact rational rounding into the subject dtype.

The structured-logit lane's own copy of the uniform model: the Hq = Hkv = 4
quick cases, their calibration and the primitives its enclosures are derived
against. The uniform lane uses ``_analytical.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from enum import Enum
from functools import cache
from types import MappingProxyType

import numpy as np
from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16

_DTYPES = frozenset({"fp16", "bf16"})
_MASKS = frozenset({"causal", "window"})
_FORMAT = {
    "fp16": (10, 5, 15),
    "bf16": (7, 8, 127),
}
UNIFORM_BASIS_OFFSETS = (0, 64, 128, 192)
UNIFORM_FOCUSED_ROWS = (0, 62, 63, 64, 127, 128, 255)
# KV head h scales its V profile by 2**h, so heads are distinguishable: a kernel
# that reads the wrong V head, uses a wrong head stride, or writes one output head
# everywhere moves every legal coordinate by at least a factor of two. A power of
# two is exact in both dtypes and commutes with every rounding the kernel performs
# (the P cast, FP32 accumulation, normalization, the output cast), so each head's
# correct output is the head-0 output shifted in exponent and the zero-ULP
# argument is unchanged. The cap keeps outputs well inside the FP16 range.
_MAX_KV_HEADS = 8


class UniformMutation(str, Enum):
    """Analyzer-side legal-set and denominator mutations."""

    INCLUDE_FUTURE_KEY = "include_future_key"
    EXCLUDE_LAST_KEY = "exclude_last_key"
    INCLUDE_OLD_WINDOW_KEY = "include_old_window_key"
    EXCLUDE_OLDEST_WINDOW_KEY = "exclude_oldest_window_key"
    DENOMINATOR_PLUS_ONE = "denominator_plus_one"


@dataclass(frozen=True)
class UniformCase:
    """Logical dense-attention shape and semantics for one uniform-probability case."""

    batch: int
    seqlen_q: int
    seqlen_k: int
    nhead_q: int
    nhead_k: int
    head_dim: int
    dtype: str
    mask: str
    window: int = 0

    def __post_init__(self) -> None:
        for name in (
            "batch",
            "seqlen_q",
            "seqlen_k",
            "nhead_q",
            "nhead_k",
            "head_dim",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.nhead_q % self.nhead_k:
            raise ValueError("nhead_q must be a multiple of nhead_k")
        if self.nhead_k > _MAX_KV_HEADS:
            raise ValueError(
                f"nhead_k must be at most {_MAX_KV_HEADS} for distinct head scales"
            )
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
        if self.mask not in _MASKS:
            raise ValueError(f"mask must be one of {sorted(_MASKS)}")
        if self.mask == "window":
            if self.window <= 0:
                raise ValueError("window mask requires window > 0")
        elif self.window:
            raise ValueError("window must be zero unless mask='window'")

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        """Return the runner's logical ``[B, Sq, Hq, D]`` output shape."""
        return (self.batch, self.seqlen_q, self.nhead_q, self.head_dim)

    @property
    def key_shape(self) -> tuple[int, int, int, int]:
        """Return the runner's logical ``[B, Sk, Hkv, D]`` K/V shape."""
        return (self.batch, self.seqlen_k, self.nhead_k, self.head_dim)

    @property
    def head_scale_exponents(self) -> tuple[int, ...]:
        """Per query head, the exponent of its KV head's V scale ``2**h_kv``.

        Query head ``h_q`` reads KV head ``h_q // (nhead_q // nhead_k)``."""
        group = self.nhead_q // self.nhead_k
        return tuple(head // group for head in range(self.nhead_q))

    @property
    def case_id(self) -> str:
        """Return a stable identifier used to close calibration scope."""
        mask = f"window{self.window}" if self.mask == "window" else self.mask
        return (
            f"{self.dtype}-{mask}-b{self.batch}-sq{self.seqlen_q}-"
            f"sk{self.seqlen_k}-hq{self.nhead_q}-hk{self.nhead_k}-d{self.head_dim}"
        )


UNIFORM_CAUSAL_FP16 = UniformCase(1, 256, 256, 4, 4, 64, "fp16", "causal")
UNIFORM_WINDOW_BF16 = UniformCase(1, 256, 256, 4, 4, 64, "bf16", "window", 64)
# The GPU lane's parameter list. A CPU contract pins it to the calibrated case set,
# so a calibrated case cannot silently stop executing.
UNIFORM_QUICK_CASES = (UNIFORM_CAUSAL_FP16, UNIFORM_WINDOW_BF16)


@dataclass(frozen=True)
class UniformSpecFingerprint:
    """Proof-relevant fields of the dispatch-resolved dense subject."""

    path: str
    persistent: bool
    runtime_shape: bool
    head_dim: int
    block_m: int
    block_n: int
    use_exp2_fast: bool
    use_sinks: bool
    emit_lse: bool


@dataclass(frozen=True)
class CalibrationEntry:
    """One ratified, closed-scope zero-ULP policy entry.

    Installing an entry in ``UNIFORM_CALIBRATION`` is the ratification; the
    develop revision and gfx942 evidence behind it belong in the commit that
    edits the table, not in a field nothing reads."""

    ulp_budget: int
    case_ids: tuple[str, ...]
    spec: UniformSpecFingerprint

    def __post_init__(self) -> None:
        if self.ulp_budget != 0:
            raise ValueError(
                "the uniform-probability check currently supports only ulp_budget=0"
            )
        if not self.case_ids or len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("case_ids must be nonempty and unique")

    def mismatch(
        self, case: UniformCase, spec: UniformSpecFingerprint
    ) -> tuple[str, ...]:
        """Name every reason this entry does not authorize ``case`` and ``spec``."""
        problems: list[str] = []
        if case.case_id not in self.case_ids:
            problems.append(f"case {case.case_id} not in calibrated {self.case_ids}")
        for item in fields(UniformSpecFingerprint):
            calibrated = getattr(self.spec, item.name)
            resolved = getattr(spec, item.name)
            if calibrated != resolved:
                problems.append(
                    f"{item.name}: calibrated {calibrated!r}, resolved {resolved!r}"
                )
        return tuple(problems)

    def covers(self, case: UniformCase, spec: UniformSpecFingerprint) -> bool:
        """Return whether this entry authorizes ``case`` and resolved ``spec``."""
        return not self.mismatch(case, spec)


_DENSE_QUICK_SPEC_FP16 = UniformSpecFingerprint(
    path="dense",
    persistent=False,
    runtime_shape=True,
    head_dim=64,
    block_m=128,
    block_n=64,
    use_exp2_fast=True,
    use_sinks=False,
    emit_lse=False,
)

_DENSE_QUICK_SPEC_BF16 = UniformSpecFingerprint(
    path="dense",
    persistent=False,
    runtime_shape=True,
    head_dim=64,
    block_m=64,
    block_n=64,
    use_exp2_fast=True,
    use_sinks=False,
    emit_lse=False,
)


# Ratified on gfx942 against the exact quick cases below. A future shape or
# resolved-spec change is uncalibrated even when candidate name and dtype match.
UNIFORM_CALIBRATION: Mapping[tuple[str, str, str], CalibrationEntry] = MappingProxyType(
    {
        (
            "gfx942",
            "attention_gfx942_dense",
            "fp16",
        ): CalibrationEntry(
            ulp_budget=0,
            case_ids=(UNIFORM_CAUSAL_FP16.case_id,),
            spec=_DENSE_QUICK_SPEC_FP16,
        ),
        (
            "gfx942",
            "attention_gfx942_dense",
            "bf16",
        ): CalibrationEntry(
            ulp_budget=0,
            case_ids=(UNIFORM_WINDOW_BF16.case_id,),
            spec=_DENSE_QUICK_SPEC_BF16,
        ),
    }
)


_RECALIBRATION_REMEDY = (
    "re-run the case on gfx942 with ROCKE_ORACLE_FREE_CALIBRATION_OUT=<dir>, "
    "review the records, and update UNIFORM_CALIBRATION (spec, case_ids) "
    "deliberately"
)


def require_calibration(
    arch: str,
    candidate_name: str,
    case: UniformCase,
    spec: UniformSpecFingerprint,
    *,
    table: Mapping[tuple[str, str, str], CalibrationEntry] = UNIFORM_CALIBRATION,
    detail: str = "",
) -> CalibrationEntry:
    """Return the entry that authorizes this resolved case, or fail closed.

    A missing entry or any scope mismatch is a failure rather than a skip: a
    dispatch change that picks another variant must be re-verified on the
    hardware, not silently dropped from the gate."""
    key = (arch, candidate_name, case.dtype)
    suffix = f"; {detail}" if detail else ""
    entry = table.get(key)
    if entry is None:
        raise AssertionError(
            f"no uniform-probability calibration for {key}, case={case.case_id}; "
            f"{_RECALIBRATION_REMEDY}{suffix}"
        )
    problems = entry.mismatch(case, spec)
    if problems:
        raise AssertionError(
            f"uniform-probability calibration {key} does not cover "
            f"case={case.case_id}: {'; '.join(problems)}; "
            f"{_RECALIBRATION_REMEDY}{suffix}"
        )
    return entry


def _coerce_mutation(mutation: UniformMutation | str | None) -> UniformMutation | None:
    if mutation is None:
        return None
    try:
        return (
            mutation
            if isinstance(mutation, UniformMutation)
            else UniformMutation(mutation)
        )
    except ValueError as error:
        raise ValueError(f"unknown mutation {mutation!r}") from error


def _validate_mutation(case: UniformCase, mutation: UniformMutation | None) -> None:
    if (
        mutation
        in {
            UniformMutation.INCLUDE_FUTURE_KEY,
            UniformMutation.EXCLUDE_LAST_KEY,
        }
        and case.mask != "causal"
    ):
        raise ValueError(f"mutation {mutation.value} requires a causal case")
    if (
        mutation
        in {
            UniformMutation.INCLUDE_OLD_WINDOW_KEY,
            UniformMutation.EXCLUDE_OLDEST_WINDOW_KEY,
        }
        and case.mask != "window"
    ):
        raise ValueError(f"mutation {mutation.value} requires a window case")


def legal_interval(
    case: UniformCase,
    query: int,
    mutation: UniformMutation | str | None = None,
) -> tuple[int, int]:
    """Return inclusive analyzer legal bounds, or ``(0, -1)`` when empty."""
    if not 0 <= query < case.seqlen_q:
        raise IndexError(query)
    selected = _coerce_mutation(mutation)
    _validate_mutation(case, selected)

    upper_shift = 0
    if selected == UniformMutation.INCLUDE_FUTURE_KEY:
        upper_shift = 1
    elif selected == UniformMutation.EXCLUDE_LAST_KEY:
        upper_shift = -1
    upper = min(query + upper_shift, case.seqlen_k - 1)

    lower = 0
    if case.mask == "window":
        window = case.window
        if selected == UniformMutation.INCLUDE_OLD_WINDOW_KEY:
            window += 1
        elif selected == UniformMutation.EXCLUDE_OLDEST_WINDOW_KEY:
            window -= 1
        lower = max(0, query - window + 1)
    return (lower, upper) if lower <= upper else (0, -1)


def legal_keys(
    case: UniformCase,
    query: int,
    mutation: UniformMutation | str | None = None,
) -> tuple[int, ...]:
    """Enumerate the independently derived legal keys for one query."""
    lower, upper = legal_interval(case, query, mutation)
    return tuple(range(lower, upper + 1))


def residue_count(first: int, last: int, residue: int, period: int) -> int:
    """Count integers in ``[first,last]`` having one nonnegative residue."""
    if period <= 0:
        raise ValueError("period must be positive")
    if not 0 <= residue < period:
        raise ValueError("residue must lie in [0, period)")
    if last < first:
        return 0
    return (last - residue) // period - (first - 1 - residue) // period


def _round_ratio_to_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    twice = 2 * remainder
    return quotient + int(
        twice > denominator or (twice == denominator and quotient & 1)
    )


@cache
def target_bits(numerator: int, denominator: int, dtype: str) -> int:
    """Round one nonnegative exact rational to FP16 or BF16 with RNE."""
    if dtype not in _FORMAT:
        raise ValueError(dtype)
    if numerator < 0 or denominator <= 0:
        raise ValueError("target ratio requires numerator >= 0 and denominator > 0")
    if numerator == 0:
        return 0

    fraction_bits, exponent_bits, bias = _FORMAT[dtype]
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < denominator << exponent:
            exponent -= 1
    elif numerator << -exponent < denominator:
        exponent -= 1

    min_exponent = 1 - bias
    max_exponent = (1 << exponent_bits) - 2 - bias
    if exponent < min_exponent:
        shift = fraction_bits - min_exponent
        rounded = _round_ratio_to_even(numerator << shift, denominator)
        if rounded >= 1 << fraction_bits:
            return 1 << fraction_bits
        return rounded

    shift = exponent - fraction_bits
    if shift >= 0:
        rounded = _round_ratio_to_even(numerator, denominator << shift)
    else:
        rounded = _round_ratio_to_even(numerator << -shift, denominator)
    if rounded == 1 << (fraction_bits + 1):
        rounded >>= 1
        exponent += 1
    if exponent > max_exponent:
        return ((1 << exponent_bits) - 1) << fraction_bits
    encoded_exponent = exponent + bias
    return (encoded_exponent << fraction_bits) | (rounded - (1 << fraction_bits))


def ordered_bits(bits: int) -> int:
    """Map binary16 bits to a monotone integer, coalescing signed zero."""
    raw = int(bits) & 0xFFFF
    if raw & 0x7FFF == 0:
        return 0x8000
    if raw & 0x8000:
        return 0x8000 - (raw & 0x7FFF)
    return 0x8000 + raw


def ulp_distance(a_bits: int, b_bits: int) -> int:
    """Return integer target-format ULP distance between two raw values."""
    return abs(ordered_bits(a_bits) - ordered_bits(b_bits))


def _ordered_array(bits: np.ndarray) -> np.ndarray:
    raw = np.asarray(bits, dtype=np.uint16).astype(np.uint32)
    magnitude = raw & np.uint32(0x7FFF)
    return np.where(
        magnitude == 0,
        np.uint32(0x8000),
        np.where(
            raw & np.uint32(0x8000),
            np.uint32(0x8000) - magnitude,
            np.uint32(0x8000) + raw,
        ),
    )


def decode_bits(bits: np.ndarray | int, dtype: str) -> np.ndarray:
    """Decode raw FP16/BF16 bits to FP64 without losing signed-zero input bits."""
    if dtype not in _DTYPES:
        raise ValueError(dtype)
    raw = np.asarray(bits, dtype=np.uint16)
    if dtype == "fp16":
        return raw.copy().view(np.float16).astype(np.float64)
    # The shared codec returns at least 1-D; keep a scalar input scalar.
    return bf16_to_f32(raw).reshape(raw.shape).astype(np.float64)


def encode_bits(values: np.ndarray, dtype: str) -> np.ndarray:
    """Encode FP32 values as raw FP16/BF16 bits, refusing any value that would round.

    Exactness keeps the subject's inputs independent of conversion rounding: the
    expectations assume the kernel reads exactly these values. Each decoded value
    must reproduce its FP32 source bit-for-bit, so signed zero survives and NaN or
    out-of-range inputs are rejected."""
    if dtype not in _DTYPES:
        raise ValueError(dtype)
    source = np.ascontiguousarray(values, dtype=np.float32)
    # NaN and infinity survive the bit round trip below, so reject them first.
    nonfinite = ~np.isfinite(source)
    if np.any(nonfinite):
        first = tuple(int(index) for index in np.argwhere(nonfinite)[0])
        raise ValueError(
            f"{int(np.count_nonzero(nonfinite))} value(s) are not exactly "
            f"representable in {dtype}, first {float(source[first])!r} at {first}"
        )
    if dtype == "fp16":
        with np.errstate(over="ignore"):
            bits = source.astype(np.float16).view(np.uint16)
    else:
        # Round-to-nearest-even, but an accepted value never rounds: the
        # round-trip check below rejects anything that is not exact.
        bits = f32_to_bf16(source).reshape(source.shape)
    round_trip = decode_bits(bits, dtype).astype(np.float32)
    inexact = round_trip.view(np.uint32) != source.view(np.uint32)
    if np.any(inexact):
        first = tuple(int(index) for index in np.argwhere(inexact)[0])
        raise ValueError(
            f"{int(np.count_nonzero(inexact))} value(s) are not exactly "
            f"representable in {dtype}, first {float(source[first])!r} at {first}"
        )
    return np.ascontiguousarray(bits)


def _deterministic_k(case: UniformCase) -> np.ndarray:
    """Construct finite, nonconstant, exactly representable K values."""
    batch = np.arange(case.batch, dtype=np.int64)[:, None, None, None]
    key = np.arange(case.seqlen_k, dtype=np.int64)[None, :, None, None]
    head = np.arange(case.nhead_k, dtype=np.int64)[None, None, :, None]
    dim = np.arange(case.head_dim, dtype=np.int64)[None, None, None, :]
    values = ((11 * batch + 3 * key + 5 * head + 7 * dim) % 17) - 8
    return (values.astype(np.float32) / 8.0).copy()


def _kv_head_scales(case: UniformCase) -> np.ndarray:
    """Return the per-KV-head V scales ``2**h``."""
    return np.ldexp(np.float32(1.0), np.arange(case.nhead_k)).astype(np.float32)


def uniform_inputs(
    case: UniformCase, basis_offset: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build zero-Q, nonconstant-K, shifted-identity-V inputs, V scaled per head."""
    if basis_offset < 0:
        raise ValueError("basis_offset must be non-negative")
    q = np.zeros(case.output_shape, dtype=np.float32)
    k = _deterministic_k(case)
    v = np.zeros(case.key_shape, dtype=np.float32)
    scales = _kv_head_scales(case)
    for dim in range(case.head_dim):
        key = basis_offset + dim
        if key < case.seqlen_k:
            v[:, key, :, dim] = scales
    return q, k, v


def uniform_inputs_folded(
    case: UniformCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build folded-basis V, where key ``k`` maps to dimension ``k mod D``,
    scaled per head like the basis profiles."""
    q = np.zeros(case.output_shape, dtype=np.float32)
    k = _deterministic_k(case)
    v = np.zeros(case.key_shape, dtype=np.float32)
    scales = _kv_head_scales(case)
    for key in range(case.seqlen_k):
        v[:, key, :, key % case.head_dim] = scales
    return q, k, v


@dataclass(frozen=True, eq=False)
class UniformObservation:
    """Named raw-bit subject outputs from basis and folded exposures."""

    exposures: tuple[tuple[str, np.ndarray], ...]

    def __post_init__(self) -> None:
        if not self.exposures:
            raise ValueError("an observation requires at least one exposure")
        normalized: list[tuple[str, np.ndarray]] = []
        names: set[str] = set()
        for name, values in self.exposures:
            if name in names:
                raise ValueError(f"duplicate exposure {name!r}")
            if name != "folded":
                _basis_offset(name)
            raw = np.asarray(values)
            if raw.dtype != np.uint16:
                raise ValueError(f"exposure {name!r} must contain raw uint16 bits")
            copy = np.ascontiguousarray(raw).copy()
            copy.setflags(write=False)
            normalized.append((name, copy))
            names.add(name)
        object.__setattr__(self, "exposures", tuple(normalized))

    @classmethod
    def from_mapping(cls, exposures: Mapping[str, np.ndarray]) -> UniformObservation:
        """Copy a mapping into one immutable observation."""
        return cls(tuple(exposures.items()))

    @property
    def by_name(self) -> Mapping[str, np.ndarray]:
        """Return a read-only name-to-bits view."""
        return MappingProxyType(dict(self.exposures))


def _basis_offset(name: str) -> int:
    prefix = "basis@"
    if not name.startswith(prefix):
        raise ValueError(f"unknown exposure {name!r}")
    text = name[len(prefix) :]
    try:
        offset = int(text)
    except ValueError as error:
        raise ValueError(f"invalid basis exposure {name!r}") from error
    if offset < 0 or str(offset) != text:
        raise ValueError(f"invalid basis exposure {name!r}")
    return offset


def _basis_offsets(observation: UniformObservation) -> tuple[int, ...]:
    return tuple(
        sorted(
            _basis_offset(name) for name, _ in observation.exposures if name != "folded"
        )
    )


def _validate_observation(
    case: UniformCase, observation: UniformObservation, *, require_complete: bool
) -> None:
    for name, values in observation.exposures:
        if values.shape != case.output_shape:
            raise ValueError(
                f"exposure {name!r} shape {values.shape} != {case.output_shape}"
            )
    if not require_complete:
        return
    names = observation.by_name
    if "folded" not in names:
        raise ValueError("a gated observation requires a folded exposure")
    offsets = _basis_offsets(observation)
    exposed = [
        offset + dim
        for offset in offsets
        for dim in range(case.head_dim)
        if offset + dim < case.seqlen_k
    ]
    if exposed != list(range(case.seqlen_k)):
        raise ValueError("basis exposures must expose every key exactly once")


def _coordinate_representation(
    case: UniformCase,
    exposure: str,
    dim: int,
    first: int,
    last: int,
) -> tuple[bool, int]:
    if exposure == "folded":
        has_key = dim < case.seqlen_k
        represented = residue_count(first, last, dim, case.head_dim) if has_key else 0
        return has_key, represented
    key = _basis_offset(exposure) + dim
    has_key = key < case.seqlen_k
    return has_key, int(has_key and first <= key <= last)


def uniform_expected_bits(
    case: UniformCase,
    exposure: str,
    mutation: UniformMutation | str | None = None,
) -> np.ndarray:
    """Construct correctly rounded synthetic output bits for one expectation."""
    selected = _coerce_mutation(mutation)
    _validate_mutation(case, selected)
    output = np.zeros(case.output_shape, dtype=np.uint16)
    for query in range(case.seqlen_q):
        first, last = legal_interval(case, query, selected)
        legal_count = max(0, last - first + 1)
        if legal_count == 0:
            if selected is None:
                raise ValueError(
                    f"correct legal-key rule has an empty row at q={query}"
                )
            continue
        denominator = legal_count + int(
            selected == UniformMutation.DENOMINATOR_PLUS_ONE
        )
        for dim in range(case.head_dim):
            _, represented = _coordinate_representation(
                case, exposure, dim, first, last
            )
            if represented:
                output[:, query, :, dim] = _head_target_bits(
                    case, represented, denominator
                )
    return output


def _head_target_bits(
    case: UniformCase, represented: int, denominator: int
) -> np.ndarray:
    """Per query head, the correctly rounded ``represented * 2**h_kv / N`` bits."""
    return np.array(
        [
            target_bits(represented << exponent, denominator, case.dtype)
            for exponent in case.head_scale_exponents
        ],
        dtype=np.uint16,
    )


def correct_uniform_observation(
    case: UniformCase,
    *,
    basis_offsets: Sequence[int] | None = None,
) -> UniformObservation:
    """Build a complete correctly rounded synthetic observation."""
    offsets = (
        tuple(range(0, case.seqlen_k, case.head_dim))
        if basis_offsets is None
        else tuple(basis_offsets)
    )
    exposures = [
        (f"basis@{offset}", uniform_expected_bits(case, f"basis@{offset}"))
        for offset in offsets
    ]
    exposures.append(("folded", uniform_expected_bits(case, "folded")))
    return UniformObservation(tuple(exposures))


def broadcast_query_head(
    observation: UniformObservation, head: int = 0
) -> UniformObservation:
    """Copy one query head's output into every head of every exposure.

    The observation a kernel with a head-indexing bug would produce (every head
    reading one V head, or one output head written everywhere). The head-specific
    V scales must make the policy reject it."""
    exposures = []
    for name, values in observation.exposures:
        if not 0 <= head < values.shape[2]:
            raise IndexError(head)
        broadcast = np.repeat(values[:, :, head : head + 1, :], values.shape[2], 2)
        exposures.append((name, broadcast))
    return UniformObservation(tuple(exposures))


@dataclass(frozen=True)
class UniformNqStats:
    """One exposure's reduced evidence for a legal count."""

    rows: int
    compared: int
    exact: int
    max_ulp: int
    max_stored_sum_residual: float


@dataclass
class _MutableNqStats:
    rows: int = 0
    compared: int = 0
    exact: int = 0
    max_ulp: int = 0
    max_stored_sum_residual: float = 0.0


@dataclass(frozen=True)
class UniformResult:
    """Decomposed uniform-probability verdict plus calibration diagnostics."""

    violations: tuple[str, ...]
    invalid_count: int
    max_negative: float
    max_masked_abs: float
    total_masked_mass: float
    max_row_masked_mass: float
    max_trailing_abs: float
    max_ulp: int
    worst_ulp_nq: int
    worst_coordinate: tuple[int, int, int, int] | None
    worst_exposure: str | None
    compared_count: int
    exact_count: int
    max_stored_sum_residual: float
    real_row_sum_error: float
    legacy_abs_residual: float
    focused_row_errors: Mapping[int, float]
    max_ulp_within_proven: int
    max_ulp_beyond_proven: int
    per_nq: Mapping[str, Mapping[int, UniformNqStats]]

    @property
    def exact_rate(self) -> float:
        """Return the target-bit exact-hit fraction."""
        return self.exact_count / self.compared_count if self.compared_count else 0.0


def _proven_limit(dtype: str, exposure: str) -> int:
    fraction_bits = _FORMAT[dtype][0]
    limit = 1 << (21 - fraction_bits)
    return limit * 2 // 3 if exposure == "folded" else limit


def _freeze_groups(
    groups: Mapping[str, Mapping[int, _MutableNqStats]],
) -> Mapping[str, Mapping[int, UniformNqStats]]:
    frozen: dict[str, Mapping[int, UniformNqStats]] = {}
    for exposure, by_nq in groups.items():
        frozen[exposure] = MappingProxyType(
            {
                nq: UniformNqStats(
                    rows=item.rows,
                    compared=item.compared,
                    exact=item.exact,
                    max_ulp=item.max_ulp,
                    max_stored_sum_residual=item.max_stored_sum_residual,
                )
                for nq, item in sorted(by_nq.items())
            }
        )
    return MappingProxyType(frozen)


def analyze_uniform_policy(
    case: UniformCase,
    observation: UniformObservation,
    *,
    budget: int,
    mutation: UniformMutation | str | None = None,
    focused_rows: Sequence[int] = UNIFORM_FOCUSED_ROWS,
    require_complete: bool = True,
) -> UniformResult:
    """Apply the decomposed zero-ULP uniform-probability policy to raw output bits."""
    if budget != 0:
        raise ValueError(
            "the uniform-probability check currently supports only budget=0"
        )
    selected = _coerce_mutation(mutation)
    _validate_mutation(case, selected)
    _validate_observation(case, observation, require_complete=require_complete)

    row_shape = (case.batch, case.seqlen_q, case.nhead_q)
    masked_row_mass = np.zeros(row_shape, dtype=np.float64)
    basis_actual = np.zeros(row_shape, dtype=np.float64)
    basis_target = np.zeros(row_shape, dtype=np.float64)
    basis_valid = np.ones(row_shape, dtype=bool)
    basis_represented = np.zeros(case.seqlen_q, dtype=np.int64)
    basis_present = False

    invalid_count = 0
    max_negative = 0.0
    max_masked_abs = 0.0
    total_masked_mass = 0.0
    max_trailing_abs = 0.0
    max_ulp = 0
    worst_ulp_nq = 0
    worst_coordinate: tuple[int, int, int, int] | None = None
    worst_exposure: str | None = None
    compared_count = 0
    exact_count = 0
    max_stored_sum_residual = 0.0
    real_row_sum_error = 0.0
    legacy_abs_residual = 0.0
    max_ulp_within_proven = 0
    max_ulp_beyond_proven = 0
    focus = {query: 0.0 for query in focused_rows if 0 <= query < case.seqlen_q}
    groups: dict[str, dict[int, _MutableNqStats]] = {}

    # Magnitude diagnostics are in probability units: each head's values are
    # divided by its exact power-of-two V scale. Target-bit ULP checks use the
    # raw bits against the scaled targets, so they stay head-aware.
    head_scales = np.ldexp(1.0, np.array(case.head_scale_exponents))
    for exposure, raw in observation.exposures:
        decoded = decode_bits(raw, case.dtype) / head_scales[None, None, :, None]
        fraction_bits, exponent_bits, _ = _FORMAT[case.dtype]
        exponent_mask = ((1 << exponent_bits) - 1) << fraction_bits
        finite = (raw & np.uint16(exponent_mask)) != np.uint16(exponent_mask)
        invalid_count += int(np.count_nonzero(~finite))
        negative = (
            finite & ((raw & np.uint16(0x8000)) != 0) & ((raw & np.uint16(0x7FFF)) != 0)
        )
        if np.any(negative):
            max_negative = max(max_negative, float(np.max(np.abs(decoded[negative]))))

        exposure_groups = groups.setdefault(exposure, {})
        folded_actual = np.zeros(row_shape, dtype=np.float64)
        folded_target = np.zeros(row_shape, dtype=np.float64)
        folded_valid = np.ones(row_shape, dtype=bool)
        folded_represented = np.zeros(case.seqlen_q, dtype=np.int64)

        for query in range(case.seqlen_q):
            first, last = legal_interval(case, query, selected)
            legal_count = max(0, last - first + 1)
            if legal_count == 0:
                if selected is None:
                    raise ValueError(
                        f"correct legal-key rule has an empty row at q={query}"
                    )
                invalid_count += case.batch * case.nhead_q
            denominator = legal_count + int(
                selected == UniformMutation.DENOMINATOR_PLUS_ONE
            )
            group = exposure_groups.setdefault(legal_count, _MutableNqStats())
            group.rows += case.batch * case.nhead_q

            for dim in range(case.head_dim):
                has_key, represented = _coordinate_representation(
                    case, exposure, dim, first, last
                )
                values = decoded[:, query, :, dim]
                value_bits = raw[:, query, :, dim]
                valid = finite[:, query, :, dim]
                coordinate_magnitudes = np.where(valid, np.abs(values), 0.0)

                if not has_key:
                    local = float(np.max(coordinate_magnitudes))
                    max_trailing_abs = max(max_trailing_abs, local)
                    continue
                if represented == 0:
                    local = float(np.max(coordinate_magnitudes))
                    max_masked_abs = max(max_masked_abs, local)
                    total_masked_mass += float(np.sum(coordinate_magnitudes))
                    masked_row_mass[:, query, :] += coordinate_magnitudes
                    continue

                targets = _head_target_bits(case, represented, denominator)
                target_value = decode_bits(targets, case.dtype) / head_scales
                distances = np.abs(
                    _ordered_array(value_bits).astype(np.int64)
                    - _ordered_array(targets).astype(np.int64)[None, :]
                )
                valid_distances = distances[valid]
                compared = int(valid_distances.size)
                exact = int(np.count_nonzero(valid_distances == 0))
                compared_count += compared
                exact_count += exact
                group.compared += compared
                group.exact += exact
                local_ulp = int(np.max(valid_distances)) if compared else 0
                group.max_ulp = max(group.max_ulp, local_ulp)
                limit = _proven_limit(case.dtype, exposure)
                if legal_count < limit:
                    max_ulp_within_proven = max(max_ulp_within_proven, local_ulp)
                else:
                    max_ulp_beyond_proven = max(max_ulp_beyond_proven, local_ulp)
                if local_ulp > max_ulp:
                    max_ulp = local_ulp
                    worst_ulp_nq = legal_count
                    worst_exposure = exposure
                    masked_distances = np.where(valid, distances, -1)
                    batch, head = np.unravel_index(
                        int(np.argmax(masked_distances)), masked_distances.shape
                    )
                    worst_coordinate = (int(batch), query, int(head), dim)

                real_expected = represented / denominator
                if np.any(valid):
                    errors = np.abs(values[valid] - real_expected)
                    local_error = float(np.max(errors))
                    legacy_abs_residual = max(legacy_abs_residual, local_error)
                    if query in focus:
                        focus[query] = max(focus[query], local_error)

                if exposure == "folded":
                    folded_actual[:, query, :] += np.where(valid, values, 0.0)
                    folded_target[:, query, :] += target_value
                    folded_valid[:, query, :] &= valid
                    folded_represented[query] += represented
                else:
                    basis_present = True
                    basis_actual[:, query, :] += np.where(valid, values, 0.0)
                    basis_target[:, query, :] += target_value
                    basis_valid[:, query, :] &= valid
                    basis_represented[query] += represented

            if exposure == "folded" and folded_represented[query] == legal_count:
                valid_rows = folded_valid[:, query, :]
                if np.any(valid_rows):
                    residuals = np.abs(
                        folded_actual[:, query, :] - folded_target[:, query, :]
                    )
                    local = float(np.max(residuals[valid_rows]))
                    max_stored_sum_residual = max(max_stored_sum_residual, local)
                    group.max_stored_sum_residual = max(
                        group.max_stored_sum_residual, local
                    )
                    real_errors = np.abs(folded_actual[:, query, :] - 1.0)
                    real_row_sum_error = max(
                        real_row_sum_error, float(np.max(real_errors[valid_rows]))
                    )

    if basis_present:
        basis_groups = groups.setdefault("basis_union", {})
        for query in range(case.seqlen_q):
            first, last = legal_interval(case, query, selected)
            legal_count = max(0, last - first + 1)
            group = basis_groups.setdefault(legal_count, _MutableNqStats())
            group.rows += case.batch * case.nhead_q
            if basis_represented[query] != legal_count:
                continue
            valid_rows = basis_valid[:, query, :]
            if not np.any(valid_rows):
                continue
            residuals = np.abs(basis_actual[:, query, :] - basis_target[:, query, :])
            local = float(np.max(residuals[valid_rows]))
            max_stored_sum_residual = max(max_stored_sum_residual, local)
            group.max_stored_sum_residual = max(group.max_stored_sum_residual, local)
            real_errors = np.abs(basis_actual[:, query, :] - 1.0)
            real_row_sum_error = max(
                real_row_sum_error, float(np.max(real_errors[valid_rows]))
            )

    max_row_masked_mass = float(np.max(masked_row_mass))
    violations: list[str] = []
    if invalid_count:
        violations.append(f"invalid={invalid_count}")
    if max_negative != 0.0:
        violations.append(f"negative_max={max_negative:.9g}")
    if max_masked_abs != 0.0:
        violations.append(f"masked_max={max_masked_abs:.9g}")
    if total_masked_mass != 0.0:
        violations.append(f"masked_mass={total_masked_mass:.9g}")
    if max_trailing_abs != 0.0:
        violations.append(f"trailing_max={max_trailing_abs:.9g}")
    if max_ulp > budget:
        violations.append(f"target_ulp={max_ulp}>{budget} at N_q={worst_ulp_nq}")
    if max_stored_sum_residual != 0.0:
        violations.append(f"stored_row_sum={max_stored_sum_residual:.9g}")
    if compared_count == 0:
        violations.append("no_legal_coordinate_observed")

    return UniformResult(
        violations=tuple(violations),
        invalid_count=invalid_count,
        max_negative=max_negative,
        max_masked_abs=max_masked_abs,
        total_masked_mass=total_masked_mass,
        max_row_masked_mass=max_row_masked_mass,
        max_trailing_abs=max_trailing_abs,
        max_ulp=max_ulp,
        worst_ulp_nq=worst_ulp_nq,
        worst_coordinate=worst_coordinate,
        worst_exposure=worst_exposure,
        compared_count=compared_count,
        exact_count=exact_count,
        max_stored_sum_residual=max_stored_sum_residual,
        real_row_sum_error=real_row_sum_error,
        legacy_abs_residual=legacy_abs_residual,
        focused_row_errors=MappingProxyType(focus),
        max_ulp_within_proven=max_ulp_within_proven,
        max_ulp_beyond_proven=max_ulp_beyond_proven,
        per_nq=_freeze_groups(groups),
    )


def format_uniform_result(result: UniformResult) -> str:
    """Render the policy verdict and worst target-bit diagnostics."""
    coordinate = (
        "n/a"
        if result.worst_coordinate is None
        else ":".join(str(value) for value in result.worst_coordinate)
    )
    violations = ", ".join(result.violations) if result.violations else "none"
    return (
        f"violations={violations}; max_ulp={result.max_ulp} "
        f"at {result.worst_exposure or 'n/a'} b:q:h:d={coordinate} "
        f"N_q={result.worst_ulp_nq}; exact={result.exact_count}/"
        f"{result.compared_count} ({result.exact_rate:.6f}); "
        f"stored_sum={result.max_stored_sum_residual:.9g}; "
        f"legacy_abs={result.legacy_abs_residual:.9g}"
    )
