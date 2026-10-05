# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Typed MMA traits loaded from the generated ``mma_traits.json`` table.

``mma_traits.json`` is the committed traits table -- values are never hand-typed into code.
This module gives that raw table a typed, validated, descriptive face:

- :class:`MmaTraits` -- one frozen, validated record per MMA intrinsic. The table's column
  codes (ABK/AKN/AR/BKN/BR/CM/CMN) become descriptive fields; the provenance is preserved
  in the JSON ``_meta.column_glossary``.
- :class:`MmaTraitsCatalog` -- the loaded table with lookup/selection.
- :func:`load_mma_traits` -- read + validate the JSON into a catalog.

Block-hiding intrinsics (dims/params carrying ``X``/``Y`` markers, B>1) are **reserved**:
they load into the catalog as reserved entries so a lookup fails fast with
``NotImplementedError`` rather than silently vanishing. M1 uses B=1 dense rows only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = ["MmaTraits", "MmaTraitsCatalog", "load_mma_traits", "DEFAULT_TRAITS_PATH"]

DEFAULT_TRAITS_PATH = Path(__file__).resolve().parent / "data" / "mma_traits.json"

_VALID_FAMILIES = frozenset({"dense", "sparse", "scaled"})
_VALID_WAVE_SIZES = frozenset({32, 64})

def _require_clean_int(field_name: str, raw_value: str, op_id: str) -> int:
    """Parse a non-negative integer SOT cell, failing fast on markers/garbage.

    Block-hiding rows carry ``X``/``Y`` markers (e.g. ``"16X"``); those are a
    recognized-but-unsupported case (``NotImplementedError``). Anything else that is
    not a clean non-negative integer is malformed SOT data (``ValueError``).
    """
    if raw_value.isdigit():
        return int(raw_value)
    if any(marker in raw_value for marker in ("X", "Y")):
        raise NotImplementedError(
            f"block-hiding intrinsic not supported yet -- op_id={op_id!r}, "
            f"{field_name}={raw_value!r} carries a block marker"
        )
    raise ValueError(
        f"malformed SOT value -- op_id={op_id!r}, {field_name}={raw_value!r}, "
        f"expected a non-negative integer"
    )

@dataclass(frozen=True)
class MmaTraits:
    """One MMA intrinsic's traits, typed and validated (fields from the SOT).

    Descriptive field <- SOT column code (CK trait name):
    ``k_ab_per_lane`` <- ABK (kABKPerLane); ``a_k_num_access`` <- AKN (kAKNumAccess);
    ``a_repeat`` <- AR (kARepeat); ``b_k_num_access`` <- BKN (kBKNumAccess);
    ``b_repeat`` <- BR (kBRepeat); ``c_m_per_lane`` <- CM (kCMPerLane);
    ``c_m_num_access`` <- CMN (kCMNumAccess).
    """

    op_id: str
    llvm_builtin: str
    family: str
    wave_size: int
    input_dtype: str
    output_dtype: str
    # Fragment dims (B=1 for the dense no-block rows M1 uses).
    m: int
    n: int
    k: int
    b: int
    r: int
    s: int
    # Layout params (CK trait meanings; see class docstring).
    k_ab_per_lane: int
    a_k_num_access: int
    a_repeat: int
    b_k_num_access: int
    b_repeat: int
    c_m_per_lane: int
    c_m_num_access: int
    # Compact unmerge-merge descriptors, verbatim from the SOT.
    a_layout: str
    b_layout: str
    c_d_layout: str
    supported_targets: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.family not in _VALID_FAMILIES:
            raise ValueError(
                f"unknown MMA family -- op_id={self.op_id!r}, family={self.family!r}, "
                f"expected one of {sorted(_VALID_FAMILIES)}"
            )
        if self.wave_size not in _VALID_WAVE_SIZES:
            raise ValueError(
                f"invalid wave_size -- op_id={self.op_id!r}, wave_size={self.wave_size}, "
                f"expected one of {sorted(_VALID_WAVE_SIZES)}"
            )
        for dim_name in ("m", "n", "k", "b", "r", "s"):
            value = getattr(self, dim_name)
            if value <= 0:
                raise ValueError(
                    f"non-positive fragment dim -- op_id={self.op_id!r}, "
                    f"{dim_name}={value}, expected > 0"
                )
        # NB: an empty supported_targets is a legitimate table state (an intrinsic present
        # in the matrix but not marked for any supported gfx target); such rows simply
        # never match a selection. It is not a data error.

    def supports(self, target: str) -> bool:
        """True if this intrinsic is supported on ``target`` (e.g. ``gfx90a``)."""
        return target in self.supported_targets

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> MmaTraits:
        """Build a validated ``MmaTraits`` from one ``mma_traits.json`` operation record.

        Raises ``NotImplementedError`` for block-hiding rows (reserved) and ``ValueError``
        for malformed SOT data.
        """
        op_id = str(record["op_id"])
        dims = record["dims"]
        params = record["layout_params"]
        assert isinstance(dims, Mapping) and isinstance(params, Mapping)
        return cls(
            op_id=op_id,
            llvm_builtin=str(record["llvm_builtin"]),
            family=str(record["family"]),
            wave_size=int(record["wave_size"]),  # type: ignore[arg-type]
            input_dtype=str(record["input_dtype"]),
            output_dtype=str(record["output_dtype"]),
            m=_require_clean_int("M", str(dims["M"]), op_id),
            n=_require_clean_int("N", str(dims["N"]), op_id),
            k=_require_clean_int("K", str(dims["K"]), op_id),
            b=_require_clean_int("B", str(dims["B"]), op_id),
            r=_require_clean_int("R", str(dims["R"]), op_id),
            s=_require_clean_int("S", str(dims["S"]), op_id),
            k_ab_per_lane=_require_clean_int("ABK", str(params["ABK"]), op_id),
            a_k_num_access=_require_clean_int("AKN", str(params["AKN"]), op_id),
            a_repeat=_require_clean_int("AR", str(params["AR"]), op_id),
            b_k_num_access=_require_clean_int("BKN", str(params["BKN"]), op_id),
            b_repeat=_require_clean_int("BR", str(params["BR"]), op_id),
            c_m_per_lane=_require_clean_int("CM", str(params["CM"]), op_id),
            c_m_num_access=_require_clean_int("CMN", str(params["CMN"]), op_id),
            a_layout=str(record["a_layout"]),
            b_layout=str(record["b_layout"]),
            c_d_layout=str(record["c_d_layout"]),
            supported_targets=tuple(record["supported_targets"]),  # type: ignore[arg-type]
        )

@dataclass(frozen=True)
class MmaTraitsCatalog:
    """The loaded MMA traits table with lookup and selection.

    ``by_op_id`` holds the usable (parseable) intrinsics; ``reserved`` maps op_ids that
    are recognized but unsupported (block hiding) to a human-readable reason, so a lookup
    fails fast with the reason instead of a bare "not found".
    """

    by_op_id: Mapping[str, MmaTraits]
    reserved: Mapping[str, str]
    source_of_truth: str

    def get(self, op_id: str) -> MmaTraits:
        """Return traits for ``op_id`` or fail fast (reserved -> NotImplementedError)."""
        traits = self.by_op_id.get(op_id)
        if traits is not None:
            return traits
        if op_id in self.reserved:
            raise NotImplementedError(self.reserved[op_id])
        valid = sorted(self.by_op_id)
        raise ValueError(f"unknown op_id={op_id!r} -- valid: {valid}")

    def select(
        self,
        *,
        target: str,
        input_dtype: str,
        output_dtype: str,
        m: int,
        n: int,
        k: int,
        family: str = "dense",
    ) -> MmaTraits:
        """Resolve a single intrinsic by logical intent + target (the selector core).

        Fails fast if zero or multiple intrinsics match, naming the query and listing the
        candidates supported on ``target``.
        """
        matches = [
            traits
            for traits in self.by_op_id.values()
            if traits.family == family
            and traits.supports(target)
            and traits.input_dtype == input_dtype
            and traits.output_dtype == output_dtype
            and (traits.m, traits.n, traits.k) == (m, n, k)
        ]
        query = (
            f"target={target!r} input={input_dtype!r} output={output_dtype!r} "
            f"shape=({m},{n},{k}) family={family!r}"
        )
        if len(matches) == 1:
            return matches[0]
        available = sorted(
            t.op_id for t in self.by_op_id.values() if t.supports(target)
        )
        if not matches:
            raise ValueError(
                f"no MMA intrinsic for {query} -- available on {target!r}: {available}"
            )
        raise ValueError(
            f"ambiguous MMA selection for {query} -- "
            f"matched {[t.op_id for t in matches]}"
        )

def load_mma_traits(path: Path = DEFAULT_TRAITS_PATH) -> MmaTraitsCatalog:
    """Load and validate ``mma_traits.json`` into a :class:`MmaTraitsCatalog`.

    Block-hiding rows are recorded as reserved (not errors); malformed rows fail fast.
    """
    if not path.is_file():
        raise ValueError(f"mma_traits table not found -- path={path}")
    document = json.loads(path.read_text())
    operations = document.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ValueError(
            f"mma_traits table has no operations -- path={path}, "
            f"expected a non-empty 'operations' list"
        )
    by_op_id: dict[str, MmaTraits] = {}
    reserved: dict[str, str] = {}
    for record in operations:
        op_id = str(record["op_id"])
        try:
            by_op_id[op_id] = MmaTraits.from_record(record)
        except NotImplementedError as reason:
            # Block-hiding rows (X/Y markers): recognized but unsupported.
            reserved[op_id] = str(reason)
        except ValueError as reason:
            # Rows whose SOT schema lacks the CK layout params (e.g. the WMMA sheet's
            # wave64 rows have empty ABK/AKN/...). Not usable, not a load-abort; kept as
            # reserved with the reason so a lookup fails fast instead of vanishing.
            reserved[op_id] = str(reason)
    source_of_truth = str(document.get("_meta", {}).get("source_of_truth", "unknown"))
    return MmaTraitsCatalog(
        by_op_id=by_op_id, reserved=reserved, source_of_truth=source_of_truth
    )
