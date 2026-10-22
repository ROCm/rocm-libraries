# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The MMA catalog: one typed, validated row per matrix (MMA) instruction variant.

``data/mma_traits.json`` is a hand-maintained table; ``traits/README.md`` holds its schema and
edit procedure. This module gives the table a typed, validated face:

- :class:`MmaLayoutParams` -- the encoder layout parameters of one instruction.
- :class:`MmaScale` -- the scale contract of one scaled instruction.
- :class:`MmaTraits` -- one catalog row.
- :class:`MmaTraitsCatalog` -- the loaded table, with lookup by name and selection by intent.
- :func:`load_mma_traits` -- read the JSON into a catalog; any malformed row fails the load.

Operand dtypes are rocke's canonical ``core.dtypes`` names (``fp16``, ``fp8e4m3``, ``i8``, ...).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

from ....core.arch.target import MmaScaleBlockK, MmaScaleDType
from ....core.dtypes import normalize_dtype

__all__ = [
    "MmaLayoutParams",
    "MmaScale",
    "MmaTraits",
    "MmaTraitsCatalog",
    "load_mma_traits",
    "DEFAULT_TRAITS_PATH",
]

DEFAULT_TRAITS_PATH = Path(__file__).resolve().parent / "data" / "mma_traits.json"

SCHEMA_VERSION = 2

_VALID_FAMILIES = frozenset({"dense", "sparse", "scaled"})
_VALID_WAVE_SIZES = frozenset({32, 64})
_VALID_LAYOUT_SOURCES = frozenset(
    {"calculator", "isa_doc", "rocke_core", "ck", "legacy_table"}
)
# How an integer row's A and B accept signedness: "selectable" = each operand has its own
# signedness flag (integer WMMA), so the unsigned twin of the row's signed dtype also matches;
# "signed" = signed operands only (integer MFMA).
_VALID_SIGNEDNESS = frozenset({"selectable", "signed"})
_INTEGER_OPERAND_DTYPES = frozenset({"i8", "i4"})
_UNSIGNED_TWIN = {"i8": "u8", "i4": "u4"}

# JSON code of each layout parameter (CK trait name in the comment) -> MmaLayoutParams field.
_LAYOUT_PARAM_CODES = {
    "ABK": "k_ab_per_lane",  # kABKPerLane
    "AKN": "a_k_num_access",  # kAKNumAccess
    "AR": "a_repeat",  # kARepeat
    "BKN": "b_k_num_access",  # kBKNumAccess
    "BR": "b_repeat",  # kBRepeat
    "CM": "c_m_per_lane",  # kCMPerLane
    "CMN": "c_m_num_access",  # kCMNumAccess
    "AMB": "a_m_register_blocks",  # rocke: M blocks one lane holds in its A registers
}
_DIM_CODES = ("M", "N", "K", "B", "R", "S")

_ROW_FIELDS = frozenset(
    {
        "op_id",
        "llvm_builtin",
        "family",
        "wave_size",
        "supported_targets",
        "a_dtype",
        "b_dtype",
        "c_dtype",
        "d_dtype",
        "auto_select",
        "dims",
        "layout_params",
        "layout_source",
        "a_layout",
        "b_layout",
        "c_d_layout",
        "scale",
        "index_layout",
    }
)
_OPTIONAL_ROW_FIELDS = frozenset({"operand_signedness"})
_SCALE_FIELDS = frozenset({"block_k", "a_dtypes", "b_dtypes"})
_META_FIELDS = frozenset({"schema_version"})


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True)
class MmaLayoutParams:
    """The encoder layout parameters of one MMA instruction, in logical elements per lane.

    CK's seven trait parameters plus ``a_m_register_blocks`` (the M blocks one lane holds in its
    A registers). JSON code (CK trait name): ``k_ab_per_lane`` ABK (kABKPerLane),
    ``a_k_num_access`` AKN (kAKNumAccess), ``a_repeat`` AR (kARepeat), ``b_k_num_access`` BKN
    (kBKNumAccess), ``b_repeat`` BR (kBRepeat), ``c_m_per_lane`` CM (kCMPerLane),
    ``c_m_num_access`` CMN (kCMNumAccess), ``a_m_register_blocks`` AMB.
    """

    k_ab_per_lane: int
    a_k_num_access: int
    a_repeat: int
    b_k_num_access: int
    b_repeat: int
    c_m_per_lane: int
    c_m_num_access: int
    a_m_register_blocks: int

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not _positive_int(value):
                raise ValueError(
                    f"layout parameter {field.name}={value!r} must be a positive integer"
                )


@dataclass(frozen=True)
class MmaScale:
    """The scale contract of one scaled MMA instruction.

    ``block_k`` is the number of K elements that share one scale; ``a_dtypes`` / ``b_dtypes`` are
    the scale formats the instruction accepts for A and B (``core.arch`` ``MmaScaleDType`` values).
    A scale format changes how a scale byte is decoded, never a register layout, so the accepted
    formats are a set on one row rather than separate rows.
    """

    block_k: int
    a_dtypes: tuple[str, ...]
    b_dtypes: tuple[str, ...]

    def __post_init__(self) -> None:
        valid_block_k = sorted(int(v) for v in MmaScaleBlockK)
        if not _positive_int(self.block_k) or self.block_k not in valid_block_k:
            raise ValueError(
                f"scale block_k={self.block_k!r} must be one of {valid_block_k}"
            )
        valid_dtypes = sorted(v.value for v in MmaScaleDType)
        for operand, dtypes in (("a", self.a_dtypes), ("b", self.b_dtypes)):
            if not dtypes:
                raise ValueError(f"scale {operand}_dtypes lists no scale format")
            for dtype in dtypes:
                if dtype not in valid_dtypes:
                    raise ValueError(
                        f"scale {operand}_dtypes entry {dtype!r} is not one of "
                        f"{valid_dtypes}"
                    )


def _operand_accepts(row_dtype: str, asked: str, signedness: str | None) -> bool:
    """True if an operand of dtype ``row_dtype`` takes ``asked``: the same name, or the unsigned
    twin of a signed integer operand whose signedness is selectable."""
    if asked == row_dtype:
        return True
    return signedness == "selectable" and asked == _UNSIGNED_TWIN.get(row_dtype)


@dataclass(frozen=True)
class MmaTraits:
    """One MMA catalog row: one instruction variant, typed and validated.

    ``a_dtype`` / ``b_dtype`` are the A and B operand dtypes, ``c_dtype`` the accumulator taken in,
    ``d_dtype`` the result returned (they differ only where the instruction says so). ``m``..``s``
    are the dims: ``b`` independent blocks per instruction, ``r`` replication, ``s`` as named by
    the layout strings. ``auto_select=False`` marks a pin-only row: reachable by its name, never
    chosen by :meth:`MmaTraitsCatalog.select`.
    """

    op_id: str
    llvm_builtin: str
    family: str
    wave_size: int
    supported_targets: tuple[str, ...]
    a_dtype: str
    b_dtype: str
    c_dtype: str
    d_dtype: str
    operand_signedness: str | None
    auto_select: bool
    m: int
    n: int
    k: int
    b: int
    r: int
    s: int
    layout_params: MmaLayoutParams
    layout_source: str
    a_layout: str
    b_layout: str
    c_d_layout: str
    scale: MmaScale | None
    index_layout: str | None

    def __post_init__(self) -> None:
        row = f"mma_traits.json row {self.op_id!r}"
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
        for dtype_field in ("a_dtype", "b_dtype", "c_dtype", "d_dtype"):
            name = getattr(self, dtype_field)
            if not _is_canonical_dtype(name):
                raise ValueError(
                    f"{row}: {dtype_field} {name!r} is not a canonical rocke dtype name "
                    f"-- use the name core.dtypes.dtype_info returns, e.g. 'fp8e4m3'"
                )
        self._check_signedness(row)
        if self.auto_select is not True and self.auto_select is not False:
            raise ValueError(
                f"{row}: auto_select={self.auto_select!r} must be true or false"
            )
        if self.layout_source not in _VALID_LAYOUT_SOURCES:
            raise ValueError(
                f"{row}: layout_source {self.layout_source!r} is not one of "
                f"{sorted(_VALID_LAYOUT_SOURCES)}"
            )
        if self.family == "scaled" and self.scale is None:
            raise ValueError(f"{row}: family 'scaled' needs a scale contract")
        if self.family != "scaled" and self.scale is not None:
            raise ValueError(f"{row}: family {self.family!r} takes no scale contract")
        if self.family != "sparse" and self.index_layout is not None:
            raise ValueError(
                f"{row}: family {self.family!r} takes no index_layout (sparse rows only)"
            )
        # An empty supported_targets is accepted here: such a row is reachable by name only and
        # never matches a selection.

    def _check_signedness(self, row: str) -> None:
        """Integer rows state how their operands accept signedness; other rows omit the field."""
        integer_row = bool({self.a_dtype, self.b_dtype} & _INTEGER_OPERAND_DTYPES)
        if self.operand_signedness is None:
            if integer_row:
                raise ValueError(
                    f"{row}: integer operands need operand_signedness, one of "
                    f"{sorted(_VALID_SIGNEDNESS)}"
                )
        elif not integer_row:
            raise ValueError(
                f"{row}: operand_signedness is for integer rows only "
                f"(a_dtype={self.a_dtype!r}, b_dtype={self.b_dtype!r})"
            )
        elif self.operand_signedness not in _VALID_SIGNEDNESS:
            raise ValueError(
                f"{row}: operand_signedness {self.operand_signedness!r} is not one of "
                f"{sorted(_VALID_SIGNEDNESS)}"
            )

    def supports(self, target: str) -> bool:
        """True if this instruction is listed on ``target`` (e.g. ``gfx90a``)."""
        return target in self.supported_targets

    def accepts(self, *, a: str, b: str, c: str, d: str) -> bool:
        """True if this instruction takes A and B of canonical dtypes ``a`` and ``b``, accumulates
        in ``c`` and returns ``d``. A row whose integer signedness is selectable also takes the
        unsigned twin (``u8`` / ``u4``) of its signed A or B dtype, independently per operand.
        """
        return (
            _operand_accepts(self.a_dtype, a, self.operand_signedness)
            and _operand_accepts(self.b_dtype, b, self.operand_signedness)
            and (c, d) == (self.c_dtype, self.d_dtype)
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> MmaTraits:
        """Build a validated ``MmaTraits`` from one ``mma_traits.json`` row.

        Raises ``ValueError`` naming the row and the field for any malformed data.
        """
        op_id = str(record.get("op_id", "<no op_id>"))
        row = f"mma_traits.json row {op_id!r}"
        present = set(record)
        missing = sorted(_ROW_FIELDS - present)
        unexpected = sorted(present - _ROW_FIELDS - _OPTIONAL_ROW_FIELDS)
        if missing or unexpected:
            clauses = []
            if missing:
                clauses.append(f"missing fields {missing}")
            if unexpected:
                clauses.append(f"unexpected fields {unexpected}")
            raise ValueError(
                f"{row}: {'; '.join(clauses)} -- a row carries exactly the fields listed "
                f"in helpers/tiling/traits/README.md"
            )
        dims = _positive_int_table(record["dims"], "dims", _DIM_CODES, row)
        params = _positive_int_table(
            record["layout_params"], "layout_params", tuple(_LAYOUT_PARAM_CODES), row
        )
        return cls(
            op_id=op_id,
            llvm_builtin=str(record["llvm_builtin"]),
            family=str(record["family"]),
            wave_size=record["wave_size"],  # type: ignore[arg-type]
            supported_targets=tuple(record["supported_targets"]),  # type: ignore[arg-type]
            a_dtype=record["a_dtype"],  # type: ignore[arg-type]
            b_dtype=record["b_dtype"],  # type: ignore[arg-type]
            c_dtype=record["c_dtype"],  # type: ignore[arg-type]
            d_dtype=record["d_dtype"],  # type: ignore[arg-type]
            operand_signedness=record.get("operand_signedness"),  # type: ignore[arg-type]
            auto_select=record["auto_select"],  # type: ignore[arg-type]
            m=dims["M"],
            n=dims["N"],
            k=dims["K"],
            b=dims["B"],
            r=dims["R"],
            s=dims["S"],
            layout_params=MmaLayoutParams(
                **{_LAYOUT_PARAM_CODES[code]: value for code, value in params.items()}
            ),
            layout_source=str(record["layout_source"]),
            a_layout=str(record["a_layout"]),
            b_layout=str(record["b_layout"]),
            c_d_layout=str(record["c_d_layout"]),
            scale=_scale_from_record(record["scale"], row),
            index_layout=record["index_layout"],  # type: ignore[arg-type]
        )


def _is_canonical_dtype(name: object) -> bool:
    if not isinstance(name, str):
        return False
    try:
        return normalize_dtype(name) == name
    except ValueError:
        return False


def _positive_int_table(
    table: object, table_name: str, codes: tuple[str, ...], row: str
) -> dict[str, int]:
    """Read a JSON object that maps exactly ``codes`` to positive integers."""
    if not isinstance(table, Mapping) or set(table) != set(codes):
        found = sorted(table) if isinstance(table, Mapping) else table
        raise ValueError(
            f"{row}: {table_name} holds {found!r}, expected the keys {sorted(codes)}"
        )
    for code in codes:
        value = table[code]
        if not _positive_int(value):
            raise ValueError(
                f"{row}: {table_name}[{code!r}]={value!r} is not a positive integer "
                f"-- put the block count in dims['B']"
            )
    return {code: table[code] for code in codes}


def _scale_from_record(scale: object, row: str) -> MmaScale | None:
    """Read a row's ``scale`` field: ``null`` or a ``{block_k, a_dtypes, b_dtypes}`` object."""
    if scale is None:
        return None
    if not isinstance(scale, Mapping) or set(scale) != _SCALE_FIELDS:
        raise ValueError(
            f"{row}: scale must be null or an object with exactly the fields "
            f"{sorted(_SCALE_FIELDS)}"
        )
    return MmaScale(
        block_k=scale["block_k"],  # type: ignore[arg-type]
        a_dtypes=tuple(scale["a_dtypes"]),  # type: ignore[arg-type]
        b_dtypes=tuple(scale["b_dtypes"]),  # type: ignore[arg-type]
    )


def _scale_block_k(traits: MmaTraits) -> int | None:
    return traits.scale.block_k if traits.scale is not None else None


def _selection_key(traits: MmaTraits, target: str) -> tuple:
    """What a selection names: target, family, A/B/C/D dtypes, shape and -- for a scaled
    instruction -- the scale block size (block-16 and block-32 instructions decode scales
    differently, so a kernel states which one it holds)."""
    return (
        target,
        traits.family,
        traits.a_dtype,
        traits.b_dtype,
        traits.c_dtype,
        traits.d_dtype,
        traits.m,
        traits.n,
        traits.k,
        _scale_block_k(traits),
    )


def _describe_selection_key(key: tuple) -> str:
    target, family, a, b, c, d, m, n, k, block_k = key
    text = f"{target}, {family}, {a} x {b}, c={c}, d={d}, {m}x{n}x{k}"
    return text if block_k is None else f"{text}, scale_block_k={block_k}"


@dataclass(frozen=True)
class MmaTraitsCatalog:
    """The loaded MMA catalog: look a row up by name, or select one by intent."""

    by_op_id: Mapping[str, MmaTraits]

    def get(self, op_id: str) -> MmaTraits:
        """Return the row named ``op_id``, or fail fast listing the valid names."""
        traits = self.by_op_id.get(op_id)
        if traits is None:
            raise ValueError(
                f"unknown op_id={op_id!r} -- valid: {sorted(self.by_op_id)}"
            )
        return traits

    @staticmethod
    def _is_selectable(traits: MmaTraits, target: str, family: str) -> bool:
        """The one selection predicate: listed on ``target``, of ``family``, a single block per
        instruction, and not pin-only."""
        return (
            traits.supports(target)
            and traits.family == family
            and traits.b == 1
            and traits.auto_select
        )

    def select(
        self,
        *,
        target: str,
        a: str,
        b: str,
        c: str,
        m: int,
        n: int,
        k: int,
        d: str | None = None,
        family: str = "dense",
        scale_block_k: int | None = None,
    ) -> MmaTraits:
        """Select the one instruction for this intent on ``target``.

        ``a``, ``b``, ``c`` and ``d`` are canonical ``core.dtypes`` names; ``d=None`` means the
        result has the accumulator's dtype (``d = c``). ``scale_block_k`` (the K elements sharing
        one scale) is required for family ``scaled`` and refused for every other family. Selection
        keys are unique by a load-time invariant, so at most one row matches.
        """
        if family == "scaled" and scale_block_k is None:
            raise ValueError(
                "select(family='scaled') needs scale_block_k: block-16 and block-32 scaled "
                "instructions decode scales differently, so name the one the kernel holds"
            )
        if family != "scaled" and scale_block_k is not None:
            raise ValueError(
                f"scale_block_k={scale_block_k!r} applies to family 'scaled' only "
                f"-- family={family!r}"
            )
        d = c if d is None else d
        for traits in self.by_op_id.values():
            if (
                self._is_selectable(traits, target, family)
                and traits.accepts(a=a, b=b, c=c, d=d)
                and (traits.m, traits.n, traits.k) == (m, n, k)
                and _scale_block_k(traits) == scale_block_k
            ):
                return traits
        selectable = sorted(
            t.op_id
            for t in self.by_op_id.values()
            if self._is_selectable(t, target, family)
        )
        block = "" if scale_block_k is None else f" scale_block_k={scale_block_k}"
        raise ValueError(
            f"no MMA instruction for target={target!r} a={a!r} b={b!r} c={c!r} d={d!r} "
            f"shape=({m},{n},{k}) family={family!r}{block} "
            f"-- selectable on {target!r}: {selectable}"
        )


def _check_unique_selection(rows: list[MmaTraits]) -> None:
    """Load-time invariant: no two selectable rows answer the same selection key."""
    answers: dict[tuple, list[str]] = {}
    for traits in rows:
        if traits.b != 1 or not traits.auto_select:
            continue
        for target in traits.supported_targets:
            answers.setdefault(_selection_key(traits, target), []).append(traits.op_id)
    for key, op_ids in answers.items():
        if len(op_ids) > 1:
            raise ValueError(
                f"mma_traits.json: rows {op_ids} answer the same selection "
                f"({_describe_selection_key(key)}) -- set auto_select: false on all but "
                f"one (README: Selection)"
            )


def load_mma_traits(path: Path = DEFAULT_TRAITS_PATH) -> MmaTraitsCatalog:
    """Load ``mma_traits.json`` into a :class:`MmaTraitsCatalog`; any data error fails the load."""
    if not path.is_file():
        raise ValueError(f"mma_traits table not found -- path={path}")
    document = json.loads(path.read_text())
    meta = document.get("_meta", {})
    version = meta.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"mma_traits.json schema_version={version}, expected {SCHEMA_VERSION} "
            f"-- see helpers/tiling/traits/README.md"
        )
    unexpected_meta = sorted(set(meta) - _META_FIELDS)
    if unexpected_meta:
        raise ValueError(
            f"mma_traits.json _meta: unexpected fields {unexpected_meta} "
            f"-- see helpers/tiling/traits/README.md"
        )
    operations = document.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ValueError(
            f"mma_traits table has no operations -- path={path}, "
            f"expected a non-empty 'operations' list"
        )
    for op_id, count in Counter(str(r.get("op_id")) for r in operations).items():
        if count > 1:
            raise ValueError(
                f"mma_traits.json: op_id {op_id!r} appears {count} times -- one row per "
                f"instruction variant; name multi-format variants "
                f"'<stem>_<a_dtype>_<b_dtype>'"
            )
    rows = [MmaTraits.from_record(record) for record in operations]
    _check_unique_selection(rows)
    return MmaTraitsCatalog(by_op_id={traits.op_id: traits for traits in rows})
