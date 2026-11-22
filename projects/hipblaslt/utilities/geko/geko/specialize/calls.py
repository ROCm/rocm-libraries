# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Mask-128 hipBLASLt trace ingestion for ``--specialize``.

Unlike ``geko.bench.log.parse`` this keeps every logged fact of every record:
no column is dropped, no default is filled and no value is normalised. A call
identity compares the scalar text exactly as logged (``-0.000000`` and
``0.000000`` stay apart);
``CallIdentity.typed`` gives the YAML-typed fact derived from that text. A fact
absent from a record is ``UNKNOWN`` in that record's identity only. The one
exception: a NaN activation argument (``nan``, ``-nan``, ``+nan``, ``.nan``; the
sign of a NaN is meaningless) is canonicalised to ``nan`` so all NaN arguments
form one identity; ``inf`` and ``-inf`` stay distinct.

Inputs are summed as supplied: the same file given twice counts twice (no path
deduplication). Any input that is not a mask-128 trace refuses the whole load:
mask 64/34, unparseable or truncated YAML, and grouped-GEMM mask-128 records
(the producer writes unquoted comma lists, so they are not usable YAML). A
record without ``solution_Name``/``kernel_name`` (the RocRoller writer) is kept
with those observation fields ``UNKNOWN``; ``--specialize`` refuses that call
alone.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from geko.constants import (
    SPECIALIZE_IDENTITY_FIELDS,
    SPECIALIZE_OBSERVATION_FIELDS,
    SPECIALIZE_RECORD_FIELDS,
)

LOG_GLOB = "hipblaslt_*.log"
_INT_TAG = "tag:yaml.org,2002:int"


def configuration_name(directory: Path) -> str:
    """Run configuration of a trace directory: the ``run_configuration`` its ``run.json``
    records, else the directory name. The one rule of ``geko route-facts`` and ``--specialize``
    (a model-run ``.../<run>/out`` directory is named by its run.json, not ``out``). An
    unreadable run.json is refused where it is read, not here."""
    run = Path(directory) / "run.json"
    try:
        meta = json.loads(run.read_text()) if run.is_file() else {}
    except (OSError, ValueError):
        meta = {}
    name = meta.get("run_configuration") if isinstance(meta, dict) else None
    return name if isinstance(name, str) and name else Path(directory).name


class _Unknown:
    """A fact the record did not log; equal only to itself, a singleton under copy/pickle."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNKNOWN"

    def __reduce__(self) -> str:
        return "UNKNOWN"


UNKNOWN = _Unknown()

# One logged fact: (scalar text as logged, YAML-typed value derived from it).
Fact = Tuple[str, Any]
_NAN_ARGS = ("activation_arg0", "activation_arg1")
_NAN_TEXT = {"nan", "+nan", "-nan", ".nan", "+.nan", "-.nan"}


def _canonical(name: str, fact: Any) -> Any:
    """A NaN activation argument as the single fact ("nan", "nan"); any other fact unchanged."""
    if fact is not None and name in _NAN_ARGS and str(fact[0]).strip().lower() in _NAN_TEXT:
        return ("nan", "nan")
    return fact


class CallIdentity(Mapping):
    """Immutable, hashable mapping of SPECIALIZE_IDENTITY_FIELDS to the logged scalar text."""

    __slots__ = ("_logged", "_typed")

    def __init__(self, logged: Tuple[Any, ...], typed: Tuple[Any, ...]):
        if hasattr(self, "_logged"):
            raise TypeError("CallIdentity is immutable")
        object.__setattr__(self, "_logged", tuple(logged))
        object.__setattr__(self, "_typed", tuple(typed))

    @classmethod
    def from_facts(cls, facts: Mapping[str, Fact]) -> CallIdentity:
        known = [_canonical(name, facts.get(name)) for name in SPECIALIZE_IDENTITY_FIELDS]
        return cls(
            (UNKNOWN if f is None else f[0] for f in known),
            (UNKNOWN if f is None else f[1] for f in known),
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("CallIdentity is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("CallIdentity is immutable")

    def __reduce__(self):
        return (CallIdentity, (self._logged, self._typed))

    def _index(self, name: str) -> int:
        try:
            return SPECIALIZE_IDENTITY_FIELDS.index(name)
        except ValueError:
            raise KeyError(name) from None

    def __getitem__(self, name: str) -> Any:
        return self._logged[self._index(name)]

    def typed(self, name: str) -> Any:
        """The YAML-typed value of a fact (int/float/bool/str), or UNKNOWN."""
        return self._typed[self._index(name)]

    def __iter__(self) -> Iterator[str]:
        return iter(SPECIALIZE_IDENTITY_FIELDS)

    def __len__(self) -> int:
        return len(SPECIALIZE_IDENTITY_FIELDS)

    def __hash__(self) -> int:
        return hash(self._logged)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CallIdentity):
            return self._logged == other._logged
        return Mapping.__eq__(self, other)

    def __repr__(self) -> str:
        return f"CallIdentity({dict(self)!r})"


Observation = Tuple[Any, Any, Any]  # typed (solution_index, solution_Name, kernel_name)
Source = Tuple[str, str]  # (str(path), sha256 of the bytes parsed)


@dataclass
class CallEntry:
    """Calls of one identity: summed weight, recipes seen and source files, all call-weighted."""

    call_count: int = 0
    observations: Dict[Observation, int] = field(default_factory=dict)
    sources: Dict[Source, int] = field(default_factory=dict)

    def add(self, count: int, observation: Observation, source: Source) -> None:
        self.call_count += count
        self.observations[observation] = self.observations.get(observation, 0) + count
        self.sources[source] = self.sources.get(source, 0) + count


CallTable = Dict[CallIdentity, CallEntry]


def _log_files(paths: Path | str | List[Path | str]) -> List[Path]:
    if isinstance(paths, (str, Path)):
        path = Path(paths)
        if path.is_dir():
            files = sorted(path.glob(LOG_GLOB))
            if not files:
                raise ValueError(f"{path}: directory holds no {LOG_GLOB} trace")
            return files
        return [path]
    return [Path(p) for p in paths]


def _records(path: Path, text: str) -> List[Dict[str, Fact]]:
    """Records as {field: (logged text, typed value)}, validated as mask 128."""
    loader = yaml.SafeLoader(text)
    try:
        root = loader.get_single_node()
        if root is None:
            return []
        if not isinstance(root, yaml.SequenceNode):
            raise ValueError(f"{path}: not a mask-128 trace (expected a YAML list of records)")
        records = []
        for i, node in enumerate(root.value):
            where = f"{path} record {i}"
            if not isinstance(node, yaml.MappingNode):
                raise ValueError(f"{where}: not a mask-128 record (expected a mapping)")
            for key, value in node.value:  # grouped writer: unquoted comma lists break the fields
                if isinstance(key, yaml.ScalarNode) and (key.value == "grouped_gemm" or (
                        key.value == "function" and "grouped" in str(getattr(value, "value", "")))):
                    raise ValueError(f"{where}: grouped-GEMM mask-128 record; grouped GEMM is "
                                     "unsupported, whole input refused")
            record: Dict[str, Fact] = {}
            for key, value in node.value:
                if not isinstance(key, yaml.ScalarNode) or not isinstance(value, yaml.ScalarNode):
                    raise ValueError(f"{where}: not a mask-128 record (non-scalar field)")
                if key.value in record:
                    raise ValueError(f"{where}: duplicate field {key.value}")
                logged = text[value.start_mark.index:value.end_mark.index]
                record[key.value] = (logged, loader.construct_object(value))
            _check(where, record, node)
            records.append(record)
        return records
    except yaml.YAMLError as exc:
        hint = "; grouped-GEMM records are unsupported" if "grouped" in text else ""
        raise ValueError(f"{path}: not a mask-128 YAML trace ({exc.__class__.__name__}{hint})") from None
    finally:
        loader.dispose()


def _check(where: str, record: Dict[str, Fact], node: yaml.MappingNode) -> None:
    if "algo_method" in record:
        raise ValueError(f"{where}: mask-64 record (algo_method); --specialize needs mask 128 only")
    if "call_count" not in record:
        raise ValueError(f"{where}: not a mask-128 record, missing call_count")
    unexpected = sorted(set(record) - set(SPECIALIZE_RECORD_FIELDS))
    if unexpected:
        raise ValueError(f"{where}: unrecognised fields {', '.join(unexpected)}")
    count_node = next(v for k, v in node.value if k.value == "call_count")
    logged, count = record["call_count"]
    if count_node.tag != _INT_TAG or count_node.style is not None or count < 1:
        raise ValueError(f"{where}: call_count {logged} is not a positive integer")


class TraceError(ValueError):
    """A trace file that is not mask-128 input; ``path`` names the file."""

    def __init__(self, path: Path, message: str):
        super().__init__(message)
        self.path = Path(path)


def load_calls(paths: Path | str | List[Path | str], allow_empty: bool = False) -> CallTable:
    """Aggregate mask-128 trace records into ``CallIdentity -> CallEntry``.

    Args:
        paths: A trace file, a directory of ``hipblaslt_*.log`` process traces,
            or a list of trace files. All of them merge into one table; a path
            supplied twice is counted twice.
        allow_empty: Return an empty table, instead of refusing, when the
            inputs hold no record (the caller then needs its own zero-call proof).

    Raises:
        ValueError: With the reason, if any input is not a mask-128 trace
            (unparseable/truncated YAML, mask 64/34, incomplete records,
            unrecognised fields, invalid call_count, not UTF-8; a
            ``TraceError`` naming the file) or no record was found. Nothing is
            returned from a partly refused input.
    """
    table: CallTable = {}
    for path in _log_files(paths):
        with open(path, "rb") as f:
            data = f.read()
        source = (str(path), hashlib.sha256(data).hexdigest())
        try:
            text = data.decode()
        except UnicodeDecodeError as exc:
            raise TraceError(path, f"{path}: not a mask-128 trace (not UTF-8 at byte {exc.start})") from None
        try:
            records = _records(path, text)
        except ValueError as exc:
            raise TraceError(path, str(exc)) from None
        for record in records:
            observation = tuple(
                record[name][1] if name in record else UNKNOWN for name in SPECIALIZE_OBSERVATION_FIELDS
            )
            table.setdefault(CallIdentity.from_facts(record), CallEntry()).add(
                record["call_count"][1], observation, source
            )
    if not table and not allow_empty:
        raise ValueError(f"{paths}: no mask-128 call records")
    return table
