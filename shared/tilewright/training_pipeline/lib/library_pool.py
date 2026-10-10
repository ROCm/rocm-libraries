# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The kernel pool of a run: the library kernels tilewright ranks, in the
order it ranks them.

The config's `library` section names the pool:

    kind      origami   the Prediction table of the host library file
                        (`hipblaslt.library_stem`), in table order: the
                        kernels Origami ranks. A host file without a
                        Prediction table is an error.
              bok       a bank of kernels: the solutions of the rows of the
                        `bok_rows` kinds of every library file of the host's
                        problem type, the host included, that the bench
                        device's hardware predicate accepts, in ascending
                        solution index.
              a list    those kinds in its order. The default is origami.
    bok_rows  the row kinds a bank of kernels takes: freesize and equality
              (the default) and gridbased.

A solution index is one entry of a pool, that of the first kind (and file)
that holds it, since TensileLite tells the kernels it ranks apart by
solution index. Solutions that repeat the name of another under another
index stay, as TensileLite ranks each of them; `summary` counts them. Two
files share a problem type when their stems agree once `_ID<chip ids>` and
then `_CU<count>` are removed before the final `_<arch>`. A pool reads only
the solution indices of a row, never the problem sizes its table is keyed
by; `read_type_rows`, which names the rows that can serve a problem, reads
the sizes of Equality rows too.

The hardware predicate is evaluated the way TensileLite does: `Processor`,
`CUCount` (physical CU count), `PciChipId` (only on processors with chip-id
predicates, with the chip-id fallbacks of TensileLite's ChipIdRegistry) and
`TargetProcessor`, under `And` / `Or` / `Not`. A predicate this module does
not know raises instead of guessing. The device is the bench device
(`--hardware-device`): its processor is the config's `arch`, its CU count
`lib.hardware.device_hardware`, its chip id the config's
`hardware.pci_chip_id` or else the KFD topology `device_id`.

A config without a `library` section names its pool with the planner's
legacy keys, which count only when `planner.enabled`:

    planner.bank: none            origami (the default)
    planner.bank: auto            [origami, bok] with bok rows equality
    planner.bank_library_stems    the same with the bank read from these
                                  files alone (it overrides `bank`)

A legacy host file without a Prediction table gives every solution of it,
in file order. Setting `library` and a legacy key is an error.

TensileLite ranks the Prediction table and then its bank, the solutions of
the Equality rows of the files of the problem type in ascending solution
index without the table's indices: `top1_index` of a `TILEWRIGHT_PICK_LOG`
line is a position in the pool of `[origami, bok]` with bok rows equality.

`config_pool` leaves out of the pool the kernels the config's blocklist
names (lib/correctness.py), as TensileLite does with the sidecar stage06
installs next to the model, so positions stay those of the runtime.
`describe` gives the lines preflight prints and `summary` the record the
report shows.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set
from typing import Tuple

from .arch import canonical_arch
from .correctness import config_blocklist
from .dat import (
    NoPredictionTable,
    is_tensile_contraction_logic,
    kernel_dat_info,
    library_logic_path,
    library_table_kernels,
    logic_stem,
    read_tensile_logic,
)
from .features import SIGNATURE_FIELDS

KINDS: Tuple[str, ...] = ("origami", "bok")
# bok_rows name -> the predicate TensileLite tags a library row of that kind
# with.
ROW_PREDICATES: Dict[str, str] = {
    "freesize": "FreeSizeMatching",
    "equality": "EqualityMatching",
    "gridbased": "GridBasedMatching",
}
DEFAULT_BOK_ROWS: Tuple[str, ...] = ("freesize", "equality")
# The predicate TensileLite tags a library row with -> the row's kind as
# `solution_rows` names it; a row of another predicate (an MLP library's
# TruePred) is `other`.
ROW_KINDS: Dict[str, str] = {
    "EqualityMatching": "Equality",
    "RangeMatching": "Range",
    "GridBasedMatching": "GridBased",
    "FreeSizeMatching": "FreeSize",
    "PredictionMatching": "Prediction",
}
# The problem properties of an Equality row's key -> the position, in (M, N,
# batch, K), of the size each reads: the first free index of A and of B, the
# first batch index and the first bound index.
EQUALITY_SIZE_PROPERTIES: Dict[Tuple[str, int], int] = {
    ("FreeSizeA", 0): 0,
    ("FreeSizeB", 0): 1,
    ("BatchSize", 0): 2,
    ("BoundSize", 0): 3,
}
# The rows of TensileLite's bank.
SERVED_BOK_ROWS: Tuple[str, ...] = ("equality",)
BANK_MODES = ("none", "auto")
LEGACY_KEYS: Tuple[str, ...] = ("planner.bank", "planner.bank_library_stems")

# Processors whose PciChipId predicates can be true
# (ChipIdRegistry::supportsChipIdPredicate).
CHIP_ID_PROCESSORS = frozenset({"gfx950"})

# Chip id -> the chip ids whose solutions it can also run
# (ChipIdRegistry fallbacks).
CHIP_FALLBACKS: Dict[int, Tuple[int, ...]] = {
    0x75B0: (0x75A0,),
    0x75A2: (0x75A0,),
    0x75B2: (0x75A0,),
    0x75A3: (0x75A0,),
    0x75B3: (0x75A0,),
    0x75A8: (0x75A0,),
    0x75B8: (0x75A0,),
}

# AMDGPU::Processor values that are not the digits of the name.
_PROCESSOR_VALUES = {"gfx90a": 910}

_ID_PART = re.compile(r"^(.*)_ID[0-9a-fA-F-]+$")
_CU_PART = re.compile(r"^(.*)_CU[0-9]+$")


def problem_type_key(stem: str) -> str:
    """`stem` without its `_ID<chip ids>` and then `_CU<count>` parts before
    the final `_<arch>`."""
    head, sep, arch = stem.rpartition("_")
    if not sep:
        return stem
    m = _ID_PART.match(head)
    if m:
        head = m.group(1)
    m = _CU_PART.match(head)
    if m:
        head = m.group(1)
    return f"{head}_{arch}"


def type_tail(stem: str) -> Optional[str]:
    """The data types, operation and arch of a stem's problem type (its key
    from `_Type_` on); None when it names none."""
    key = problem_type_key(stem)
    at = key.find("_Type_")
    return key[at:] if at >= 0 else None


def is_cu_file(stem: str) -> bool:
    """Whether a stem names a library for one CU count (`_CU<count>`)."""
    head, sep, _arch = stem.rpartition("_")
    if not sep:
        return False
    m = _ID_PART.match(head)
    return _CU_PART.match(m.group(1) if m else head) is not None


# ── configuration ────────────────────────────────────────────────────────────


def _names(value: Any, choices: Sequence[str], what: str) -> Tuple[str, ...]:
    items = [value] if isinstance(value, str) else value
    if (
        not isinstance(items, (list, tuple))
        or not items
        or any(not isinstance(v, str) or v not in choices for v in items)
        or len(set(items)) != len(items)
    ):
        raise ValueError(
            f"{what} must be one of {list(choices)} or a list of distinct ones, "
            f"got {value!r}"
        )
    return tuple(items)


def parse_kinds(value: Any) -> Tuple[str, ...]:
    """`library.kind`: a kind or a non-empty list of distinct kinds."""
    return _names(value, KINDS, "library.kind")


def parse_rows(value: Any) -> Tuple[str, ...]:
    """`library.bok_rows`: a row kind or a non-empty list of distinct ones."""
    return _names(value, tuple(ROW_PREDICATES), "library.bok_rows")


@dataclass(frozen=True)
class PoolSource:
    """What a config makes its pool of (`pool_source`): `kinds` in pool
    order, the row kinds of `bok`, the files `bok` reads (None: every file of
    the host's problem type) and whether the planner's legacy keys named
    it."""

    kinds: Tuple[str, ...] = ("origami",)
    bok_rows: Tuple[str, ...] = DEFAULT_BOK_ROWS
    stems: Optional[Tuple[str, ...]] = None
    legacy: bool = False

    @property
    def reads_bok(self) -> bool:
        return "bok" in self.kinds

    def text(self) -> str:
        kinds = self.kinds[0] if len(self.kinds) == 1 else f"[{', '.join(self.kinds)}]"
        if self.reads_bok:
            kinds += f" with bok rows {', '.join(self.bok_rows)}"
        if not self.legacy:
            return f"library.kind {kinds}"
        if self.stems is not None:
            return f"planner.bank_library_stems: {kinds} from the files it names"
        if self.reads_bok:
            return f"planner.bank auto: {kinds}"
        return f"{kinds} (the default)"


def library_section(cfg: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    """The config's `library` section; empty when it sets none."""
    lib = (cfg or {}).get("library")
    return lib if isinstance(lib, Mapping) else {}


def legacy_keys_set(cfg: Optional[Mapping[str, Any]]) -> List[str]:
    """The planner's legacy pool keys the config sets, to any value."""
    planner = (cfg or {}).get("planner")
    if not isinstance(planner, Mapping):
        return []
    return [k for k in LEGACY_KEYS if k.split(".", 1)[1] in planner]


def pool_source(cfg: Optional[Mapping[str, Any]]) -> PoolSource:
    """The pool source of a pipeline config. ValueError for a kind or row
    kind this module does not know, an unknown `planner.bank` mode, or a
    config that sets both `library` and a legacy key."""
    lib = library_section(cfg)
    if lib:
        legacy = legacy_keys_set(cfg)
        if legacy:
            raise ValueError(
                f"library and {' and '.join(legacy)} both name the kernel pool: "
                f"keep library and delete {', '.join(legacy)}"
            )
        kind, rows = lib.get("kind"), lib.get("bok_rows")
        return PoolSource(
            kinds=("origami",) if kind is None else parse_kinds(kind),
            bok_rows=DEFAULT_BOK_ROWS if rows is None else parse_rows(rows),
        )
    planner = (cfg or {}).get("planner") or {}
    if not planner.get("enabled"):
        return PoolSource(legacy=True)
    stems = tuple(str(s) for s in planner.get("bank_library_stems") or ())
    if stems:
        return PoolSource(("origami", "bok"), SERVED_BOK_ROWS, stems, legacy=True)
    mode = planner.get("bank") or "none"
    if mode not in BANK_MODES:
        raise ValueError(f"planner.bank must be one of {BANK_MODES}: {mode!r}")
    if mode == "auto":
        return PoolSource(("origami", "bok"), SERVED_BOK_ROWS, legacy=True)
    return PoolSource(legacy=True)


def config_problems(cfg: Optional[Mapping[str, Any]]) -> Tuple[List[str], List[str]]:
    """(errors, warnings) of the config's pool source: what TensileLite
    and the config's training make of it."""
    try:
        source = pool_source(cfg)
    except ValueError as e:
        return [str(e)], []
    errors: List[str] = []
    warnings: List[str] = []
    if source.legacy:
        return errors, warnings
    train = (cfg or {}).get("train") or {}
    if "bok_rows" in library_section(cfg) and not source.reads_bok:
        warnings.append("library.bok_rows is ignored: library.kind has no bok")
    if not source.reads_bok:
        return errors, warnings
    if "gridbased" in source.bok_rows:
        warnings.append(
            "library.bok_rows gridbased: a bridge for a bank of kernels built as "
            "GridBased rows, from which TensileLite answers problems of their "
            "sizes before tilewright ranks"
        )
    if str(train.get("catalog") or "v2") != "v3":
        errors.append(
            "library.kind bok needs train.catalog v3: only a v3 model tells "
            "TensileLite to rank the bank it was trained on"
        )
    elif train.get("trained_key_gate") is False:
        errors.append(
            "library.kind bok needs train.trained_key_gate: it keeps the kernels "
            "a bank gains after training to problems no trained key fits"
        )
    if "origami" not in source.kinds:
        warnings.append(
            "library.kind bok: TensileLite serves tilewright through a Prediction "
            "library only, so a model of this pool is trained and evaluated but "
            "cannot be served yet"
        )
    elif source.kinds.index("bok") < source.kinds.index("origami"):
        warnings.append(
            "library.kind lists bok before origami: TensileLite ranks the "
            "Prediction table before its bank, so its pool positions, and the "
            "member of a key it serves, are not the pipeline's"
        )
    return errors, warnings


# ── hardware predicates ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class PoolDevice:
    """What a hardware predicate sees of a device. `chip_id` None: the
    device reports no PCI chip id (every PciChipId predicate is false)."""

    processor: str
    cu_count: int
    chip_id: Optional[int] = None

    def text(self) -> str:
        chip = "" if self.chip_id is None else f", chip 0x{self.chip_id:04x}"
        return f"{self.processor} with {self.cu_count} CUs{chip}"


def processor_value(name: str) -> int:
    """AMDGPU::Processor value of a processor name (ValueError if unknown)."""
    if name in _PROCESSOR_VALUES:
        return _PROCESSOR_VALUES[name]
    if name.startswith("gfx") and name[3:].isdigit():
        return int(name[3:])
    raise ValueError(f"unknown processor {name!r}")


def runs_kernel_targeting(processor: str, target: str) -> bool:
    """AMDGPU::runsKernelTargeting: can `processor` run code built for
    `target`."""
    mine, other = processor_value(processor), processor_value(target)
    if other > mine:
        return False
    if other == mine:
        return True
    return other == processor_value("gfx900")


def chip_id_accepts(device: PoolDevice, chip_id: int) -> bool:
    """AMDGPU PciChipIdEqual: ChipIdRegistry::canUseSolution."""
    if device.processor not in CHIP_ID_PROCESSORS or device.chip_id is None:
        return False
    return device.chip_id == chip_id or chip_id in CHIP_FALLBACKS.get(
        device.chip_id, ()
    )


def _logical(pred: Mapping[str, Any], device: PoolDevice, leaf) -> Optional[bool]:
    kind, value = pred.get("type"), pred.get("value")
    if kind == "TruePred":
        return True
    if kind == "FalsePred":
        return False
    if kind == "And":
        return all(leaf(p, device) for p in value)
    if kind == "Or":
        return any(leaf(p, device) for p in value)
    if kind == "Not":
        return not leaf(value, device)
    return None


def amdgpu_accepts(pred: Mapping[str, Any], device: PoolDevice) -> bool:
    """An AMDGPU predicate on `device`."""
    out = _logical(pred, device, amdgpu_accepts)
    if out is not None:
        return out
    kind, value = pred.get("type"), pred.get("value")
    if kind == "Processor":
        return str(value) == device.processor
    if kind == "CUCount":
        return int(value) == device.cu_count
    if kind == "PciChipId":
        return chip_id_accepts(device, int(value))
    if kind == "TargetProcessor":
        return runs_kernel_targeting(device.processor, str(value))
    raise ValueError(f"unknown AMDGPU predicate {kind!r}")


def hardware_accepts(pred: Optional[Mapping[str, Any]], device: PoolDevice) -> bool:
    """A solution's `hardwarePredicate` on `device`; a solution without one
    runs anywhere."""
    if pred is None:
        return True
    out = _logical(pred, device, hardware_accepts)
    if out is not None:
        return out
    if pred.get("type") == "AMDGPU":
        return amdgpu_accepts(pred["value"], device)
    raise ValueError(f"unknown hardware predicate {pred.get('type')!r}")


def pool_device(
    cfg: Mapping[str, Any], device_index: int = 0, topology: Optional[Path] = None
) -> PoolDevice:
    """The bench device as hardware predicates see it. RuntimeError when the
    processor has chip-id predicates and neither the config nor the KFD
    topology gives the chip id."""
    from . import hardware as hwlib

    topo = None if topology is None else Path(topology)
    processor = canonical_arch(str(cfg.get("arch") or "").strip().lower())
    if not processor:
        raise ValueError("the config sets no arch")
    n_cu = hwlib.device_hardware(cfg, device_index, topo).n_cu
    chip_id = hwlib.device_pci_chip_id(cfg, device_index, topo)
    if chip_id is None and processor in CHIP_ID_PROCESSORS:
        raise RuntimeError(
            f"no PCI chip id for device {device_index}: set hardware.pci_chip_id"
        )
    return PoolDevice(processor=processor, cu_count=n_cu, chip_id=chip_id)


# ── library files ────────────────────────────────────────────────────────────

_ROW_TITLES = {
    "freesize": "a FreeSize",
    "equality": "an Equality",
    "gridbased": "a GridBased",
}


def _row_library_indices(library: Mapping[str, Any], row: str) -> Iterable[int]:
    kind = library.get("type")
    if kind in ("Matching", "FreeSize"):
        for entry in library.get("table") or []:
            value = entry.get("value")
            if "index" in entry:
                yield int(entry["index"])
            elif isinstance(value, Mapping) and "index" in value:
                yield int(value["index"])
            else:
                raise ValueError(f"a {kind} table entry names no solution index")
    elif kind == "Single":
        yield int(library["index"])
    else:
        raise ValueError(f"{_ROW_TITLES[row]} row holds a {kind!r} library")


def row_indices(
    library: Optional[Mapping[str, Any]], rows: Iterable[str]
) -> Dict[int, str]:
    """{solution index: row kind} of the rows of the kinds `rows`
    (ROW_PREDICATES) of a file's `Problem` library, descending into rows that
    are `Problem` libraries themselves; an index in several rows has the
    first one's kind. A file whose library is not a `Problem` library has
    none."""
    wanted = {ROW_PREDICATES[r]: r for r in rows}
    out: Dict[int, str] = {}

    def walk(lib: Any) -> None:
        if not isinstance(lib, Mapping) or lib.get("type") != "Problem":
            return
        for entry in lib.get("rows") or []:
            sub = entry.get("library") or {}
            if sub.get("type") == "Problem":
                walk(sub)
                continue
            row = wanted.get((entry.get("predicate") or {}).get("type"))
            if row is not None:
                for index in _row_library_indices(sub, row):
                    out.setdefault(index, row)

    walk(library)
    return out


def _listing(dat_dir: Any, library_stem: str) -> Tuple[List[str], List[str]]:
    """(stems of the contraction logic files of `dat_dir` of the host's
    problem type, the host included; stems of those of its data types and
    operation but another problem type), both sorted."""
    key, tail = problem_type_key(library_stem), type_tail(library_stem)
    stems = sorted(
        {
            logic_stem(name)
            for name in os.listdir(dat_dir)
            if name.startswith("TensileLibrary_") and is_tensile_contraction_logic(name)
        }
    )
    same = [s for s in stems if problem_type_key(s) == key]
    near = [
        s
        for s in stems
        if tail is not None and problem_type_key(s) != key and type_tail(s) == tail
    ]
    return same, near


def same_type_stems(dat_dir: Any, library_stem: str) -> List[str]:
    """Stems of the contraction logic files in `dat_dir` of `library_stem`'s
    problem type (itself included), sorted."""
    return _listing(dat_dir, library_stem)[0]


# (M, N, batch, K) of the problems an Equality row answers; None for a size
# its key does not hold, which the row does not compare.
RowSize = Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]


@dataclass(frozen=True)
class RowEntry:
    """A solution of a row of a library file: the file's stem, the row's
    kind (ROW_KINDS; `other` for another predicate), the solution index and,
    for an Equality row, the problem size its table entry is keyed by
    (`size`; None for a row of another kind)."""

    stem: str
    kind: str
    index: int
    size: Optional[RowSize] = None

    def answers_every_size(self) -> bool:
        """Whether the row can answer a problem of any size: a row of another
        kind than Equality, or an Equality row whose key holds no size."""
        return self.size is None or all(v is None for v in self.size)

    def answers(self, m: int, n: int, batch: int, k: int) -> bool:
        """Whether the row can answer a problem of that size: the sizes its
        key holds are the problem's."""
        if self.size is None:
            return True
        return all(
            want is None or want == got
            for want, got in zip(self.size, (m, n, batch, k))
        )


def _equality_size(properties: Any, key: Any) -> RowSize:
    """The (M, N, batch, K) that an Equality table entry's `key` holds under
    its library's `properties` (EQUALITY_SIZE_PROPERTIES); None for a size
    no property names."""
    out: List[Optional[int]] = [None, None, None, None]
    if isinstance(properties, (list, tuple)) and isinstance(key, (list, tuple)):
        for prop, value in zip(properties, key):
            if not isinstance(prop, Mapping) or isinstance(value, bool):
                continue
            at = EQUALITY_SIZE_PROPERTIES.get(
                (str(prop.get("type")), prop.get("index", 0))
            )
            if at is not None and isinstance(value, int):
                out[at] = value
    return (out[0], out[1], out[2], out[3])


def _held_entries(library: Mapping[str, Any]) -> List[Tuple[int, Any]]:
    """(solution index, table key) of the solutions a row's library holds: a
    Prediction table (no key), the entries of a Matching or FreeSize table, a
    Single library's (no key); none of a library of another type."""
    kind = library.get("type")
    if kind == "Prediction":
        return [
            (int(i), None)
            for i in library.get("table") or []
            if isinstance(i, int) and not isinstance(i, bool)
        ]
    if kind == "Single" and "index" in library:
        return [(int(library["index"]), None)]
    out = []
    if kind in ("Matching", "FreeSize"):
        for entry in library.get("table") or []:
            if not isinstance(entry, Mapping):
                continue
            value = entry.get("value")
            if "index" in entry:
                out.append((int(entry["index"]), entry.get("key")))
            elif isinstance(value, Mapping) and "index" in value:
                out.append((int(value["index"]), entry.get("key")))
    return out


def row_entries(library: Optional[Mapping[str, Any]], stem: str = "") -> List[RowEntry]:
    """Every solution of every row of a file's `Problem` library, rows and
    table entries in file order, descending into rows that are `Problem`
    libraries themselves; `stem` names the file."""
    out: List[RowEntry] = []

    def walk(lib: Any) -> None:
        if not isinstance(lib, Mapping) or lib.get("type") != "Problem":
            return
        for entry in lib.get("rows") or []:
            sub = entry.get("library") if isinstance(entry, Mapping) else None
            if not isinstance(sub, Mapping):
                continue
            if sub.get("type") == "Problem":
                walk(sub)
                continue
            kind = ROW_KINDS.get((entry.get("predicate") or {}).get("type"), "other")
            for index, key in _held_entries(sub):
                size = (
                    _equality_size(sub.get("properties"), key)
                    if kind == "Equality"
                    else None
                )
                out.append(RowEntry(stem, kind, index, size))

    walk(library)
    return out


def solution_rows(library: Optional[Mapping[str, Any]]) -> Dict[int, str]:
    """{solution index: the kind (ROW_KINDS) of the first row that holds it}
    of a file's `Problem` library, descending into rows that are `Problem`
    libraries themselves."""
    out: Dict[int, str] = {}
    for entry in row_entries(library):
        out.setdefault(entry.index, entry.kind)
    return out


@dataclass
class TypeRows:
    """The rows of the contraction logic files of a library's data types
    and operation (`read_type_rows`): every solution of every row
    (`entries`, the files in stem order), the file and row kind of each
    solution index (`rows`: an index several files hold keeps the first
    file's, and its first row's kind) and the kernel of each (`kernels`,
    lib.dat.kernel_dat_info of the first file's solution; one whose
    sizeMapping cannot be read is named by its solution name alone)."""

    entries: List[RowEntry] = field(default_factory=list)
    rows: Dict[int, Tuple[str, str]] = field(default_factory=dict)
    kernels: Dict[int, Dict[str, Any]] = field(default_factory=dict)


def _row_kernel(sol: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        return kernel_dat_info(dict(sol))
    except (KeyError, IndexError, TypeError, ValueError):
        name = sol.get("name") or sol.get("kernelName") or ""
        return {"sol_idx_global": sol.get("index"), "kernel_name": str(name)}


def read_type_rows(dat_dir: Any, library_stem: str) -> TypeRows:
    """The rows of every contraction logic file in `dat_dir` of
    `library_stem`'s data types and operation (its `type_tail`; its problem
    type when it names no `_Type_`), whatever their problem type: the rows
    that can answer a problem of the library's types, the Equality rows of
    its sizes and GridBased rows of other files among them. ValueError,
    naming the file, for one that cannot be read."""
    same, near = _listing(dat_dir, library_stem)
    out = TypeRows()
    for stem in sorted(set(same) | set(near)):
        path = library_logic_path(dat_dir, stem)
        data = read_tensile_logic(path)
        if data is None:
            raise ValueError(f"unreadable Tensile logic file: {path}")
        try:
            entries = row_entries(data.get("library"), stem)
            held = set()
            for entry in entries:
                out.rows.setdefault(entry.index, (stem, entry.kind))
                held.add(entry.index)
            for sol in data.get("solutions") or []:
                index = sol.get("index") if isinstance(sol, Mapping) else None
                if index is None or int(index) not in held:
                    continue
                if int(index) not in out.kernels:
                    out.kernels[int(index)] = _row_kernel(sol)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{path}: {e}") from None
        out.entries += entries
    return out


def type_rows(dat_dir: Any, library_stem: str) -> Dict[int, Tuple[str, str]]:
    """{solution index: (stem, row kind)} of the rows `read_type_rows`
    reads: an index several files hold keeps the first file's, in stem
    order. ValueError, naming the file, for one that cannot be read."""
    return read_type_rows(dat_dir, library_stem).rows


@dataclass
class _RowFile:
    """A file a bank reads: the solution index and row kind of every
    solution of its rows (`rows`), those the pool holds already (`held`) and
    the others the device accepts (`accepted`, their solutions)."""

    stem: str
    rows: Dict[int, str]
    held: Set[int] = field(default_factory=set)
    accepted: Dict[int, Mapping[str, Any]] = field(default_factory=dict)


def _row_files(
    dat_dir: Any,
    stems: Sequence[str],
    device: PoolDevice,
    rows: Sequence[str],
    held: Iterable[int],
) -> List[_RowFile]:
    """The files `stems` as a bank reads them, in order: an index the pool
    holds already, or an earlier file accepts, is held, and its hardware
    predicate is not evaluated."""
    taken = set(held)
    out = []
    for stem in stems:
        path = library_logic_path(dat_dir, stem)
        data = read_tensile_logic(path)
        if data is None:
            raise ValueError(f"unreadable Tensile logic file: {path}")
        try:
            f = _RowFile(stem, row_indices(data.get("library"), rows))
            for sol in data.get("solutions") or []:
                index = sol.get("index")
                if index is None or int(index) not in f.rows:
                    continue
                if int(index) in taken:
                    if int(index) not in f.accepted:
                        f.held.add(int(index))
                    continue
                if hardware_accepts(sol.get("hardwarePredicate"), device):
                    f.accepted[int(index)] = sol
                    taken.add(int(index))
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(f"{path}: {e}") from None
        out.append(f)
    return out


# ── pools ────────────────────────────────────────────────────────────────────


@dataclass
class Bank:
    """The kernels of a pool after its first kind (`kernel_dat_info`, pool
    order) and the files that hold at least one of them."""

    kernels: List[Dict[str, Any]] = field(default_factory=list)
    stems: List[str] = field(default_factory=list)


@dataclass
class PoolPart:
    """What one kind adds to a pool: its kernels in pool order, the files
    that hold them, the kernels per row kind (`prediction` for a Prediction
    table, `file` for every solution of a legacy host without one), the
    kernels it leaves out as copies of entries held earlier
    (`duplicates`) or blocked, and why a file it read adds none
    (`idle`)."""

    kind: str
    kernels: List[Dict[str, Any]] = field(default_factory=list)
    stems: List[str] = field(default_factory=list)
    rows: Dict[str, int] = field(default_factory=dict)
    duplicates: int = 0
    blocked: int = 0
    idle: Dict[str, str] = field(default_factory=dict)


@dataclass
class LibraryPool:
    """The pool of one library: `table`, the kernels of its first kind (of
    `[origami, ...]` the Prediction table), then `bank`, those of the kinds
    after it (`kernel_dat_info` lists). `parts` says what each kind added,
    and `near_type` names the files of the library's data types and
    operation that hold another problem type, which a bank does not read."""

    table: List[Dict[str, Any]]
    bank: Bank = field(default_factory=Bank)
    parts: List[PoolPart] = field(default_factory=list)
    source: Optional[PoolSource] = None
    stem: Optional[str] = None
    device: Optional[PoolDevice] = None
    near_type: List[str] = field(default_factory=list)

    @property
    def kernels(self) -> List[Dict[str, Any]]:
        return self.table + self.bank.kernels

    @property
    def indices(self) -> List[int]:
        return [int(k["sol_idx_global"]) for k in self.kernels]

    @property
    def files(self) -> List[str]:
        """The files that hold a kernel of the pool, in pool order."""
        if not self.parts:
            own = [self.stem] if self.stem and self.table else []
            return list(dict.fromkeys(own + self.bank.stems))
        return list(dict.fromkeys(s for p in self.parts for s in p.stems))


def _origami_part(
    dat_dir: Any, library_stem: str, held: Set[int], legacy: bool
) -> PoolPart:
    """The Prediction table of the library (of a legacy host without one,
    every solution of it), without the indices an earlier kind holds."""
    try:
        kernels, has_table = library_table_kernels(
            dat_dir, library_stem, require_table=not legacy
        )
    except NoPredictionTable as e:
        raise ValueError(
            f"library.kind origami: {e}; the kernels of a library without one "
            f"are a bank of kernels, library.kind bok"
        ) from None
    before = set(held)
    part = PoolPart("origami")
    for k in kernels:
        index = k.get("sol_idx_global")
        if index is not None and int(index) in before:
            part.duplicates += 1
            continue
        if index is not None:
            held.add(int(index))
        part.kernels.append(k)
    if part.kernels:
        part.stems = [library_stem]
        part.rows = {"prediction" if has_table else "file": len(part.kernels)}
    return part


def _bok_part(files: Sequence[_RowFile], held: Set[int]) -> PoolPart:
    """The bank kernels of `files` (`_row_files`) in ascending solution
    index, an index from the first file that accepts it."""
    part = PoolPart("bok")
    part.duplicates = len(set().union(*(f.held for f in files)))
    owner: Dict[int, Tuple[str, Mapping[str, Any], str]] = {}
    for f in files:
        for index, sol in f.accepted.items():
            owner.setdefault(index, (f.stem, sol, f.rows[index]))
    adds: Counter = Counter()
    for index in sorted(owner):
        stem, sol, row = owner[index]
        try:
            part.kernels.append(kernel_dat_info(sol))
        except ValueError as e:
            raise ValueError(f"{stem}: {e}") from None
        held.add(index)
        part.rows[row] = part.rows.get(row, 0) + 1
        adds[stem] += 1
    part.stems = [f.stem for f in files if adds[f.stem]]
    for f in files:
        if adds[f.stem]:
            continue
        rejected = len(f.rows) - len(f.held)
        if not f.rows:
            why = "no rows of the bank's kinds"
        elif not rejected:
            why = "the pool holds every kernel of it already"
        elif f.held:
            why = (
                f"the device's hardware predicate rejects the {rejected:,} kernels "
                f"of it the pool does not hold already"
            )
        else:
            why = f"the device's hardware predicate rejects its {rejected:,} kernels"
        part.idle.setdefault(f.stem, why)
    return part


def legacy_bank(
    dat_dir: Any, stems: Sequence[str], device: PoolDevice, exclude: Iterable[int] = ()
) -> PoolPart:
    """The bank a legacy pool reads from the files `stems`: the solutions of
    their Equality rows that `device` accepts, in ascending solution index,
    without the indices in `exclude`; an index several files hold is the
    first one's."""
    held = {int(i) for i in exclude}
    return _bok_part(_row_files(dat_dir, stems, device, SERVED_BOK_ROWS, held), held)


def resolve(
    dat_dir: Any,
    library_stem: str,
    source: PoolSource,
    device: Optional[PoolDevice] = None,
) -> LibraryPool:
    """The pool of `source` for the library `library_stem` on `device`
    (needed when the source reads `bok`). FileNotFoundError / ValueError,
    naming the file, when a file is missing or unreadable, the host of
    `origami` has no Prediction table, or a pool kernel holds a value
    TensileLite cannot read."""
    if source.reads_bok and device is None:
        raise ValueError("a pool with bok kernels needs the bench device")
    same: List[str] = []
    near: List[str] = []
    if source.reads_bok and source.stems is None:
        same, near = _listing(dat_dir, library_stem)
    held: Set[int] = set()
    parts: List[PoolPart] = []
    for kind in source.kinds:
        if kind == "origami":
            parts.append(_origami_part(dat_dir, library_stem, held, source.legacy))
            continue
        assert device is not None
        stems = list(source.stems) if source.stems is not None else same
        files = _row_files(dat_dir, stems, device, source.bok_rows, held)
        parts.append(_bok_part(files, held))
    rest = parts[1:]
    return LibraryPool(
        table=list(parts[0].kernels),
        bank=Bank(
            [k for p in rest for k in p.kernels],
            list(dict.fromkeys(s for p in rest for s in p.stems)),
        ),
        parts=parts,
        source=source,
        stem=library_stem,
        device=device,
        near_type=near,
    )


def without_blocked(pool: LibraryPool, blocklist: Optional[Any]) -> LibraryPool:
    """`pool` without the kernels `blocklist` (`lib.correctness.Blocklist`)
    names; its parts count them, its files stay."""
    if not blocklist:
        return pool
    parts = []
    for p in pool.parts:
        kept = blocklist.kept(p.kernels)
        parts.append(
            PoolPart(
                p.kind,
                kept,
                list(p.stems),
                dict(p.rows),
                p.duplicates,
                p.blocked + len(p.kernels) - len(kept),
                dict(p.idle),
            )
        )
    return LibraryPool(
        table=blocklist.kept(pool.table),
        bank=Bank(blocklist.kept(pool.bank.kernels), list(pool.bank.stems)),
        parts=parts,
        source=pool.source,
        stem=pool.stem,
        device=pool.device,
        near_type=list(pool.near_type),
    )


def config_pool(
    cfg: Optional[Mapping[str, Any]],
    dat_dir: Any,
    library_stem: str,
    device_index: int = 0,
    topology: Optional[Path] = None,
    apply_blocklist: bool = True,
) -> LibraryPool:
    """The pool the config names (`pool_source`) for `library_stem` on its
    bench device `device_index` (read only for a pool with bok kernels),
    without the kernels its blocklist names
    (`lib.correctness.config_blocklist`) unless `apply_blocklist` is
    false."""
    source = pool_source(cfg)
    device = None
    if source.reads_bok:
        assert cfg is not None
        device = pool_device(cfg, device_index, topology)
    pool = resolve(dat_dir, library_stem, source, device)
    return without_blocked(pool, config_blocklist(cfg) if apply_blocklist else None)


def pool_files(
    cfg: Optional[Mapping[str, Any]],
    dat_dir: Any,
    library_stem: str,
    device_index: int = 0,
    topology: Optional[Path] = None,
) -> List[str]:
    """The library files besides `library_stem` that benches load and
    stages index for the config's pool: those that hold a kernel of it (of
    `planner.bank_library_stems`, the files it names)."""
    source = pool_source(cfg)
    if source.stems is not None:
        return list(source.stems)
    if not source.reads_bok:
        return []
    pool = config_pool(
        cfg, dat_dir, library_stem, device_index, topology, apply_blocklist=False
    )
    return [s for s in pool.files if s != library_stem]


# ── composition ──────────────────────────────────────────────────────────────


def _keys(kernels: Iterable[Mapping[str, Any]]) -> Tuple[int, int]:
    """(keys, kernels of the largest key) of `kernels` grouped by their
    signature, the key of a v3 model without an attribute spec."""
    counts = Counter(tuple(int(k[f]) for f in SIGNATURE_FIELDS) for k in kernels)
    return len(counts), max(counts.values(), default=0)


def repeated_names(pool: LibraryPool) -> Tuple[int, int]:
    """(kernels of the pool with the solution name of an earlier one, those
    of them whose earlier namesake another kind holds)."""
    first: Dict[str, int] = {}
    n = across = 0
    parts = pool.parts or [PoolPart("", pool.kernels)]
    for i, part in enumerate(parts):
        for k in part.kernels:
            name = k.get("kernel_name")
            if not name:
                continue
            if name in first:
                n += 1
                across += first[name] != i
            else:
                first[name] = i
    return n, across


def summary(pool: LibraryPool) -> Dict[str, Any]:
    """What the pool is made of: per kind its files, kernels per row kind,
    kernels, copies of entries held earlier and blocked kernels it left out,
    keys, kernels of its largest key, and the files it read that add none
    (with why); for the whole pool its kernels, keys, largest key, the
    kernels that repeat the solution name of an earlier one (those whose
    namesake another kind holds apart) and the files of another problem type
    of its data types and operation."""
    parts = []
    for p in pool.parts:
        keys, largest = _keys(p.kernels)
        parts.append(
            {
                "kind": p.kind,
                "files": list(p.stems),
                "rows": dict(p.rows),
                "kernels": len(p.kernels),
                "duplicates": p.duplicates,
                "blocked": p.blocked,
                "keys": keys,
                "largest_key": largest,
                "idle_files": dict(p.idle),
            }
        )
    keys, largest = _keys(pool.kernels)
    repeats, across = repeated_names(pool)
    source = pool.source
    return {
        "source": source.text() if source is not None else None,
        "kinds": list(source.kinds) if source is not None else None,
        "bok_rows": (
            list(source.bok_rows) if source is not None and source.reads_bok else None
        ),
        "legacy": None if source is None else bool(source.legacy),
        "library_stem": pool.stem,
        "device": pool.device.text() if pool.device is not None else None,
        "parts": parts,
        "kernels": len(pool.kernels),
        "keys": keys,
        "largest_key": largest,
        "repeated_names": repeats,
        "repeated_across_kinds": across,
        "near_type_files": list(pool.near_type),
    }


def _count(n: int, one: str, many: str) -> str:
    return f"{n:,} {one if n == 1 else many}"


def _files_text(stems: Sequence[str], shown: int = 3) -> str:
    more = len(stems) - shown
    return ", ".join(stems[:shown]) + (f" and {more:,} more" if more > 0 else "")


ROW_TEXT = {"prediction": "Prediction table", "file": "every solution"}


def rows_text(rows: Mapping[str, int]) -> str:
    """The kernels per row kind of a part, as `summary` gives them."""
    return ", ".join(f"{ROW_TEXT.get(r, r)} {n:,}" for r, n in rows.items())


def describe(pool: LibraryPool) -> List[Tuple[str, str]]:
    """(level, line) of what the pool is made of, the lines preflight
    prints: `warn` for what TensileLite ranks otherwise or cannot rank,
    else `info`."""
    s = summary(pool)
    out: List[Tuple[str, str]] = []
    head = s["source"] or "the library's kernel pool"
    if s["device"]:
        head += f", on the bench device ({s['device']})"
    out.append(("info", head))
    for p in s["parts"]:
        kind = p["kind"]
        line = (
            f"{kind}: {p['kernels']:,} kernels in {p['keys']:,} keys (largest "
            f"{p['largest_key']:,})"
        )
        if p["rows"]:
            line += f", {rows_text(p['rows'])}"
        if p["files"]:
            n = len(p["files"])
            line += (
                f", from {'1 file' if n == 1 else f'{n:,} files'}: "
                f"{_files_text(p['files'])}"
            )
        if p["duplicates"]:
            line += f"; {p['duplicates']:,} copies of kernels held earlier left out"
        if p["blocked"]:
            line += f"; {p['blocked']:,} blocked"
        out.append(("info", line))
        if p["idle_files"]:
            idle = "; ".join(f"{st} ({why})" for st, why in p["idle_files"].items())
            out.append(("info", f"{kind}: files that add no kernel: {idle}"))
        if p["rows"].get("file"):
            out.append(
                (
                    "warn",
                    f"{kind}: the library has no Prediction table, so the pool is "
                    f"every solution of it; TensileLite ranks no library without "
                    f"one with tilewright",
                )
            )
        if kind != "bok":
            continue
        cu = [f for f in p["files"] if is_cu_file(f)]
        if cu:
            out.append(
                (
                    "info",
                    f"bok: {_count(len(cu), 'file holds', 'files hold')} kernels "
                    f"of one CU count ({_files_text(cu, 1)}): the pool depends on "
                    f"the device's CU count, and so on its partition mode",
                )
            )
        other = {r: n for r, n in p["rows"].items() if r not in SERVED_BOK_ROWS}
        if other:
            out.append(
                (
                    "warn",
                    f"bok: {sum(other.values()):,} kernels come from "
                    f"{' and '.join(other)} rows, which TensileLite's bank does not "
                    f"take: it ranks a smaller pool than the model's",
                )
            )
        if not p["kernels"]:
            out.append(("warn", "bok: no kernel of the bank on this device"))
    if s["near_type_files"]:
        near = s["near_type_files"]
        out.append(
            (
                "info",
                f"not read, of another problem type with the library's data "
                f"types and operation: {_files_text(near)}",
            )
        )
    out.append(
        (
            "info",
            f"pool: {s['kernels']:,} kernels in {s['keys']:,} keys (largest "
            f"{s['largest_key']:,})",
        )
    )
    if s["repeated_names"]:
        across = s["repeated_across_kinds"]
        out.append(
            (
                "warn" if across else "info",
                f"{_count(s['repeated_names'], 'kernel repeats', 'kernels repeat')} "
                f"the solution name of an earlier pool kernel under another "
                f"solution index"
                + (f" ({across:,} that of another kind)" if across else "")
                + ": TensileLite ranks each of them, so each is benched and ranked "
                "as a kernel of its own",
            )
        )
    source = pool.source
    if source is not None and source.stems and pool.stem:
        key = problem_type_key(pool.stem)
        other_type = [st for st in source.stems if problem_type_key(st) != key]
        if other_type:
            out.append(
                (
                    "warn",
                    f"planner.bank_library_stems names {_files_text(other_type)}, "
                    f"of another problem type than the library; TensileLite's bank "
                    f"takes the library's problem type only",
                )
            )
    return out
