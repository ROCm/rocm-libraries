# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The kernel pool TensileLite ranks with tilewright: the Prediction table
plus the bank.

When the model carries kernel attributes, TensileLite ranks a merged pool
for each device:

    positions [0, P)   the Prediction table of the library, in table order
    position  P + j    bank[j]

The bank holds the solutions of the Equality rows (`EqualityMatching` row
predicate) of every library file of the same problem type, nested `Problem`
libraries included, whose hardware predicate accepts the device; in
ascending solution index, minus the table's indices. Two files share a
problem type when their stems agree once `_ID<chip ids>` and then
`_CU<count>` are removed before the final `_<arch>`.

The hardware predicate is evaluated the way TensileLite does: `Processor`,
`CUCount` (physical CU count), `PciChipId` (only on processors with chip-id
predicates, with the chip-id fallbacks of TensileLite's ChipIdRegistry) and
`TargetProcessor`, under `And` / `Or` / `Not`. A predicate this module does
not know raises instead of guessing.

`top1_index` of a `TILEWRIGHT_PICK_LOG` line is a position in this pool,
so the pipeline must index the same order.

Configuration (`planner` section, only when `planner.enabled`):
    bank_library_stems  explicit bank files; the rule above is applied to them
                        alone (they override `bank`)
    bank: auto          every file of the library's problem type
    bank: none          no bank (default)
The device is the bench device (`--hardware-device`): its processor is the
config's `arch`, its CU count `lib.hardware.device_hardware`, its chip id the
config's `hardware.pci_chip_id` or else the KFD topology `device_id`.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .arch import canonical_arch
from .dat import (
    is_tensile_contraction_logic,
    kernel_dat_info,
    library_logic_path,
    load_library_kernels,
    logic_stem,
    read_tensile_logic,
)

BANK_MODES = ("none", "auto")

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


# ── hardware predicates ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class BankDevice:
    """What a hardware predicate sees of a device. `chip_id` None: the
    device reports no PCI chip id (every PciChipId predicate is false)."""

    processor: str
    cu_count: int
    chip_id: Optional[int] = None


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


def chip_id_accepts(device: BankDevice, chip_id: int) -> bool:
    """AMDGPU PciChipIdEqual: ChipIdRegistry::canUseSolution."""
    if device.processor not in CHIP_ID_PROCESSORS or device.chip_id is None:
        return False
    return device.chip_id == chip_id or chip_id in CHIP_FALLBACKS.get(
        device.chip_id, ()
    )


def _logical(pred: Mapping[str, Any], device: BankDevice, leaf) -> Optional[bool]:
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


def amdgpu_accepts(pred: Mapping[str, Any], device: BankDevice) -> bool:
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


def hardware_accepts(pred: Optional[Mapping[str, Any]], device: BankDevice) -> bool:
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


# ── library files ────────────────────────────────────────────────────────────


def _row_library_indices(library: Mapping[str, Any]) -> Iterable[int]:
    kind = library.get("type")
    if kind == "Matching":
        for entry in library.get("table") or []:
            value = entry.get("value")
            if "index" in entry:
                yield int(entry["index"])
            elif isinstance(value, Mapping) and "index" in value:
                yield int(value["index"])
            else:
                raise ValueError("a Matching table entry names no solution index")
    elif kind == "Single":
        yield int(library["index"])
    else:
        raise ValueError(f"an Equality row holds a {kind!r} library")


def equality_indices(library: Optional[Mapping[str, Any]]) -> Set[int]:
    """Solution indices of the Equality rows of a file's `Problem` library,
    descending into rows that are `Problem` libraries themselves. A file
    whose library is not a `Problem` library has none."""
    out: Set[int] = set()
    if not isinstance(library, Mapping) or library.get("type") != "Problem":
        return out
    for row in library.get("rows") or []:
        sub = row.get("library") or {}
        if sub.get("type") == "Problem":
            out |= equality_indices(sub)
        elif (row.get("predicate") or {}).get("type") == "EqualityMatching":
            out.update(_row_library_indices(sub))
    return out


def same_type_stems(dat_dir: Any, library_stem: str) -> List[str]:
    """Stems of the contraction logic files in `dat_dir` of `library_stem`'s
    problem type (itself included), sorted."""
    key = problem_type_key(library_stem)
    stems = {
        logic_stem(name)
        for name in os.listdir(dat_dir)
        if name.startswith("TensileLibrary_") and is_tensile_contraction_logic(name)
    }
    return sorted(s for s in stems if problem_type_key(s) == key)


@dataclass
class Bank:
    """Bank kernels (`kernel_dat_info`, ascending solution index) and the
    stems that hold at least one of them."""

    kernels: List[Dict[str, Any]] = field(default_factory=list)
    stems: List[str] = field(default_factory=list)


def bank_kernels(
    dat_dir: Any,
    stems: Sequence[str],
    device: BankDevice,
    exclude: Iterable[int] = (),
) -> Bank:
    """The Equality-row solutions of the files `stems` that `device` accepts,
    in ascending solution index, without the indices in `exclude`. A solution
    index held by several files keeps the first file's solution."""
    skip = {int(i) for i in exclude}
    found: Dict[int, Tuple[str, Mapping[str, Any]]] = {}
    for stem in stems:
        path = library_logic_path(dat_dir, stem)
        data = read_tensile_logic(path)
        if data is None:
            raise ValueError(f"unreadable Tensile logic file: {path}")
        try:
            wanted = equality_indices(data.get("library")) - skip
            for sol in data.get("solutions") or []:
                index = sol.get("index")
                if index is None or int(index) not in wanted or int(index) in found:
                    continue
                if hardware_accepts(sol.get("hardwarePredicate"), device):
                    found[int(index)] = (stem, sol)
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(f"{path}: {e}") from None
    out = Bank()
    used: Set[str] = set()
    for index in sorted(found):
        stem, sol = found[index]
        try:
            out.kernels.append(kernel_dat_info(sol))
        except ValueError as e:
            raise ValueError(f"{stem}: {e}") from None
        used.add(stem)
    out.stems = [s for s in stems if s in used]
    return out


@dataclass
class LibraryPool:
    """The pool of one library: `table` then `bank` (both `kernel_dat_info`
    lists)."""

    table: List[Dict[str, Any]]
    bank: Bank = field(default_factory=Bank)

    @property
    def kernels(self) -> List[Dict[str, Any]]:
        return self.table + self.bank.kernels

    @property
    def indices(self) -> List[int]:
        return [int(k["sol_idx_global"]) for k in self.kernels]


def library_pool(
    dat_dir: Any,
    library_stem: str,
    device: Optional[BankDevice] = None,
    bank_stems: Optional[Sequence[str]] = None,
) -> LibraryPool:
    """The pool TensileLite ranks for `library_stem` on `device`: the bank is
    taken from `bank_stems`, or from every file of the problem type when it
    is None. Without a device there is no bank."""
    table = load_library_kernels(dat_dir, library_stem)
    if device is None:
        return LibraryPool(table)
    stems = (
        list(bank_stems)
        if bank_stems is not None
        else same_type_stems(dat_dir, library_stem)
    )
    return LibraryPool(
        table,
        bank_kernels(
            dat_dir, stems, device, exclude=(k["sol_idx_global"] for k in table)
        ),
    )


# ── configuration ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BankSource:
    """Where a run takes its bank from: explicit `stems`, or every file of
    the problem type (`auto`), or nowhere."""

    stems: Tuple[str, ...] = ()
    auto: bool = False

    @property
    def enabled(self) -> bool:
        return self.auto or bool(self.stems)


def bank_source(cfg: Optional[Mapping[str, Any]]) -> BankSource:
    """The bank source of a pipeline config (none unless the planner is on)."""
    planner = (cfg or {}).get("planner") or {}
    if not planner.get("enabled"):
        return BankSource()
    stems = tuple(str(s) for s in planner.get("bank_library_stems") or ())
    if stems:
        return BankSource(stems=stems)
    mode = planner.get("bank") or "none"
    if mode not in BANK_MODES:
        raise ValueError(f"planner.bank must be one of {BANK_MODES}: {mode!r}")
    return BankSource(auto=mode == "auto")


def bank_device(
    cfg: Mapping[str, Any], device_index: int = 0, topology: Optional[Path] = None
) -> BankDevice:
    """The bench device as hardware predicates see it. RuntimeError when the
    processor has chip-id predicates and neither the config nor the KFD
    topology gives the chip id."""
    from . import hardware as hwlib

    topo = hwlib.KFD_TOPOLOGY if topology is None else Path(topology)
    processor = canonical_arch(str(cfg.get("arch") or "").strip().lower())
    if not processor:
        raise ValueError("the config sets no arch")
    n_cu = hwlib.device_hardware(cfg, device_index, topo).n_cu
    chip_id = hwlib.device_pci_chip_id(cfg, device_index, topo)
    if chip_id is None and processor in CHIP_ID_PROCESSORS:
        raise RuntimeError(
            f"no PCI chip id for device {device_index}: set hardware.pci_chip_id"
        )
    return BankDevice(processor=processor, cu_count=n_cu, chip_id=chip_id)


def config_pool(
    cfg: Optional[Mapping[str, Any]],
    dat_dir: Any,
    library_stem: str,
    device_index: int = 0,
    topology: Optional[Path] = None,
) -> LibraryPool:
    """`library_pool` with the bank the config asks for."""
    source = bank_source(cfg)
    if not source.enabled:
        return LibraryPool(load_library_kernels(dat_dir, library_stem))
    assert cfg is not None
    return library_pool(
        dat_dir,
        library_stem,
        bank_device(cfg, device_index, topology),
        list(source.stems) if source.stems else None,
    )


def bank_library_stems(
    cfg: Optional[Mapping[str, Any]],
    dat_dir: Any,
    library_stem: str,
    device_index: int = 0,
    topology: Optional[Path] = None,
) -> List[str]:
    """Stems benches must load for the config's bank: the explicit stems, or
    with `bank: auto` the files that hold bank kernels for the device."""
    source = bank_source(cfg)
    if source.stems:
        return list(source.stems)
    if not source.auto:
        return []
    pool = config_pool(cfg, dat_dir, library_stem, device_index, topology)
    return [s for s in pool.bank.stems if s != library_stem]
