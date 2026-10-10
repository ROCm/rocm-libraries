# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The kernel pool by its first names: the Prediction table plus the bank.

lib/library_pool.py resolves every pool (`library.kind`, and the planner's
legacy `bank` / `bank_library_stems` keys, which make the pool
`[origami, bok]` with bok rows equality). This module keeps the names the
stages and scripts call, as thin wrappers of it:

    positions [0, P)   the pool's first kind (the Prediction table of the
                       library, in table order, of `[origami, ...]`)
    position  P + j    bank[j], the kernels of the kinds after it

`bank_source` says whether a pool has a bank, as stage05, stage06, stage07
and the orchestrator read it to serve one; `library_pool` and
`bank_kernels` build the legacy bank TensileLite ranks
(`TENSILE_TILEWRIGHT_BANK`): the Equality rows of the library's problem type
(of `bank_stems` when given) that the device accepts, in ascending solution
index, minus the table's indices.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, List, Mapping, Optional, Sequence
from typing import Set, Tuple

from . import library_pool as lp
from .library_pool import (  # noqa: F401
    BANK_MODES,
    CHIP_FALLBACKS,
    CHIP_ID_PROCESSORS,
    Bank,
    LibraryPool,
    amdgpu_accepts,
    chip_id_accepts,
    config_pool,
    hardware_accepts,
    problem_type_key,
    processor_value,
    runs_kernel_targeting,
    same_type_stems,
    without_blocked,
)

if TYPE_CHECKING:
    from .kernel_identity import LibraryIdentity

BankDevice = lp.PoolDevice


def equality_indices(library: Optional[Mapping[str, Any]]) -> Set[int]:
    """Solution indices of the Equality rows of a file's `Problem` library,
    descending into rows that are `Problem` libraries themselves. A file
    whose library is not a `Problem` library has none."""
    return set(lp.row_indices(library, ("equality",)))


def bank_kernels(
    dat_dir: Any,
    stems: Sequence[str],
    device: BankDevice,
    exclude: Iterable[int] = (),
) -> Bank:
    """The Equality-row solutions of the files `stems` that `device` accepts,
    in ascending solution index, without the indices in `exclude`. A solution
    index held by several files keeps the first file's solution."""
    part = lp.legacy_bank(dat_dir, stems, device, exclude)
    return Bank(part.kernels, part.stems)


def library_pool(
    dat_dir: Any,
    library_stem: str,
    device: Optional[BankDevice] = None,
    bank_stems: Optional[Sequence[str]] = None,
) -> LibraryPool:
    """The pool TensileLite ranks for `library_stem` on `device`: the bank is
    taken from `bank_stems`, or from every file of the problem type when it
    is None. Without a device there is no bank."""
    if device is None:
        return lp.resolve(dat_dir, library_stem, lp.PoolSource(legacy=True))
    source = lp.PoolSource(
        ("origami", "bok"),
        lp.SERVED_BOK_ROWS,
        None if bank_stems is None else tuple(bank_stems),
        legacy=True,
    )
    return lp.resolve(dat_dir, library_stem, source, device)


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
    """The bank of a pipeline config's pool (`lib.library_pool.pool_source`):
    the kinds after its first, from the files of `planner.bank_library_stems`
    or of the problem type; none for a pool of one kind."""
    source = lp.pool_source(cfg)
    if source.stems is not None:
        return BankSource(stems=source.stems)
    return BankSource(auto=len(source.kinds) > 1)


def bank_device(
    cfg: Mapping[str, Any], device_index: int = 0, topology: Optional[Path] = None
) -> BankDevice:
    """The bench device as hardware predicates see it
    (`lib.library_pool.pool_device`)."""
    return lp.pool_device(cfg, device_index, topology)


def bank_library_stems(
    cfg: Optional[Mapping[str, Any]],
    dat_dir: Any,
    library_stem: str,
    device_index: int = 0,
    topology: Optional[Path] = None,
) -> List[str]:
    """Stems benches must load for the config's pool besides the library's
    own (`lib.library_pool.pool_files`): the files of
    `planner.bank_library_stems`, else those holding a kernel of it."""
    return lp.pool_files(cfg, dat_dir, library_stem, device_index, topology)


def config_identity(
    cfg: Optional[Mapping[str, Any]],
    dat_dir: Any,
    library_stem: str,
    device_index: int = 0,
    topology: Optional[Path] = None,
) -> LibraryIdentity:
    """`lib.kernel_identity.library_identity` of the files the config's pool
    reads: the library and the bank's files, the files stage04 loads."""
    from .kernel_identity import library_identity

    bank = bank_library_stems(cfg, dat_dir, library_stem, device_index, topology)
    return library_identity(dat_dir, [library_stem, *bank])
