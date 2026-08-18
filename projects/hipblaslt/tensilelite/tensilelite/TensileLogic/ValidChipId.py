################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""
ValidChipId
---
For chip-ID-aware architectures, ensures logic YAML Device IDs are valid and
placed under the directory that matches the chip-ID fallback rules.
"""

import re
import sys

from pathlib import Path
from typing import NamedTuple, Optional, Set

from Tensile.Common.Architectures import (
    GFX_CHIP_IDS,
    LogicFileError,
    SUPPORTED_CHIP_ID_FALLBACKS,
    _extractArchInfo,
    _verifyPredicate,
    supportsChipIdPredicate,
)


def _chipIdKey(chip_id: str) -> str:
    return f"id={chip_id.lower()}"


def _archChipIds(gfx: str) -> Set[str]:
    return {_chipIdKey(chip_id) for chip_id in GFX_CHIP_IDS.get(gfx, [])}


def _sourceChipIds(gfx: str) -> Set[str]:
    return set(SUPPORTED_CHIP_ID_FALLBACKS).intersection(_archChipIds(gfx))


def _defaultChipIds(gfx: str) -> Set[str]:
    return _archChipIds(gfx) - _sourceChipIds(gfx)


def _fallbackFamily(chip_id: str, gfx: str) -> Set[str]:
    chip_id = chip_id.lower()
    direct_fallbacks = set(SUPPORTED_CHIP_ID_FALLBACKS.get(chip_id, []))
    if not direct_fallbacks:
        return {chip_id}

    family = {chip_id, *direct_fallbacks}
    for source, fallbacks in SUPPORTED_CHIP_ID_FALLBACKS.items():
        if source in _archChipIds(gfx) and direct_fallbacks.intersection(fallbacks):
            family.add(source)
            family.update(fallbacks)
    return family


class _ChipIdDir(NamedTuple):
    chipId: Optional[str]
    hasChipIdDir: bool
    isValidFormat: bool
    dirName: Optional[str] = None


def _chipIdDirFromPath(gfx: str, filepath: Path) -> _ChipIdDir:
    """Walk the file's ancestors from nearest to farthest and return the first
    component that looks like a chip-ID directory (or the bare gfx directory).
    Iterating in reverse and stopping at the first match prevents an outer
    ``gfx950`` segment (e.g. an enclosing logic-root or CI workspace path) from
    overwriting a real ``gfx950_id<chip>`` variant nearer the file.
    """
    base_pattern = re.compile(rf"^{re.escape(gfx)}$")
    chip_id_pattern = re.compile(rf"^{re.escape(gfx)}_id([0-9a-fA-F]+)$")
    malformed_chip_id_pattern = re.compile(rf"^{re.escape(gfx)}_([0-9a-fA-F]+)$")
    for part in reversed(filepath.parts[:-1]):
        match = chip_id_pattern.match(part)
        if match:
            return _ChipIdDir(
                chipId=_chipIdKey(match.group(1)),
                hasChipIdDir=True,
                isValidFormat=True,
                dirName=part,
            )
        match = malformed_chip_id_pattern.match(part)
        if match:
            return _ChipIdDir(
                chipId=_chipIdKey(match.group(1)),
                hasChipIdDir=True,
                isValidFormat=False,
                dirName=part,
            )
        if base_pattern.match(part):
            return _ChipIdDir(chipId=None, hasChipIdDir=False, isValidFormat=True, dirName=part)
    return _ChipIdDir(chipId=None, hasChipIdDir=False, isValidFormat=True)


def _reportChipIdFailure(filepath: Path, detail: str) -> None:
    print(f"Error: {detail} (file: {filepath})", file=sys.stderr)


def _validateChipIdPlacement(gfx: str, device_ids: Set[str], filepath: Path) -> Optional[str]:
    arch_ids = _archChipIds(gfx)
    source_ids = _sourceChipIds(gfx)
    default_ids = _defaultChipIds(gfx)
    chip_id_dir = _chipIdDirFromPath(gfx, filepath)

    if not chip_id_dir.hasChipIdDir:
        if not device_ids.issubset(arch_ids):
            return (
                f"base {gfx} logic may only declare chip IDs available for {gfx} "
                f"{sorted(arch_ids)}; found {sorted(device_ids)}"
            )
        return None

    if not chip_id_dir.isValidFormat:
        return (
            f"chip-ID directory '{chip_id_dir.dirName}' must use {gfx}_id<chip> format"
        )

    if chip_id_dir.chipId not in source_ids:
        return f"{gfx}_id directory uses non-source chip ID {chip_id_dir.chipId}"

    if chip_id_dir.chipId not in device_ids:
        return f"{chip_id_dir.chipId} directory must contain {chip_id_dir.chipId} in the YAML Device list"

    declared_default_ids = device_ids.intersection(default_ids)
    if declared_default_ids:
        return (
            f"{chip_id_dir.chipId} directory may not declare default fallback chip IDs "
            f"{sorted(declared_default_ids)}"
        )

    family = _fallbackFamily(chip_id_dir.chipId, gfx)
    if not device_ids.issubset(family):
        return (
            f"{chip_id_dir.chipId} directory may only declare chip IDs in fallback family "
            f"{sorted(family)}; found {sorted(device_ids)}"
        )

    return None


def _validateChipId(
    filepath: Path,
    logic_relative_path: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> bool:
    placement_path = logic_relative_path or filepath
    report_path = report_path or placement_path
    try:
        arch_info = _extractArchInfo(
            filepath,
            validateDeviceIds=False,
        )
    except LogicFileError as e:
        _reportChipIdFailure(report_path, f"Chip ID validation failed: {e}")
        return False

    try:
        if not supportsChipIdPredicate(arch_info.Gfx):
            return True

        if not arch_info.DeviceIds:
            _reportChipIdFailure(
                report_path,
                f"{arch_info.Gfx} logic must declare at least one Device chip ID",
            )
            return False

        # _extractArchInfo lowercases device IDs at parse time, so no further
        # case normalization is needed here.
        for device_id in arch_info.DeviceIds:
            _verifyPredicate(device_id, arch_info.Gfx)

        device_ids = set(arch_info.DeviceIds)
        # Walk the logic-root-relative path so that ancestor directories outside
        # the logic root (e.g. CI workspaces containing 'gfx950') cannot
        # masquerade as chip-ID directories.
        placement_error = _validateChipIdPlacement(arch_info.Gfx, device_ids, placement_path)
        if placement_error:
            _reportChipIdFailure(report_path, placement_error)
            return False

        return True
    except (LogicFileError, ValueError) as e:
        _reportChipIdFailure(report_path, f"ValidChipId failed ({type(e).__name__}): {e}")
        return False
