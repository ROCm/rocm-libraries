#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Audit hipBLASLt Tensile library logic YAML files.

Suitable for CI: exits with code 1 when validation errors are found. By default
the enabled checks are listed with ``--list-checks``. Component CI passes
``--changed-since`` so only logic YAML added or modified on the branch is audited.

Example:
    python3 projects/hipblaslt/utilities/library_audit/audit_hipblaslt_libraries.py \
        --hipblaslt-path projects/hipblaslt \
        --changed-since origin/develop \
        --workers 16
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Tuple

import yaml

from audit_checks import (
    _afc_false_like,
    effective_solutions,
    validate_device_names,
    validate_dict_structure,
    validate_filename_policy,
    validate_kernel_name_matches_filename,
    validate_streamk_gsu,
    validate_use_custom_main_loop_schedule,
)

try:
    YAML_LOADER = yaml.CSafeLoader
except AttributeError:  # pragma: no cover
    YAML_LOADER = yaml.SafeLoader


# ========================================================================
# gemm_name
# ========================================================================

# Maps Tensile DataType ``char`` tokens (from SolutionLibrary placeholder names) to enum indices.
TYPE_CHAR_TO_INDEX: dict[str, int] = {
    "S": 0,
    "D": 1,
    "C": 2,
    "Z": 3,
    "H": 4,
    "4xi8": 5,
    "I": 6,
    "B": 7,
    "I8": 8,
    "I64": 9,
    "X": 10,
    "F8N": 11,
    "B8N": 12,
    "F8B8N": 13,
    "B8F8N": 14,
    "F8": 15,
    "B8": 16,
    "F8B8": 17,
    "B8F8": 18,
    "F6": 19,
    "B6": 20,
    "F4": 21,
    "E8": 22,
    "E5M3": 23,
}

# Problem/feature suffixes appended after the GEMM type chars (see Tensile/SolutionLibrary.py).
_FILENAME_TYPE_SUFFIXES: tuple[str, ...] = (
    "_SABV",
    "_SAV",
    "_SAB",
    "_SCD",
    "_Bias",
    "_Grad",
    "_Aux",
    "_STB",
    "_STA",
    "_GG",
    "_GB",
    "_HAI",
    "_HA",
    "_UA",
    "_A",
)

_FILENAME_TYPE_SUFFIX_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"_SP[AB](?:ML\d+)?$"),
    re.compile(r"_MX[AB][A-Za-z0-9]+$"),
    re.compile(r"_M[A-Za-z0-9]+$"),
)

# Filename pattern: {arch}_Cijk_{Alayout}_{Blayout}_{typeToken}_...
FILENAME_RE = re.compile(
    r"^(?P<arch>[a-z0-9]+)_Cijk_(?P<alayout>A[a-z]{3})_(?P<blayout>B[a-z]{3})_(?P<type>.+?)_"
    r"(?:BH_|B_|UserArgs|HAS)",
    re.IGNORECASE,
)

EPILOGUE_FROM_BASENAME_RE = re.compile(
    r"Cijk_A[a-z]{3}_B[a-z]{3}_(?P<epilogue>.+?)_UserArgs",
    re.IGNORECASE,
)


def transpose_a_from_layout(alayout: str) -> bool:
    """Return TransposeA from an ``Axxx`` layout token (e.g. ``Alik``, ``Ailk``).

    Args:
        alayout: Four-character A-matrix layout token.

    Returns:
        TransposeA boolean ('l' in position 1 => True).

    Raises:
        ValueError: If ``alayout`` is not a valid A-layout token.
    """
    if len(alayout) != 4 or alayout[0] != "A":
        raise ValueError(f"Invalid A layout token: {alayout!r}")
    return alayout[1] == "l"


def transpose_b_from_layout(blayout: str) -> bool:
    """Return TransposeB from a ``Bxxx`` layout token (e.g. ``Bjlk``, ``Bljk``).

    Args:
        blayout: Four-character B-matrix layout token.

    Returns:
        TransposeB boolean ('l' in position 2 => True).

    Raises:
        ValueError: If ``blayout`` is not a valid B-layout token.
    """
    if len(blayout) != 4 or blayout[0] != "B":
        raise ValueError(f"Invalid B layout token: {blayout!r}")
    return blayout[2] == "l"


def _lookup_type_char(token: str) -> int:
    """Map a Tensile type char token to its DataType enum index.

    Args:
        token: Short type name from a library filename (e.g. ``B``, ``F8N``).

    Returns:
        Integer DataType enum index.

    Raises:
        ValueError: If the token is not recognized.
    """
    try:
        return TYPE_CHAR_TO_INDEX[token]
    except KeyError as exc:
        raise ValueError(f"Unknown type token {token!r}") from exc


def _strip_filename_type_suffixes(token: str) -> str:
    """Remove Tensile problem/feature suffixes from a filename type token.

    Suffixes such as ``_STA`` (SwizzleTensorA) are not part of the GEMM datatype
    encoding and must be stripped before decoding (``BBS_STA`` → ``BBS``).

    Args:
        token: Raw type substring from the library filename.

    Returns:
        Token with known suffixes removed.
    """
    if token == "S_MX":
        return token
    base = token
    while True:
        changed = False
        for suffix in _FILENAME_TYPE_SUFFIXES:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                changed = True
                break
        if not changed:
            for pattern in _FILENAME_TYPE_SUFFIX_RES:
                match = pattern.search(base)
                if match:
                    base = base[: match.start()]
                    changed = True
                    break
        if not changed:
            break
    return base


def _greedy_type_tokens(type_chars: str) -> list[str]:
    """Split a concatenated type string into Tensile ``DataType`` char tokens.

    Args:
        type_chars: GEMM type portion of a filename (suffixes already stripped).

    Returns:
        Ordered list of matched type char tokens.

    Raises:
        ValueError: If the string cannot be fully tokenized.
    """
    keys = sorted(TYPE_CHAR_TO_INDEX.keys(), key=len, reverse=True)
    tokens: list[str] = []
    index = 0
    while index < len(type_chars):
        matched = None
        for key in keys:
            if type_chars[index:].startswith(key):
                matched = key
                break
        if matched is None:
            raise ValueError(f"Cannot tokenize type string {type_chars!r} at {type_chars[index:]!r}")
        tokens.append(matched)
        index += len(matched)
    return tokens


def decode_gemm_type_token(token: str) -> Tuple[int, int, int, int, int]:
    """Decode a filename type token (e.g. ``B8BS``, ``BBS_STA``) to type indices.

    Args:
        token: Type substring from the library filename.

    Returns:
        Tuple of (DataType/A, DataTypeB, DestDataType, DataTypeE, ComputeDataType)
        as Tensile enum indices.

    Raises:
        ValueError: If the token cannot be decoded.
    """
    if token == "S_MX":
        return 0, 0, 0, 0, 0

    type_chars = _strip_filename_type_suffixes(token)
    parts = _greedy_type_tokens(type_chars)

    if len(parts) == 2:
        a = _lookup_type_char(parts[0])
        b = _lookup_type_char(parts[1])
        c = d = b
        compute = a
    elif len(parts) == 3:
        a = _lookup_type_char(parts[0])
        b = _lookup_type_char(parts[1])
        c = d = b
        compute = _lookup_type_char(parts[2])
    elif len(parts) == 4:
        a = _lookup_type_char(parts[0])
        b = _lookup_type_char(parts[1])
        c = d = _lookup_type_char(parts[2])
        compute = _lookup_type_char(parts[3])
    elif len(parts) == 5:
        a = _lookup_type_char(parts[0])
        b = _lookup_type_char(parts[1])
        c = d = _lookup_type_char(parts[2])
        compute = _lookup_type_char(parts[3])
    else:
        raise ValueError(f"Cannot decode GEMM type token {token!r} ({len(parts)} parts)")

    return a, b, c, d, compute


def parse_filename(stem: str) -> Optional[dict[str, object]]:
    """Parse architecture, layouts, and datatype token from a library filename stem.

    Args:
        stem: Filename without directory or extension.

    Returns:
        Dictionary of parsed fields, or None if the pattern does not match.
    """
    match = FILENAME_RE.match(stem)
    if not match:
        return None
    alayout = match.group("alayout")
    blayout = match.group("blayout")
    ta_a = transpose_a_from_layout(alayout)
    tb_b = transpose_b_from_layout(blayout)
    try:
        types = decode_gemm_type_token(match.group("type"))
    except (ValueError, KeyError):
        types = None
    return {
        "architecture": match.group("arch").lower(),
        "alayout": alayout,
        "blayout": blayout,
        "type_token": match.group("type"),
        "transpose_a": ta_a,
        "transpose_b": tb_b,
        "types": types,
    }


def extract_epilogue_key(name: str) -> str:
    """Extract epilogue suffix from a solution or library basename.

    Args:
        name: ``KernelNameMin``, ``SolutionNameMin``, or library filename stem.

    Returns:
        Epilogue key string, or empty string if not found.
    """
    match = EPILOGUE_FROM_BASENAME_RE.search(name)
    return match.group("epilogue") if match else ""

# ========================================================================
# lib_access
# ========================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

DEVICE_RE = re.compile(r"device\s+([0-9a-f]+)", re.IGNORECASE)

# Repository-relative Tensile logic root. Used to select files changed on a branch.
LOGIC_REL_PREFIX = (
    "projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/"
)

# Library types that use device-specific directories when those directories exist.
DEVICE_SCOPED_LIBRARY_TYPES = frozenset({"Equality"})

# Default (generic) device ids per architecture for path placement.
DEFAULT_DEVICE_BY_ARCH: dict[str, str] = {
    "gfx950": "75a0",
    "gfx1250": "73f0",
}


@dataclass(frozen=True)
class LibraryRecord:
    """Parsed metadata for one hipBLASLt logic YAML file."""

    path: Path
    format: str  # "list" or "dict"
    architecture: str
    schedule_name: str
    device_ids: tuple[str, ...]
    library_type: str
    problem_type: dict[str, Any]
    solutions: list[dict[str, Any]]
    index_order: Any
    exact_logic: Any
    range_logic: Any
    perf_metric: Optional[str]
    device_names_raw: Any = None
    default_solution: dict[str, Any] = field(default_factory=dict)


def _parse_devices(device_names: Any) -> tuple[str, ...]:
    if not device_names:
        return ()
    ids: list[str] = []
    for entry in device_names:
        if isinstance(entry, str):
            match = DEVICE_RE.search(entry)
            if match:
                ids.append(match.group(1).lower())
    return tuple(ids)


def _architecture_from_field(field: Any) -> str:
    if isinstance(field, dict):
        return str(field.get("Architecture", "")).lower()
    return str(field).lower()


def normalize_library(data: Any, path: Path) -> LibraryRecord:
    """Convert raw YAML data into a ``LibraryRecord``.

    Args:
        data: Parsed YAML root (list or dict).
        path: Path to the source file.

    Returns:
        Normalized library record.

    Raises:
        ValueError: If the document structure is not recognized.
    """
    if isinstance(data, list):
        if len(data) < 12:
            raise ValueError(f"List library has {len(data)} fields, expected at least 12")
        architecture = _architecture_from_field(data[2])
        return LibraryRecord(
            path=path,
            format="list",
            architecture=architecture,
            schedule_name=str(data[1]),
            device_ids=_parse_devices(data[3]),
            library_type=str(data[11]),
            problem_type=data[4],
            solutions=list(data[5]),
            index_order=data[6],
            exact_logic=data[7],
            range_logic=data[8],
            perf_metric=str(data[10]) if len(data) > 10 and data[10] is not None else None,
            device_names_raw=data[3],
        )

    if isinstance(data, dict):
        architecture = str(data.get("ArchitectureName", "")).lower()
        library_type = str(data.get("LibraryType", ""))
        lib_table = data.get("Library", {})
        if isinstance(lib_table, dict) and lib_table.get("distance"):
            library_type = str(lib_table["distance"])
        return LibraryRecord(
            path=path,
            format="dict",
            architecture=architecture,
            schedule_name=str(data.get("ScheduleName", "")),
            device_ids=_parse_devices(data.get("DeviceNames")),
            library_type=library_type,
            problem_type=dict(data.get("ProblemType", {})),
            solutions=list(data.get("Solutions", [])),
            index_order=data.get("IndexOrder"),
            exact_logic=data.get("ExactLogic"),
            range_logic=data.get("RangeLogic"),
            perf_metric=data.get("PerfMetric"),
            device_names_raw=data.get("DeviceNames"),
            default_solution=(
                dict(data["DefaultSolution"])
                if isinstance(data.get("DefaultSolution"), dict)
                else {}
            ),
        )

    raise ValueError(f"Unsupported YAML root type: {type(data).__name__}")


def iter_gemm_sizes(exact_logic: Any) -> Iterable[list[Any]]:
    """Yield problem-size vectors from ExactLogic entries.

    Args:
        exact_logic: ExactLogic field from a library file.

    Yields:
        Size vectors (typically length 4 or 8).
    """
    if not exact_logic:
        return
    for entry in exact_logic:
        if not isinstance(entry, (list, tuple)) or len(entry) < 1:
            continue
        size = entry[0]
        if isinstance(size, (list, tuple)):
            yield list(size)


def iter_performance_values(exact_logic: Any) -> Iterable[float]:
    """Yield performance numbers stored in ExactLogic mappings.

    Args:
        exact_logic: ExactLogic field from a library file.

    Yields:
        Floating-point performance values when present.
    """
    if not exact_logic:
        return
    for entry in exact_logic:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        mapping = entry[1]
        if isinstance(mapping, (list, tuple)) and len(mapping) >= 2:
            perf = mapping[1]
            if isinstance(perf, (int, float)):
                yield float(perf)


def solution_tiles(solutions: list[dict[str, Any]]) -> set[tuple[int, int, int]]:
    """Collect unique macro-tile shapes from solutions.

    Args:
        solutions: Solution dictionaries from the library file.

    Returns:
        Set of (MacroTile0, MacroTile1, DepthU) tuples.
    """
    tiles: set[tuple[int, int, int]] = set()
    for sol in solutions:
        if not isinstance(sol, dict):
            continue
        m0 = sol.get("MacroTile0")
        m1 = sol.get("MacroTile1")
        du = sol.get("DepthU")
        if m0 is not None and m1 is not None and du is not None:
            tiles.add((int(m0), int(m1), int(du)))
    return tiles


def solution_epilogue_keys(solutions: list[dict[str, Any]]) -> set[str]:
    """Collect epilogue keys from solution names.

    Args:
        solutions: Solution dictionaries.

    Returns:
        Set of epilogue key strings.
    """
    keys: set[str] = set()
    for sol in solutions:
        if not isinstance(sol, dict):
            continue
        for field in ("KernelNameMin", "SolutionNameMin"):
            value = sol.get(field)
            if isinstance(value, str):
                key = extract_epilogue_key(value)
                if key:
                    keys.add(key)
                    break
    return keys


def path_device_id(path: Path, architecture: str) -> Optional[str]:
    """Extract device id from a path segment like ``gfx950_id75a3``.

    Args:
        path: Library file path.
        architecture: Architecture string (e.g. ``gfx950``).

    Returns:
        Lowercase device id or None.
    """
    prefix = f"{architecture}_id"
    for part in path.parts:
        lower = part.lower()
        if lower.startswith(prefix):
            return lower[len(prefix) :]
    return None


def display_library_type(library_type: str) -> str:
    """Map YAML library type to the report label.

    Prediction libraries live under Origami and are reported as Origami.

    Args:
        library_type: Raw ``LibraryType`` from the YAML file.

    Returns:
        Library type string used in statistics and reports.
    """
    if library_type == "Prediction":
        return "Origami"
    return library_type


def path_library_folder(path: Path) -> str:
    """Return the library-type folder name (Equality, Origami, GridBased, ...).

    Args:
        path: Library file path.

    Returns:
        Folder name, or empty string if unknown.
    """
    known = {
        "Equality",
        "Origami",
        "GridBased",
        "Range",
        "Prediction",
        "FreeSize",
    }
    for part in path.parts:
        if part in known:
            return part
    return ""


def discover_device_folders(logic_root: Path) -> dict[str, set[str]]:
    """Map architecture to device ids that have dedicated ``{arch}_id*`` folders.

    Args:
        logic_root: Root hipBLASLt logic library directory.

    Returns:
        Dict arch -> set of device ids.
    """
    result: dict[str, set[str]] = {}
    for arch_dir in logic_root.iterdir():
        if not arch_dir.is_dir():
            continue
        arch = arch_dir.name.lower()
        ids: set[str] = set()
        for child in arch_dir.iterdir():
            if child.is_dir() and child.name.lower().startswith(f"{arch}_id"):
                ids.add(child.name.lower()[len(f"{arch}_id") :])
        if ids:
            result[arch] = ids
    return result

# ========================================================================
# cache
# ========================================================================

from pathlib import Path
from typing import Any, Optional

# Bump when validation rules or result schema change.
CACHE_VERSION = 6

# Relative path from hipBLASLt project root to Tensile logic libraries on disk.
HIPBLASLT_LOGIC_REL = Path(
    "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full"
)


def logic_dir_from_hipblaslt(hipblaslt_path: Path) -> Path:
    """Resolve hipBLASLt logic library directory from a project root.

    Args:
        hipblaslt_path: Path to ``rocm-libraries/projects/hipblaslt``.

    Returns:
        Absolute path to ``Tensile/Logic/asm_full``.
    """
    return hipblaslt_path.resolve() / HIPBLASLT_LOGIC_REL


def default_hipblaslt_path() -> Path:
    """Return the default hipBLASLt project path for this environment.

    Returns:
        First existing candidate path, otherwise the first default candidate.
    """
    workspace = os.environ.get("GITHUB_WORKSPACE")
    candidates: list[Path] = []
    if workspace:
        candidates.append(Path(workspace) / "projects" / "hipblaslt")
    candidates.extend(
        [
            Path(__file__).resolve().parents[2],
            Path("projects/hipblaslt"),
            Path("rocm-libraries/projects/hipblaslt"),
        ]
    )
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


def resolve_logic_dir(hipblaslt_path: Optional[Path] = None) -> Path:
    """Resolve the hipBLASLt logic library directory from CLI arguments.

    The directory is derived from ``--hipblaslt-path`` or the default hipBLASLt root.

    Args:
        hipblaslt_path: Optional hipBLASLt project root.

    Returns:
        Resolved logic directory path.
    """
    base = hipblaslt_path.resolve() if hipblaslt_path is not None else default_hipblaslt_path()
    return logic_dir_from_hipblaslt(base)


def _file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


class AuditCache:
    """JSON-backed cache of per-library audit outcomes."""

    def __init__(
        self,
        cache_dir: Path,
        logic_dir: Path,
        enabled: bool = True,
        checks_key: str = "",
    ) -> None:
        """Initialize cache storage.

        Args:
            cache_dir: Directory for ``entries.json``.
            logic_dir: hipBLASLt logic root (stored in manifest for invalidation).
            enabled: When False, all lookups miss and nothing is written.
            checks_key: Enabled check ids. A different set invalidates the cache.

        Raises:
            No exceptions are raised.
        """
        self.cache_dir = cache_dir
        self.logic_dir = logic_dir.resolve()
        self.enabled = enabled
        self.checks_key = checks_key
        self._entries: dict[str, dict[str, Any]] = {}
        self._dirty = False
        self._hits = 0
        self._misses = 0
        if enabled:
            self._load()

    def _manifest_path(self) -> Path:
        return self.cache_dir / "manifest.json"

    def _entries_path(self) -> Path:
        return self.cache_dir / "entries.json"

    def _load(self) -> None:
        manifest_path = self._manifest_path()
        entries_path = self._entries_path()
        if not manifest_path.is_file() or not entries_path.is_file():
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("version") != CACHE_VERSION:
                return
            if manifest.get("checks") != self.checks_key:
                return
            if Path(manifest.get("logic_dir", "")).resolve() != self.logic_dir:
                return
            self._entries = json.loads(entries_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._entries = {}

    def save(self) -> None:
        """Persist cache to disk if entries changed.

        Raises:
            OSError: If writing cache files fails.
        """
        if not self.enabled or not self._dirty:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "version": CACHE_VERSION,
            "logic_dir": str(self.logic_dir),
            "checks": self.checks_key,
        }
        self._manifest_path().write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self._entries_path().write_text(json.dumps(self._entries, indent=2), encoding="utf-8")
        self._dirty = False

    @property
    def stats(self) -> dict[str, int]:
        """Return cache hit/miss counts.

        Returns:
            Dict with ``hits`` and ``misses`` keys.
        """
        return {"hits": self._hits, "misses": self._misses}

    def lookup(self, path: Path) -> Optional[dict[str, Any]]:
        """Return cached result if file signature matches.

        Args:
            path: Library YAML path.

        Returns:
            Cached result dict, or None on miss.
        """
        if not self.enabled:
            self._misses += 1
            return None
        key = self._rel_key(path)
        entry = self._entries.get(key)
        if not entry:
            self._misses += 1
            return None
        try:
            if entry["signature"] != _file_signature(path):
                self._misses += 1
                return None
        except OSError:
            self._misses += 1
            return None
        self._hits += 1
        return entry["result"]

    def store(self, path: Path, result: dict[str, Any]) -> None:
        """Store audit result for a file.

        Args:
            path: Library YAML path.
            result: Serializable per-file audit outcome.

        Raises:
            OSError: If the file cannot be stat'd.
        """
        if not self.enabled:
            return
        key = self._rel_key(path)
        self._entries[key] = {
            "signature": _file_signature(path),
            "result": result,
        }
        self._dirty = True

    def _rel_key(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.logic_dir))
        except ValueError:
            return str(path.resolve())

# ========================================================================
# validators
# ========================================================================

ErrorSink = Callable[[str, str, str], None]

ORIGAMI_ASSERT_FIELDS = (
    "AssertFree0ElementMultiple",
    "AssertFree1ElementMultiple",
    "AssertSummationElementMultiple",
)


def _is_origami_library(record: LibraryRecord) -> bool:
    """Return True if the library is stored under Origami or is Prediction type.

    Args:
        record: Parsed library record.

    Returns:
        True when Origami validation rules apply.
    """
    if record.library_type == "Prediction":
        return True
    return path_library_folder(record.path) == "Origami"


def _emit(sink: ErrorSink, path: Path, code: str, message: str) -> None:
    sink(str(path), code, message)


@dataclass(frozen=True)
class FileAuditContext:
    """How a changed library relates to the branch base.

    Attributes:
        added: True when the path is new or a rename destination.
        base_format: Root format (``list`` or ``dict``) at the base revision.
    """

    added: bool = False
    base_format: Optional[str] = None


def validate_list_format_added(
    record: LibraryRecord,
    sink: ErrorSink,
    device_folders: Optional[dict[str, set[str]]] = None,
    context: Optional[FileAuditContext] = None,
) -> None:
    """Fail when a list-format library is newly introduced.

    Existing list files that are only edited are allowed. Dict format is allowed
    on every architecture. A file that changes from dict to list fails.

    Args:
        record: Parsed library record.
        sink: Callback ``(path, code, message)`` for each error.
        device_folders: Unused. Present so every check shares one signature.
        context: Addition flag and the base revision's root format.

    Returns:
        None.

    Raises:
        No exceptions are raised.
    """
    del device_folders
    ctx = context or FileAuditContext()
    introduced = ctx.added or ctx.base_format == "dict"
    if record.format == "list" and introduced:
        _emit(
            sink,
            record.path,
            "list_format_added",
            "List-format library logic was added; dict format is required for new libraries.",
        )


def validate_activation_func_call(
    record: LibraryRecord,
    sink: ErrorSink,
    device_folders: Optional[dict[str, set[str]]] = None,
    context: Optional[FileAuditContext] = None,
) -> None:
    """Require the filled ActivationFuncCall value to be false-like.

    Solutions are filled with ``LibraryIO.fillSolutionDefaults``. TensileLite's
    global default is True, so an omitted key fails unless the solution or the
    file ``DefaultSolution`` sets a false-like value.

    Args:
        record: Parsed library record.
        sink: Error callback.
        device_folders: Unused. Present so every check shares one signature.
        context: Unused per-file audit context.

    Returns:
        None.

    Raises:
        No exceptions are raised.
    """
    del device_folders, context
    for sol in effective_solutions(record, sink, "activation_func_call"):
        if "ActivationFuncCall" not in sol:
            continue
        value = sol.get("ActivationFuncCall")
        if _afc_false_like(value):
            continue
        sol_idx = sol.get("SolutionIndex", "?")
        _emit(
            sink,
            record.path,
            "activation_func_call",
            f"SolutionIndex {sol_idx}: ActivationFuncCall must be false (found {value!r}).",
        )


def validate_epilogue_consistency(record: LibraryRecord, sink: ErrorSink) -> None:
    """For list-based libraries, all solutions must share the same epilogue key.

    Args:
        record: Parsed library record.
        sink: Error callback.
    """
    if record.format != "list" or not record.solutions:
        return
    keys = solution_epilogue_keys(record.solutions)
    if len(keys) > 1:
        _emit(
            sink,
            record.path,
            "epilogue_mismatch",
            f"Solutions have differing epilogue keys: {sorted(keys)}",
        )


def validate_performance_zero(record: LibraryRecord, sink: ErrorSink) -> None:
    """ExactLogic performance values must be zero when numeric.

    Args:
        record: Parsed library record.
        sink: Error callback.
    """
    for perf in iter_performance_values(record.exact_logic):
        if perf == 0.0:
            continue
        if math.isinf(perf):
            _emit(
                sink,
                record.path,
                "performance_nonzero",
                f"ExactLogic performance is {perf!r}, expected 0.0",
            )
            continue
        if abs(perf) > 1e-12:
            _emit(
                sink,
                record.path,
                "performance_nonzero",
                f"ExactLogic performance is {perf!r}, expected 0.0",
            )


def validate_path_device(
    record: LibraryRecord,
    device_folders: dict[str, set[str]],
    sink: ErrorSink,
) -> None:
    """Validate file path against device id and library type conventions.

    Args:
        record: Parsed library record.
        device_folders: Per-architecture device-specific folder ids.
        sink: Error callback.
    """
    arch = record.architecture
    if arch not in DEFAULT_DEVICE_BY_ARCH:
        return
    path = record.path
    folder = path_library_folder(path)
    path_dev = path_device_id(path, arch)
    yaml_devices = record.device_ids
    default_dev = DEFAULT_DEVICE_BY_ARCH.get(arch)
    scoped_ids = device_folders.get(arch, set())

    if not yaml_devices:
        _emit(sink, path, "missing_device", "Library has no DeviceNames entries")
        return

    primary = yaml_devices[0]

    # Device-specific Equality (and similar) libraries belong under gfx*_id<dev>/.
    if (
        folder in DEVICE_SCOPED_LIBRARY_TYPES
        and primary in scoped_ids
        and primary != default_dev
    ):
        expected_segment = f"{arch}_id{primary}"
        if expected_segment not in [p.lower() for p in path.parts]:
            _emit(
                sink,
                path,
                "path_device_folder",
                f"Device {primary} + {folder} should live under {expected_segment}/",
            )
        if path_dev and path_dev != primary:
            _emit(
                sink,
                path,
                "path_device_mismatch",
                f"Path device id {path_dev} does not match primary yaml device {primary}",
            )
        return

    # Generic / default device libraries should not sit under a device-id folder.
    if path_dev and primary == default_dev:
        _emit(
            sink,
            path,
            "path_default_in_id_folder",
            f"Default device {primary} library should not be under {arch}_id{path_dev}",
        )

    # Prediction on gfx950 uses Origami folder (may be nested gfx950/gfx950/Origami).
    if record.library_type == "Prediction" and arch == "gfx950":
        if "Origami" not in path.parts:
            _emit(
                sink,
                path,
                "path_prediction_origami",
                "Prediction libraries for gfx950 should be under an Origami folder",
            )


def validate_filename_consistency(record: LibraryRecord, sink: ErrorSink) -> None:
    """Match filename layouts and datatype token to ProblemType fields.

    Args:
        record: Parsed library record.
        sink: Error callback.
    """
    parsed = parse_filename(record.path.stem)
    if parsed is None:
        _emit(sink, record.path, "filename_parse", "Could not parse library filename")
        return

    if parsed["architecture"] != record.architecture:
        _emit(
            sink,
            record.path,
            "arch_filename_mismatch",
            f"Filename arch {parsed['architecture']} != yaml {record.architecture}",
        )

    pt = record.problem_type
    ta = pt.get("TransposeA")
    tb = pt.get("TransposeB")
    if ta is not None and bool(ta) != parsed["transpose_a"]:
        _emit(
            sink,
            record.path,
            "transpose_a_mismatch",
            f"Filename expects TransposeA={parsed['transpose_a']}, yaml has {ta!r}",
        )
    if tb is not None and bool(tb) != parsed["transpose_b"]:
        _emit(
            sink,
            record.path,
            "transpose_b_mismatch",
            f"Filename expects TransposeB={parsed['transpose_b']}, yaml has {tb!r}",
        )

    types = parsed.get("types")
    if types is None:
        return
    exp_a, exp_b, exp_c, exp_d, exp_comp = types
    checks = [
        ("DataType", exp_a),
        ("DataTypeA", exp_a),
        ("DataTypeB", exp_b),
        ("DestDataType", exp_d),
        ("DataTypeE", exp_d),
        ("ComputeDataType", exp_comp),
    ]
    for field, expected in checks:
        actual = pt.get(field)
        if actual is not None and int(actual) != expected:
            _emit(
                sink,
                record.path,
                "datatype_mismatch",
                f"Filename token {parsed['type_token']}: {field}={actual}, expected {expected}",
            )


def validate_origami_assert_multiples(
    record: LibraryRecord,
    sink: ErrorSink,
    device_folders: Optional[dict[str, set[str]]] = None,
    context: Optional[FileAuditContext] = None,
) -> None:
    """For Origami libraries, explicit assert element multiples must be 1.

    A missing assert key does not fail. Missing keys are filled from
    ``DefaultSolution`` before the comparison.

    Args:
        record: Parsed library record.
        sink: Error callback.
        device_folders: Unused. Present so every check shares one signature.
        context: Unused per-file audit context.

    Returns:
        None.

    Raises:
        No exceptions are raised.
    """
    del device_folders, context
    if not _is_origami_library(record):
        return
    for sol in effective_solutions(record, sink, "origami_assert_multiples"):
        sol_idx = sol.get("SolutionIndex", "?")
        for field_name in ORIGAMI_ASSERT_FIELDS:
            if field_name not in sol:
                continue
            value = sol.get(field_name)
            if value != 1 and value != 1.0:
                _emit(
                    sink,
                    record.path,
                    "origami_assert_multiples",
                    f"{field_name} must be 1 for SolutionIndex {sol_idx} (found {value}).",
                )


def validate_library_type_folder(record: LibraryRecord, sink: ErrorSink) -> None:
    """Library type in YAML should match the parent folder when present.

    Args:
        record: Parsed library record.
        sink: Error callback.
    """
    folder = path_library_folder(record.path)
    if not folder:
        return
    lib_type = record.library_type
    if lib_type == "Matching":
        if folder not in ("GridBased", "Equality", "Range"):
            _emit(
                sink,
                record.path,
                "library_type_folder",
                f"Matching library in unexpected folder {folder}",
            )
        return
    if lib_type != folder and not (lib_type == "Prediction" and folder == "Origami"):
        _emit(
            sink,
            record.path,
            "library_type_folder",
            f"YAML LibraryType {lib_type} does not match folder {folder}",
        )


CheckFn = Callable[
    [LibraryRecord, ErrorSink, Optional[dict[str, set[str]]], Optional[FileAuditContext]],
    None,
]


def _legacy_check(fn: Callable[..., None]) -> CheckFn:
    """Adapt an older validator to the shared check signature.

    Args:
        fn: Callable ``(record, sink)`` or ``(record, device_folders, sink)``.

    Returns:
        Wrapper ``(record, sink, device_folders, context)``.

    Raises:
        No exceptions are raised.
    """

    def wrapper(
        record: LibraryRecord,
        sink: ErrorSink,
        device_folders: Optional[dict[str, set[str]]] = None,
        context: Optional[FileAuditContext] = None,
    ) -> None:
        del context
        names = fn.__code__.co_varnames[: fn.__code__.co_argcount]
        if "device_folders" in names:
            fn(record, device_folders or {}, sink)
        else:
            fn(record, sink)

    return wrapper


# (id, description, function). ``--list-checks`` prints this registry.
# ``--checks`` replaces the set; ``--skip`` removes ids from it.
CHECK_CATALOG: tuple[tuple[str, str, CheckFn], ...] = (
    (
        "list_format_added",
        "Fail when a new file is list-format, or a dict library changes to list-format.",
        validate_list_format_added,
    ),
    (
        "activation_func_call",
        "If ActivationFuncCall is present, it must be false-like (False, 0, '0', 'false').",
        validate_activation_func_call,
    ),
    (
        "streamk_gsu",
        "If StreamK is 3, GlobalSplitU must be 0.",
        validate_streamk_gsu,
    ),
    (
        "use_custom_main_loop_schedule",
        "UseCustomMainLoopSchedule must coerce to 1 (DefaultSolution fills a missing key).",
        validate_use_custom_main_loop_schedule,
    ),
    (
        "performance_zero",
        "ExactLogic performance values must be 0.",
        _legacy_check(validate_performance_zero),
    ),
    (
        "epilogue_consistency",
        "List-format libraries: every solution must share one epilogue key.",
        _legacy_check(validate_epilogue_consistency),
    ),
    (
        "filename_fields",
        "Filename architecture, TransposeA/B, and datatype token must match ProblemType.",
        _legacy_check(validate_filename_consistency),
    ),
    (
        "filename_policy",
        "Filename must contain Cijk_ and the arch token. "
        "Origami implies Prediction; Equality/GridBased/FreeSize must match LibraryType and sizes.",
        validate_filename_policy,
    ),
    (
        "device_names",
        "gfx942, gfx950, and id75a3 filenames must use the expected DeviceNames list.",
        validate_device_names,
    ),
    (
        "path_device",
        "Device-id folder placement for architectures in the default-device map "
        "(other architectures are skipped).",
        _legacy_check(validate_path_device),
    ),
    (
        "library_type_folder",
        "YAML LibraryType must match the parent folder (Prediction may live under Origami).",
        _legacy_check(validate_library_type_folder),
    ),
    (
        "origami_assert_multiples",
        "Origami: AssertFree0/1 and AssertSummation element multiples must be 1 when present.",
        validate_origami_assert_multiples,
    ),
    (
        "kernel_name_matches_filename",
        "KernelNameMin and SolutionNameMin must start with the filename Cijk_ stem.",
        validate_kernel_name_matches_filename,
    ),
    (
        "dict_structure",
        "Dict libraries must include ArchitectureName, ProblemType, Solutions, and LibraryType.",
        validate_dict_structure,
    ),
)

CHECK_FUNCS: dict[str, CheckFn] = {check_id: fn for check_id, _, fn in CHECK_CATALOG}
DEFAULT_CHECK_IDS: tuple[str, ...] = tuple(check_id for check_id, _, _ in CHECK_CATALOG)


def format_check_list() -> str:
    """Format the check registry for ``--list-checks``.

    Args:
        None.

    Returns:
        Multi-line catalog of check id and description.

    Raises:
        No exceptions are raised.
    """
    lines = ["Enabled by default. Toggle with --checks id,id or --skip id,id.", ""]
    for check_id, description, _ in CHECK_CATALOG:
        lines.append(f"{check_id}")
        lines.append(f"    {description}")
    return "\n".join(lines)


def resolve_enabled_checks(
    checks: Optional[list[str]],
    skip: Optional[list[str]],
) -> tuple[str, ...]:
    """Resolve the check ids to run.

    Args:
        checks: Explicit set. When omitted, every default check is used.
        skip: Ids to remove from that set.

    Returns:
        Check ids in catalog order.

    Raises:
        ValueError: If an unknown check id is requested.
    """
    known = set(CHECK_FUNCS)
    selected = list(DEFAULT_CHECK_IDS if not checks else checks)
    unknown = [check_id for check_id in selected if check_id not in known]
    if skip:
        unknown.extend(check_id for check_id in skip if check_id not in known)
    if unknown:
        raise ValueError(
            "Unknown check id(s): "
            + ", ".join(unknown)
            + ". Run --list-checks to see the registry."
        )
    skipped = set(skip or [])
    return tuple(check_id for check_id in selected if check_id not in skipped)


def run_enabled_checks(
    record: LibraryRecord,
    sink: ErrorSink,
    device_folders: Optional[dict[str, set[str]]],
    context: Optional[FileAuditContext],
    enabled: Iterable[str],
) -> None:
    """Run the selected validation rules on one library record.

    Args:
        record: Parsed library.
        sink: Error callback.
        device_folders: Device-specific subfolder map.
        context: Whether the file is a new list-format introduction.
        enabled: Check ids to run.

    Returns:
        None.

    Raises:
        KeyError: If ``enabled`` contains an unknown check id.
    """
    for check_id in enabled:
        CHECK_FUNCS[check_id](record, sink, device_folders, context)


def run_all_validators(
    record: LibraryRecord,
    device_folders: dict[str, set[str]],
    sink: ErrorSink,
) -> None:
    """Run every default validation rule on one library record.

    Args:
        record: Parsed library.
        device_folders: Device-specific subfolder map.
        sink: Error callback.

    Returns:
        None.

    Raises:
        No exceptions are raised.
    """
    run_enabled_checks(record, sink, device_folders, FileAuditContext(), DEFAULT_CHECK_IDS)

# ========================================================================
# audit entrypoint
# ========================================================================

MAX_WORKERS = 32


@dataclass
class GroupStats:
    """Aggregated counters for one statistics group."""

    files: int = 0
    gemm_sizes: int = 0
    kernels: int = 0
    tiles: set[tuple[int, int, int]] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output.

        Returns:
            JSON-serializable dictionary.
        """
        return {
            "files": self.files,
            "gemm_sizes": self.gemm_sizes,
            "kernels": self.kernels,
            "unique_tiles": len(self.tiles),
            "tiles": sorted(self.tiles),
        }


@dataclass
class AuditResult:
    """Full audit output."""

    stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
    parse_failures: list[dict[str, str]] = field(default_factory=list)
    outcomes: list[dict[str, Any]] = field(default_factory=list)
    files_processed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


def _group_key(record: LibraryRecord) -> tuple[str, str, str, str]:
    type_token = ""
    parsed = parse_filename(record.path.stem)
    if parsed:
        type_token = str(parsed.get("type_token", ""))
    device = record.device_ids[0] if record.device_ids else "unknown"
    return (record.architecture, device, type_token, display_library_type(record.library_type))


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=YAML_LOADER)


def _audit_file_worker(
    args: tuple[str, dict[str, list[str]], list[str], bool, Optional[str]],
) -> dict[str, Any]:
    """Process-pool entry point.

    Args:
        args: ``(path_str, device_folders, enabled_checks, added, base_format)``.
            Device folder values are lists (sets are not picklable on all platforms).

    Returns:
        Per-file audit result without non-picklable objects.

    Raises:
        No exceptions are raised. Parse errors are returned in ``parse_failure``.
    """
    path_str, folders_lists, enabled, added, base_format = args
    path = Path(path_str)
    device_folders = {k: set(v) for k, v in folders_lists.items()}

    try:
        data = _load_yaml(path)
        record = normalize_library(data, path)
    except Exception as exc:  # noqa: BLE001
        return {
            "path": path_str,
            "parse_failure": f"{type(exc).__name__}: {exc}",
            "group_key": None,
            "gemm_sizes": 0,
            "kernels": 0,
            "tiles": [],
            "errors": [],
        }

    errors: list[dict[str, str]] = []

    def sink(file_path: str, code: str, message: str) -> None:
        errors.append({"path": file_path, "code": code, "message": message})

    run_enabled_checks(
        record,
        sink,
        device_folders,
        FileAuditContext(added=added, base_format=base_format),
        enabled,
    )
    key = _group_key(record)
    return {
        "path": path_str,
        "parse_failure": None,
        "group_key": list(key),
        "gemm_sizes": sum(1 for _ in iter_gemm_sizes(record.exact_logic)),
        "kernels": len(record.solutions),
        "tiles": [list(t) for t in sorted(solution_tiles(record.solutions))],
        "errors": errors,
    }


def _merge_file_result(
    file_result: dict[str, Any],
    groups: dict[tuple[str, str, str, str], GroupStats],
    result: AuditResult,
) -> None:
    """Merge one file result into aggregate statistics and error lists.

    Args:
        file_result: Output from ``_audit_file_worker`` or cache.
        groups: Mutable group statistics map.
        result: Mutable audit result container.
    """
    outcome = {
        "path": str(file_result.get("path", "")),
        "parse_failure": file_result.get("parse_failure"),
        "errors": list(file_result.get("errors") or []),
    }
    result.outcomes.append(outcome)
    if file_result.get("parse_failure"):
        result.parse_failures.append(
            {"path": file_result["path"], "error": file_result["parse_failure"]}
        )
        return

    result.files_processed += 1
    result.errors.extend(outcome["errors"])
    key_list = file_result.get("group_key")
    if not key_list:
        return
    key = tuple(key_list)
    stats = groups[key]
    stats.files += 1
    stats.kernels += int(file_result.get("kernels", 0))
    stats.gemm_sizes += int(file_result.get("gemm_sizes", 0))
    for tile in file_result.get("tiles", []):
        stats.tiles.add(tuple(tile))


def _collect_yaml_files(
    logic_dir: Path,
    subset: Optional[str],
    limit: Optional[int],
) -> list[Path]:
    """Build the list of YAML files to audit.

    Args:
        logic_dir: hipBLASLt logic library root directory.
        subset: Optional relative subdirectory under the logic root.
        limit: Optional maximum number of files.

    Returns:
        Sorted list of paths to audit.
    """
    if subset:
        root = (logic_dir / subset).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Subset directory not found: {root}")
        yaml_files = sorted(root.rglob("*.yaml"))
    else:
        yaml_files = sorted(logic_dir.rglob("*.yaml"))

    if limit is not None and limit > 0:
        yaml_files = yaml_files[:limit]

    return yaml_files


AUDIT_REPORT_NAME = "audit.txt"


def _display_path(path: str) -> str:
    """Return a library path relative to the working directory when possible.

    Args:
        path: Path stored on an audit outcome.

    Returns:
        A relative path when ``path`` is inside the working directory, otherwise
        ``path`` unchanged.

    Raises:
        No exceptions are raised.
    """
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
        return path


def format_audit_report(result: AuditResult) -> str:
    """Format one CTest-style pass/fail report for the audited libraries.

    Args:
        result: Completed audit result. Each library is one entry in
            ``result.outcomes``.

    Returns:
        Report text ending in a newline. Passing libraries are one line each.
        Failures include the check id and message, then a CTest failure summary.

    Raises:
        No exceptions are raised.
    """
    outcomes = sorted(result.outcomes, key=lambda item: _display_path(str(item.get("path", ""))))
    total = len(outcomes)
    lines: list[str] = []
    failed: list[tuple[int, str]] = []
    for index, outcome in enumerate(outcomes, start=1):
        display = _display_path(str(outcome.get("path", "")))
        details: list[str] = []
        parse_failure = outcome.get("parse_failure")
        if parse_failure:
            details.append(f"    parse_failure: {parse_failure}")
        for err in outcome.get("errors") or []:
            code = err.get("code", "error")
            message = err.get("message", "")
            details.append(f"    {code}: {message}")
        label = f"{index}/{total} Test #{index}: {display}"
        if details:
            lines.append(f"{label} {'.' * max(1, 72 - len(label))}***Failed")
            lines.extend(details)
            failed.append((index, display))
        else:
            lines.append(f"{label} {'.' * max(1, 72 - len(label))}   Passed")

    lines.append("")
    if total == 0:
        lines.append("No libraries audited.")
    elif not failed:
        lines.append(f"All {total} libraries passed.")
    else:
        lines.append("The following libraries FAILED:")
        for index, display in failed:
            lines.append(f"\t{index} - {display} (Failed)")
        lines.append("")
        lines.append(f"{len(failed)} {'library' if len(failed) == 1 else 'libraries'} failed out of {total}")
    return "\n".join(lines) + "\n"


def write_audit_report(result: AuditResult, report_dir: Path) -> str:
    """Write ``audit.txt`` for one audit run.

    Args:
        result: Completed audit result.
        report_dir: Directory that receives ``audit.txt``. Created if missing.

    Returns:
        The report text written to disk.

    Raises:
        OSError: If the report directory or file cannot be written.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    text = format_audit_report(result)
    (report_dir / AUDIT_REPORT_NAME).write_text(text, encoding="utf-8")
    return text


def publish_audit_report(text: str) -> None:
    """Print the audit report and append it to the GitHub job summary.

    The job log keeps the text when the audit step exits non-zero. When
    ``GITHUB_STEP_SUMMARY`` is set, the same text is added to the workflow
    run summary.

    Args:
        text: Report produced by ``format_audit_report``.

    Returns:
        None.

    Raises:
        OSError: If the job summary file cannot be appended.
    """
    print(text, end="" if text.endswith("\n") else "\n")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    body = text if text.endswith("\n") else text + "\n"
    with open(summary_path, "a", encoding="utf-8") as handle:
        handle.write("## hipBLASLt logic library audit\n\n```text\n")
        handle.write(body)
        handle.write("```\n")


def run_audit(
    logic_dir: Path,
    report_dir: Path,
    subset: Optional[str] = None,
    limit: Optional[int] = None,
    workers: int = MAX_WORKERS,
    cache_dir: Optional[Path] = None,
    no_cache: bool = False,
    files: Optional[list[Path]] = None,
    file_contexts: Optional[dict[str, FileAuditContext]] = None,
    enabled_checks: Optional[tuple[str, ...]] = None,
) -> AuditResult:
    """Run the audit over hipBLASLt logic libraries (parallel, with optional cache).

    Args:
        logic_dir: Root directory containing architecture subfolders.
        report_dir: Directory used as the default cache location.
        subset: Optional relative path under the logic root to limit scope.
        limit: Optional max file count (for testing).
        workers: Parallel worker count (capped at 32).
        cache_dir: Cache directory (default: report_dir/.cache).
        no_cache: Disable disk cache.
        files: Explicit YAML paths. When omitted, every YAML file under the logic
            root (or subset) is audited.
        file_contexts: Per-path addition and base-format info, keyed by resolved path.
        enabled_checks: Check ids to run. Defaults to the full registry.

    Returns:
        Completed audit result.

    Raises:
        FileNotFoundError: If subset path does not exist.
    """
    logic_dir = logic_dir.resolve()
    workers = max(1, min(workers, MAX_WORKERS, os.cpu_count() or 1))
    enabled = list(enabled_checks if enabled_checks is not None else DEFAULT_CHECK_IDS)
    contexts = file_contexts or {}
    device_folders = discover_device_folders(logic_dir)
    folders_serializable = {k: sorted(v) for k, v in device_folders.items()}

    yaml_files = list(files) if files is not None else _collect_yaml_files(logic_dir, subset, limit)
    groups: dict[tuple[str, str, str, str], GroupStats] = defaultdict(GroupStats)
    result = AuditResult()

    cache_path = (cache_dir or (report_dir / ".cache")).resolve()
    cache = AuditCache(
        cache_path,
        logic_dir,
        enabled=not no_cache,
        checks_key=",".join(enabled),
    )

    to_process: list[Path] = []
    for path in yaml_files:
        cached = cache.lookup(path)
        if cached is not None:
            _merge_file_result(cached, groups, result)
        else:
            to_process.append(path)

    result.cache_hits = cache.stats["hits"]
    result.cache_misses = cache.stats["misses"]

    if to_process:
        t0 = time.perf_counter()
        print(f"Cache: {result.cache_hits} hits, auditing {len(to_process)} files...", file=sys.stderr)
        job_args = []
        for path in to_process:
            ctx = contexts.get(str(path.resolve()), FileAuditContext())
            job_args.append(
                (str(path), folders_serializable, enabled, ctx.added, ctx.base_format)
            )

        if workers == 1:
            results_by_path = {}
            for a in job_args:
                file_result = _audit_file_worker(a)
                results_by_path[file_result["path"]] = file_result
        else:
            results_by_path = {}
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_audit_file_worker, a): a[0] for a in job_args}
                for future in as_completed(futures):
                    file_result = future.result()
                    results_by_path[file_result["path"]] = file_result

        for path in to_process:
            file_result = results_by_path[str(path)]
            cache.store(path, file_result)
            _merge_file_result(file_result, groups, result)

        elapsed = time.perf_counter() - t0
        print(
            f"Audited {len(to_process)} files in {elapsed:.1f}s "
            f"({workers} workers, {len(to_process) / max(elapsed, 0.001):.1f} files/s)",
            file=sys.stderr,
        )

    cache.save()
    result.cache_hits = cache.stats["hits"]
    result.cache_misses = cache.stats["misses"]

    result.stats = {
        f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": v.to_dict() for k, v in sorted(groups.items())
    }
    return result


def parse_name_status(
    lines: Iterable[str],
    prefix: str = LOGIC_REL_PREFIX,
) -> list[tuple[str, bool]]:
    """Parse ``git diff --name-status`` lines into logic YAML paths.

    Args:
        lines: Diff status lines. Rename lines are tab-separated ``R100 old new``.
        prefix: Repository-relative directory that must contain each path.

    Returns:
        ``(repo_relative_path, added)`` pairs. ``added`` is true for additions
        and rename destinations.

    Raises:
        ValueError: If a selected status line is malformed.
    """
    selected: list[tuple[str, bool]] = []
    for raw in lines:
        line = raw.strip("\n")
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        if status.startswith(("R", "C")):
            if len(parts) < 3:
                raise ValueError(f"Malformed rename status line: {line!r}")
            path = parts[2]
            added = True
        elif status.startswith("A"):
            if len(parts) < 2:
                raise ValueError(f"Malformed add status line: {line!r}")
            path = parts[1]
            added = True
        elif status.startswith("M"):
            if len(parts) < 2:
                raise ValueError(f"Malformed modify status line: {line!r}")
            path = parts[1]
            added = False
        else:
            continue
        norm = path.replace("\\", "/")
        if not norm.startswith(prefix):
            continue
        if not norm.endswith((".yaml", ".yml")):
            continue
        selected.append((norm, added))
    return selected


def git_repo_root(start: Path) -> Path:
    """Return the git repository root that contains ``start``.

    Args:
        start: Directory to run git from.

    Returns:
        Absolute repository root.

    Raises:
        subprocess.CalledProcessError: If ``start`` is not inside a git work tree.
        OSError: If git cannot be executed.
    """
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=start,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(proc.stdout.strip())


def git_changed_logic_files(repo_root: Path, base_ref: str) -> list[tuple[str, bool]]:
    """List logic YAML files added or modified since ``base_ref``.

    Args:
        repo_root: Git repository root.
        base_ref: Base commit, branch, or SHA. The diff is ``base_ref...HEAD``.

    Returns:
        ``(repo_relative_path, added)`` pairs.

    Raises:
        RuntimeError: If git diff fails.
        ValueError: If a status line is malformed.
    """
    proc = subprocess.run(
        [
            "git",
            "diff",
            "--name-status",
            "--diff-filter=AMCR",
            f"{base_ref}...HEAD",
            "--",
            LOGIC_REL_PREFIX,
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or f"git diff failed for {base_ref}"
        raise RuntimeError(detail)
    return parse_name_status(proc.stdout.splitlines())


def yaml_root_format(data: Any) -> str:
    """Classify a parsed YAML document root.

    Args:
        data: Object returned by the YAML loader.

    Returns:
        ``list``, ``dict``, or ``other``.

    Raises:
        No exceptions are raised.
    """
    if isinstance(data, list):
        return "list"
    if isinstance(data, dict):
        return "dict"
    return "other"


def base_yaml_format(repo_root: Path, base_ref: str, rel_path: str) -> Optional[str]:
    """Return the YAML root format of ``rel_path`` at ``base_ref``.

    Args:
        repo_root: Git repository root.
        base_ref: Commit that holds the previous file contents.
        rel_path: Repository-relative path.

    Returns:
        ``list``, ``dict``, ``other``, or None when the blob is missing or unreadable.

    Raises:
        No exceptions are raised.
    """
    proc = subprocess.run(
        ["git", "show", f"{base_ref}:{rel_path}"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    try:
        data = yaml.load(proc.stdout.decode("utf-8", errors="replace"), Loader=YAML_LOADER)
    except yaml.YAMLError:
        return None
    return yaml_root_format(data)


def _split_check_args(values: Optional[list[str]]) -> list[str]:
    """Split comma-separated ``--checks`` / ``--skip`` values.

    Args:
        values: Raw argparse values. Each item may contain comma-separated ids.

    Returns:
        Flat list of check ids.

    Raises:
        No exceptions are raised.
    """
    if not values:
        return []
    ids: list[str] = []
    for value in values:
        ids.extend(part.strip() for part in value.split(",") if part.strip())
    return ids


def _read_path_list(list_path: Path) -> list[str]:
    """Read newline-separated paths from a file.

    Args:
        list_path: Text file of paths.

    Returns:
        Non-empty stripped lines.

    Raises:
        OSError: If the file cannot be read.
    """
    text = list_path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed namespace.

    Raises:
        SystemExit: If argparse rejects the arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hipblaslt-path",
        type=Path,
        default=None,
        help=(
            "Path to rocm-libraries/projects/hipblaslt; "
            "logic directory is derived as library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full"
        ),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reports",
        help="Directory for the audit.txt report",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Relative path under hipBLASLt logic root (e.g. gfx950/gfx950/Equality)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of YAML files to process (for quick tests)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(MAX_WORKERS, os.cpu_count() or 1),
        help=f"Parallel workers (1..{MAX_WORKERS}, default: min(32, cpu_count))",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory (default: <report-dir>/.cache)",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable disk cache")
    parser.add_argument(
        "--list-checks",
        action="store_true",
        help="Print the check registry and exit",
    )
    parser.add_argument(
        "--checks",
        action="append",
        default=None,
        help="Comma-separated check ids to run instead of the default set",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=None,
        help="Comma-separated check ids to remove from the selected set",
    )
    parser.add_argument(
        "--changed-since",
        default=None,
        help="Audit only logic YAML added or modified since this git ref (base...HEAD)",
    )
    parser.add_argument(
        "--files-from",
        type=Path,
        default=None,
        help="Newline-separated YAML paths to audit instead of the whole logic tree",
    )
    parser.add_argument(
        "--added-files-from",
        type=Path,
        default=None,
        help="Newline-separated paths that are additions (list format fails on these)",
    )
    parser.add_argument(
        "--base-ref",
        default=None,
        help="Git ref used to detect dict-to-list changes for modified files",
    )
    return parser.parse_args(argv)


def _selection_from_args(
    args: argparse.Namespace,
    hipblaslt_path: Path,
) -> tuple[Optional[list[Path]], dict[str, FileAuditContext]]:
    """Build the file list and per-file format context from CLI flags.

    Args:
        args: Parsed command-line namespace.
        hipblaslt_path: hipBLASLt project directory, used to find the git root.

    Returns:
        ``(files, contexts)``. ``files`` is None when the whole logic tree should
        be scanned. An empty list means there is nothing to audit.

    Raises:
        ValueError: If ``--changed-since`` and ``--files-from`` are both set.
        RuntimeError: If git diff fails.
        OSError: If a path list cannot be read.
        subprocess.CalledProcessError: If the git root cannot be resolved.
    """
    if args.changed_since and args.files_from:
        raise ValueError("Pass only one of --changed-since or --files-from")

    if args.changed_since:
        repo_root = git_repo_root(hipblaslt_path)
        selected = git_changed_logic_files(repo_root, args.changed_since)
        files: list[Path] = []
        contexts: dict[str, FileAuditContext] = {}
        for rel_path, added in selected:
            path = (repo_root / rel_path).resolve()
            base_format = None
            if not added:
                base_format = base_yaml_format(repo_root, args.changed_since, rel_path)
            files.append(path)
            contexts[str(path)] = FileAuditContext(added=added, base_format=base_format)
        return files, contexts

    if args.files_from is None:
        return None, {}

    repo_root = git_repo_root(hipblaslt_path) if args.base_ref else None
    added_raw = set(_read_path_list(args.added_files_from)) if args.added_files_from else set()
    added_resolved = {str(Path(path).resolve()) for path in added_raw}
    files = []
    contexts = {}
    for raw in _read_path_list(args.files_from):
        path = Path(raw)
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        added = raw in added_raw or str(path) in added_resolved
        base_format = None
        if repo_root is not None and args.base_ref and not added:
            try:
                rel_path = str(path.relative_to(repo_root)).replace("\\", "/")
            except ValueError:
                rel_path = ""
            if rel_path:
                base_format = base_yaml_format(repo_root, args.base_ref, rel_path)
        files.append(path)
        contexts[str(path)] = FileAuditContext(added=added, base_format=base_format)
    return files, contexts


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument list.

    Returns:
        Exit code 0 on success, 1 if errors or parse failures exist, 2 on usage
        or environment errors.

    Raises:
        SystemExit: If argparse rejects the arguments.
    """
    args = parse_args(argv)
    if args.list_checks:
        print(format_check_list())
        return 0

    try:
        enabled = resolve_enabled_checks(_split_check_args(args.checks), _split_check_args(args.skip))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    hipblaslt_path = (
        args.hipblaslt_path.resolve()
        if args.hipblaslt_path is not None
        else default_hipblaslt_path()
    )
    try:
        files, contexts = _selection_from_args(args, hipblaslt_path)
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if files is not None and not files:
        print("No changed logic libraries to audit.")
        return 0

    logic_dir = resolve_logic_dir(args.hipblaslt_path)
    if not logic_dir.is_dir():
        print(f"hipBLASLt logic directory not found: {logic_dir}", file=sys.stderr)
        if args.hipblaslt_path:
            print(f"  (from --hipblaslt-path {args.hipblaslt_path.resolve()})", file=sys.stderr)
        return 2

    result = run_audit(
        logic_dir,
        args.report_dir.resolve(),
        subset=args.subset,
        limit=args.limit,
        workers=args.workers,
        cache_dir=args.cache_dir.resolve() if args.cache_dir else None,
        no_cache=args.no_cache,
        files=files,
        file_contexts=contexts,
        enabled_checks=enabled,
    )
    report_path = args.report_dir.resolve()
    report_text = write_audit_report(result, report_path)
    publish_audit_report(report_text)

    if result.parse_failures or result.errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
