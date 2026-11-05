# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Single-file hipBLASLt logic checks ported from TuningDriver library policy.

These checks run on one library at a time. Diff-only comparisons (base versus
pull-request field equality, removed sizes, benchmark uplift) are not included.
"""

from __future__ import annotations

import re
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

ErrorSink = Callable[[str, str, str], None]

_ASSERT_PARAMS: tuple[str, ...] = (
    "AssertFree0ElementMultiple",
    "AssertFree1ElementMultiple",
    "AssertSummationElementMultiple",
)

_FILENAME_TYPE_RULES: tuple[tuple[str, str], ...] = (
    ("Equality_", "Equality"),
    ("GridBased_", "GridBased"),
    ("FreeSize_", "FreeSize"),
)

def _emit(sink: ErrorSink, path: Path, code: str, message: str) -> None:
    """Record one validation error.

    Args:
        sink: Callback ``(path, code, message)``.
        path: Library file path.
        code: Stable check identifier.
        message: Human-readable failure.

    Returns:
        None.

    Raises:
        No exceptions are raised.
    """
    sink(str(path), code, message)


def _tensilelite_root() -> Path:
    """Return the directory that contains the ``tensilelite`` package.

    Args:
        None.

    Returns:
        ``projects/hipblaslt/tensilelite`` next to this audit package.

    Raises:
        No exceptions are raised.
    """
    return Path(__file__).resolve().parents[2] / "tensilelite"


def _install_rocisa_stub_if_needed() -> None:
    """Install a minimal ``rocisa`` stub when the compiled extension is absent.

    ``GlobalParameters`` imports ``rocisa`` at module load. The audit only needs
    the default-solution table, which does not call into the extension.

    Args:
        None.

    Returns:
        None.

    Raises:
        No exceptions are raised.
    """
    try:
        from rocisa import rocIsa

        rocIsa.getInstance()
        return
    except Exception:
        pass
    rocisa = types.ModuleType("rocisa")

    class _Isa:
        """Stand-in for ``rocisa.rocIsa`` used only to import global defaults."""

        @staticmethod
        def getInstance() -> Any:
            """Return an ISA object with an empty capability map.

            Args:
                None.

            Returns:
                Object whose ``getArchCaps`` is an empty dict.

            Raises:
                No exceptions are raised.
            """
            return types.SimpleNamespace(getArchCaps=lambda: {})

    rocisa.rocIsa = _Isa
    sys.modules["rocisa"] = rocisa


def _install_joblib_stub_if_needed() -> None:
    """Install a minimal ``joblib`` stub when the package is not installed.

    Importing TensileLite loads ``Common.Parallel``, which imports ``joblib``
    at module scope. The audit never calls that parallel helper.

    Args:
        None.

    Returns:
        None.

    Raises:
        No exceptions are raised.
    """
    try:
        import joblib
    except Exception:
        joblib = None
    if joblib is not None and hasattr(joblib, "Parallel") and hasattr(joblib, "delayed"):
        return
    stub = types.ModuleType("joblib")

    def _delayed(function: Callable[..., Any]) -> Callable[..., Any]:
        """Return a function unchanged, standing in for ``joblib.delayed``.

        Args:
            function: Callable that would be wrapped for delayed execution.

        Returns:
            *function* itself.

        Raises:
            No exceptions are raised.
        """
        return function

    stub.Parallel = object
    stub.delayed = _delayed
    stub.__version__ = "0"
    sys.modules["joblib"] = stub


@lru_cache(maxsize=1)
def _tensile_fill_tools() -> tuple[Callable[..., dict[str, Any]], dict[str, Any]]:
    """Load TensileLite's solution-default fill helpers.

    Args:
        None.

    Returns:
        ``normalize_execution_policy_with_defaults`` and ``defaultSolution``.

    Raises:
        ImportError: If TensileLite cannot be imported after the rocisa stub.
    """
    _install_rocisa_stub_if_needed()
    _install_joblib_stub_if_needed()
    root = str(_tensilelite_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from tensilelite.Common.GlobalParameters import defaultSolution
    from tensilelite.ExecutionPolicy import normalize_execution_policy_with_defaults

    return normalize_execution_policy_with_defaults, defaultSolution


def _fill_solution(solution: dict[str, Any], library_defaults: dict[str, Any]) -> dict[str, Any]:
    """Fill one solution the way ``LibraryIO.fillSolutionDefaults`` does.

    File ``DefaultSolution`` is applied first and the execution policy is
    normalized. Keys still missing are then taken from TensileLite's global
    ``defaultSolution``. Explicit solution values win over both.

    Args:
        solution: One solution dictionary from the library.
        library_defaults: The file's ``DefaultSolution``, or an empty map.

    Returns:
        A new solution dictionary with defaults filled.

    Raises:
        ValueError: If TensileLite rejects the execution policy.
    """
    normalize, global_defaults = _tensile_fill_tools()
    state = normalize(dict(solution), library_defaults)
    for key, val in global_defaults.items():
        if key not in state:
            state[key] = val
    return state


def effective_solutions(
    record: Any,
    sink: Optional[ErrorSink] = None,
    code: str = "solution_defaults",
) -> list[dict[str, Any]]:
    """Return solutions filled the way TensileLite reads them back.

    This follows ``LibraryIO.fillSolutionDefaults``: normalize with the file's
    ``DefaultSolution``, then fill any key still missing from the global
    ``defaultSolution``. A solution TensileLite rejects is reported and skipped
    when ``sink`` is set.

    Args:
        record: Parsed library record with ``solutions`` and ``default_solution``.
        sink: Optional error callback for a solution TensileLite rejects.
        code: Check id used when ``sink`` reports a rejected solution.

    Returns:
        Filled solution dictionaries. Rejected solutions are omitted when
        ``sink`` is set.

    Raises:
        ValueError: If a solution is rejected and ``sink`` is None.
    """
    defaults = record.default_solution if isinstance(record.default_solution, dict) else {}
    filled: list[dict[str, Any]] = []
    for sol in record.solutions:
        if not isinstance(sol, dict):
            continue
        try:
            filled.append(_fill_solution(sol, defaults))
        except ValueError as exc:
            if sink is None:
                raise
            sol_idx = sol.get("SolutionIndex", "?")
            _emit(
                sink,
                record.path,
                code,
                f"SolutionIndex {sol_idx}: TensileLite rejected the solution "
                f"while filling defaults: {exc}",
            )
    return filled


def _coerce_int_param(value: Any) -> Optional[int]:
    """Coerce a solution parameter to ``int``.

    Args:
        value: Raw YAML value.

    Returns:
        Integer value, or None when the value cannot be coerced.

    Raises:
        No exceptions are raised.
    """
    if value is None or isinstance(value, bool):
        return int(value) if isinstance(value, bool) else None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            try:
                return int(stripped)
            except ValueError:
                return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _afc_false_like(value: Any) -> bool:
    """Return whether an ActivationFuncCall value is false-like.

    Args:
        value: Raw YAML value.

    Returns:
        True for ``False``, ``0``, ``"0"``, ``"false"``, and ``"False"``.

    Raises:
        No exceptions are raised.
    """
    return value in (False, 0, "0", "false", "False")


def _expected_device_entries(filename: str, architecture: str) -> Optional[list[str]]:
    """Return the DeviceNames list required by a filename or architecture.

    Args:
        filename: Library basename.
        architecture: YAML architecture name.

    Returns:
        Expected ``Device …`` entries, or None when this file has no device rule.

    Raises:
        No exceptions are raised.
    """
    name = filename.lower()
    if re.search(r"id75a3", name):
        return ["Device 75a3", "Device 75a2"]
    if "gfx942" in name or architecture.lower() == "gfx942":
        return ["Device 0049", "Device 0050"]
    if "gfx950" in name or architecture.lower() == "gfx950":
        return ["Device 75a0"]
    return None


_SOLUTION_NAME_FIELDS: tuple[str, ...] = ("KernelNameMin", "SolutionNameMin")


def _cijk_stem_from_filename(filename: str) -> Optional[str]:
    """Return the ``Cijk_`` stem TensileLite wrote into the library filename.

    The logic writer names the file ``{ScheduleName}_{ProblemType}``. That
    problem-type string is also the first component of ``KernelNameMin`` and
    ``SolutionNameMin``. This returns that string as stored in the filename.

    Args:
        filename: Library filename.

    Returns:
        Substring starting at ``Cijk_``, without the YAML suffix, or None.

    Raises:
        No exceptions are raised.
    """
    stem = Path(filename).name
    lower = stem.lower()
    if lower.endswith(".yaml"):
        stem = stem[: -len(".yaml")]
    elif lower.endswith(".yml"):
        stem = stem[: -len(".yml")]
    pos = stem.find("Cijk_")
    if pos < 0:
        return None
    return stem[pos:]


def _name_uses_filename_stem(name: str, stem: str) -> bool:
    """Return whether a TensileLite name starts with the filename problem type.

    Args:
        name: ``KernelNameMin`` or ``SolutionNameMin``.
        stem: ``Cijk_`` stem from the library filename.

    Returns:
        True when ``name`` is the stem or continues after it with ``_``.

    Raises:
        No exceptions are raised.
    """
    return name == stem or name.startswith(stem + "_")


def _is_static_streamk(sol: dict[str, Any]) -> bool:
    """Return whether a filled solution is legacy StreamK mode 3.

    TensileLite drops ``StreamK`` while normalizing and stores mode 3 as
    ``TileProcessingStrategy=StreamK`` with ``WorkAssignment=StaticGrid``.

    Args:
        sol: Solution dictionary after ``fillSolutionDefaults``.

    Returns:
        True for StreamK 3, including its normalized form.

    Raises:
        No exceptions are raised.
    """
    if _coerce_int_param(sol.get("StreamK")) == 3:
        return True
    return (
        sol.get("TileProcessingStrategy") == "StreamK"
        and sol.get("WorkAssignment") == "StaticGrid"
    )


def _check_unused(
    device_folders: Optional[dict[str, set[str]]] = None,
    context: Any = None,
) -> None:
    """Accept the shared check signature arguments.

    Args:
        device_folders: Unused device-folder map.
        context: Unused per-file audit context.

    Returns:
        None.

    Raises:
        No exceptions are raised.
    """
    del device_folders, context


def validate_streamk_gsu(
    record: Any,
    sink: ErrorSink,
    device_folders: Optional[dict[str, set[str]]] = None,
    context: Any = None,
) -> None:
    """Require GlobalSplitU 0 when StreamK is 3.

    The solution is filled with TensileLite defaults first. StreamK 3 is then
    ``TileProcessingStrategy=StreamK`` and ``WorkAssignment=StaticGrid``. An
    omitted GlobalSplitU is the global default, 1.

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
    _check_unused(device_folders, context)
    for sol in effective_solutions(record, sink, "streamk_gsu"):
        sol_idx = sol.get("SolutionIndex", "?")
        if not _is_static_streamk(sol):
            continue
        if _coerce_int_param(sol.get("GlobalSplitU")) != 0:
            _emit(
                sink,
                record.path,
                "streamk_gsu",
                f"SolutionIndex {sol_idx}: when StreamK is 3, GlobalSplitU must be 0 "
                f"(found GlobalSplitU={sol.get('GlobalSplitU')!r}).",
            )


def validate_use_custom_main_loop_schedule(
    record: Any,
    sink: ErrorSink,
    device_folders: Optional[dict[str, set[str]]] = None,
    context: Any = None,
) -> None:
    """Require UseCustomMainLoopSchedule to coerce to 1.

    The value is the one TensileLite fills: the file ``DefaultSolution`` first,
    otherwise the global default ``-1``. ``-1`` fails this check.

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
    _check_unused(device_folders, context)
    for sol in effective_solutions(record, sink, "use_custom_main_loop_schedule"):
        sol_idx = sol.get("SolutionIndex", "?")
        if _coerce_int_param(sol.get("UseCustomMainLoopSchedule")) == 1:
            continue
        _emit(
            sink,
            record.path,
            "use_custom_main_loop_schedule",
            f"SolutionIndex {sol_idx}: UseCustomMainLoopSchedule must be 1 "
            f"(found {sol.get('UseCustomMainLoopSchedule')!r}).",
        )


def validate_device_names(
    record: Any,
    sink: ErrorSink,
    device_folders: Optional[dict[str, set[str]]] = None,
    context: Any = None,
) -> None:
    """Require DeviceNames lists implied by gfx942, gfx950, or id75a3.

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
    _check_unused(device_folders, context)
    filename = Path(record.path).name
    expected = _expected_device_entries(filename, str(record.architecture))
    if expected is None:
        return
    actual = record.device_names_raw
    if actual is None:
        _emit(
            sink,
            record.path,
            "device_names",
            f"Library root missing device list (DeviceNames) for {filename!r}.",
        )
        return
    if not isinstance(actual, list):
        _emit(
            sink,
            record.path,
            "device_names",
            f"Device field (DeviceNames) must be a list for {filename!r}; "
            f"found {type(actual).__name__}.",
        )
        return
    got = [entry.strip() if isinstance(entry, str) else str(entry).strip() for entry in actual]
    if sorted(got) != sorted(expected):
        _emit(
            sink,
            record.path,
            "device_names",
            f"The yaml file has the wrong device id (expected {expected!r}, found {got!r}).",
        )


def validate_dict_structure(
    record: Any,
    sink: ErrorSink,
    device_folders: Optional[dict[str, set[str]]] = None,
    context: Any = None,
) -> None:
    """Require the dict-format root keys used by policy validation.

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
    _check_unused(device_folders, context)
    if record.format != "dict":
        return
    missing = [
        key
        for key, present in (
            ("ArchitectureName", bool(record.architecture)),
            ("ProblemType", isinstance(record.problem_type, dict) and bool(record.problem_type)),
            ("Solutions", isinstance(record.solutions, list)),
            ("LibraryType", bool(record.library_type)),
        )
        if not present
    ]
    if missing:
        _emit(
            sink,
            record.path,
            "dict_structure",
            "Invalid library format for policy validation (structure incomplete): "
            + ", ".join(missing)
            + ".",
        )


def validate_filename_policy(
    record: Any,
    sink: ErrorSink,
    device_folders: Optional[dict[str, set[str]]] = None,
    context: Any = None,
) -> None:
    """Check filename tokens, library type, and Equality/GridBased size rules.

    The ``Cijk_`` problem-type string is the one TensileLite already wrote into
    the filename. This check does not build another name from ProblemType.

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
    _check_unused(device_folders, context)
    basename = Path(record.path).name
    if "Cijk_" not in basename:
        _emit(sink, record.path, "filename_policy", "Filename must contain 'Cijk_' orientation token.")
        return
    if not (basename.endswith("UserArgs.yaml") or basename.endswith(".yaml")):
        _emit(
            sink,
            record.path,
            "filename_policy",
            "Filename must end with 'UserArgs.yaml' (or '.yaml' for legacy files).",
        )

    if not isinstance(record.problem_type, dict) or not record.problem_type:
        _emit(sink, record.path, "filename_policy", "Problem section is missing or not a dict.")
        return

    arch = str(record.architecture or "").strip()
    if arch and arch not in basename:
        _emit(sink, record.path, "filename_policy", f"Filename must contain arch token '{arch}'.")

    lib_type = record.library_type
    if "Origami" in basename and lib_type != "Prediction":
        _emit(
            sink,
            record.path,
            "filename_policy",
            f"Origami library must have Type 'Prediction' (found '{lib_type}').",
        )

    for path_marker, expected_type in _FILENAME_TYPE_RULES:
        if path_marker in basename and lib_type != expected_type:
            _emit(
                sink,
                record.path,
                "filename_policy",
                f"Library path contains '{path_marker.rstrip('_')}' but Type is "
                f"'{lib_type}' (expected '{expected_type}').",
            )

    sizes = record.exact_logic
    has_sizes = isinstance(sizes, list) and len(sizes) > 0
    if ("Equality_" in basename or "GridBased_" in basename) and not has_sizes:
        _emit(
            sink,
            record.path,
            "filename_policy",
            "Equality/GridBased library must have non-empty size mappings.",
        )
    if ("Equality_" in basename or "GridBased_" in basename) and lib_type == "Prediction":
        _emit(
            sink,
            record.path,
            "filename_policy",
            "Equality/GridBased library must not have Type 'Prediction'.",
        )


def validate_kernel_name_matches_filename(
    record: Any,
    sink: ErrorSink,
    device_folders: Optional[dict[str, set[str]]] = None,
    context: Any = None,
) -> None:
    """Require KernelNameMin and SolutionNameMin to use the filename stem.

    Both names are produced by TensileLite ``Naming._getName``, whose first
    component is the same problem-type string written into the filename.
    ``BaseName`` is not consulted.

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
    _check_unused(device_folders, context)
    fname = Path(record.path).name
    stem = _cijk_stem_from_filename(fname)
    if stem is None:
        return
    for sol in effective_solutions(record, sink, "kernel_name_matches_filename"):
        sol_idx = sol.get("SolutionIndex", "?")
        for field in _SOLUTION_NAME_FIELDS:
            raw = sol.get(field)
            if isinstance(raw, str) and _name_uses_filename_stem(raw, stem):
                continue
            _emit(
                sink,
                record.path,
                "kernel_name_matches_filename",
                f"SolutionIndex {sol_idx}: {field} must start with the filename "
                f"problem-type stem {stem!r} (found {raw!r}).",
            )
