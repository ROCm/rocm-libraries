# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Single-file hipBLASLt logic checks ported from TuningDriver library policy.

These checks run on one library at a time. Diff-only comparisons (base versus
pull-request field equality, removed sizes, benchmark uplift) are not included.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Optional

ErrorSink = Callable[[str, str, str], None]

_ASSERT_PARAMS: tuple[str, ...] = (
    "AssertFree0ElementMultiple",
    "AssertFree1ElementMultiple",
    "AssertSummationElementMultiple",
)

_TYPE_INDEX_TO_SHORT_NAME: dict[int, str] = {0: "S", 4: "H", 7: "B"}

_TYPE_INDEX_TO_NAME_TEMP: dict[int, tuple[str, str]] = {
    0: ("f32_r", "f32_r"),
    1: ("f64_r", "f64_r"),
    2: ("f32_c", "f32_c"),
    3: ("f64_c", "f64_c"),
    4: ("f16_r", "f16_r"),
    5: ("i8", "i8"),
    6: ("i32", "i32"),
    7: ("bf16_r", "bf16_r"),
    8: ("i8_r", "i8_r"),
    10: ("f8_r", "f8_r"),
    11: ("bf8_r", "bf8_r"),
    12: ("f8_r", "bf8_r"),
    13: ("bf8_r", "f8_r"),
    15: ("f8_r", "f8_r"),
    16: ("bf8_r", "bf8_r"),
    17: ("f8_r", "bf8_r"),
    18: ("bf8_r", "f8_r"),
    21: ("f4_r", "f4_r"),
    22: ("e8m0_r", "e8m0_r"),
}

_TYPE_TO_GEMM_LETTER: dict[str, str] = {
    "f64_c": "Z",
    "f32_c": "C",
    "f64_r": "D",
    "f32_r": "S",
    "xf32_r": "S_MX",
    "f16_r": "H",
    "bf16_r": "B",
    "i32_r": "I",
    "i8_r": "I8",
    "f8_r": "F8",
    "bf8_r": "B8",
    "f4_r": "F4",
    "": "",
}

_FILENAME_TYPE_RULES: tuple[tuple[str, str], ...] = (
    ("Equality_", "Equality"),
    ("GridBased_", "GridBased"),
    ("FreeSize_", "FreeSize"),
)

_COLLAPSE_BIAS_THROUGH_HAS = re.compile(r"_Bias(.*?)_HAS_")
_TILE_MT_MI_RE = re.compile(r"_MT[0-9x]+_MI[0-9x]+")


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


def effective_solutions(record: Any) -> list[dict[str, Any]]:
    """Return solution dicts with missing keys filled from ``DefaultSolution``.

    List-format libraries have no ``DefaultSolution``, so their solution dicts
    are returned unchanged. Explicit solution values override defaults.

    Args:
        record: Parsed library record with ``solutions`` and ``default_solution``.

    Returns:
        Solution dictionaries used by solution-field checks.

    Raises:
        No exceptions are raised.
    """
    defaults = record.default_solution if isinstance(record.default_solution, dict) else {}
    filled: list[dict[str, Any]] = []
    for sol in record.solutions:
        if not isinstance(sol, dict):
            continue
        if not defaults:
            filled.append(sol)
            continue
        merged = dict(defaults)
        merged.update(sol)
        filled.append(merged)
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


def _encode_gemm_name(
    type_a: str,
    type_b: str,
    type_c: str,
    type_d: str,
    type_compute: str,
    type_input_a: str = "",
    type_input_b: str = "",
) -> str:
    """Encode datatypes into a GEMM filename token.

    Args:
        type_a: MFMA A datatype string.
        type_b: MFMA B datatype string.
        type_c: Output C datatype string.
        type_d: Output D datatype string.
        type_compute: Compute datatype string, optionally prefixed with ``c_``.
        type_input_a: Optional input A cast datatype.
        type_input_b: Optional input B cast datatype.

    Returns:
        GEMM name token such as ``HHS`` or ``F8B8F8S``.

    Raises:
        KeyError: If a datatype string is not in the letter table.
    """
    name_to_type = {value: key for key, value in _TYPE_TO_GEMM_LETTER.items() if value}
    if (
        type_a == type_b == type_c == type_d == name_to_type.get("S")
        and type_compute == "c_" + name_to_type["S_MX"]
    ):
        return "S_MX"

    compute = type_compute.replace("c_", "")
    if type_a == type_b == type_c == type_d == compute:
        return _TYPE_TO_GEMM_LETTER[type_a] + "GEMM"

    if type_a == type_b:
        type_b = ""
    if type_c == type_d:
        type_d = ""

    gemm_name = ""
    if type_input_a and type_input_b:
        gemm_name += _TYPE_TO_GEMM_LETTER[type_input_a]
        gemm_name += _TYPE_TO_GEMM_LETTER[type_input_b]
        gemm_name += "_"

    for dtype in (type_a, type_b, type_c, type_d, compute):
        if dtype:
            gemm_name += _TYPE_TO_GEMM_LETTER[dtype]
    return gemm_name


def _compute_type_from_problem(problem: dict[str, Any]) -> str:
    """Derive the compute datatype token used in GEMM filenames.

    Args:
        problem: ProblemType dictionary.

    Returns:
        Compute type string such as ``c_f32_r``.

    Raises:
        No exceptions are raised.
    """
    if "ComputeDataType" in problem:
        compute_idx = problem["ComputeDataType"]
        if compute_idx not in _TYPE_INDEX_TO_NAME_TEMP:
            return "c_f32_r"
        comp_type = _TYPE_INDEX_TO_NAME_TEMP[compute_idx][0]
        if problem.get("F32XdlMathOp") == 10:
            comp_type = "x" + comp_type
        return "c_" + comp_type

    data_idx = problem.get("DataType", 0)
    if data_idx not in _TYPE_INDEX_TO_NAME_TEMP:
        return "c_f32_r"
    a_type = _TYPE_INDEX_TO_NAME_TEMP[data_idx][0]
    if a_type == "f16_r" and problem.get("HighPrecisionAccumulate"):
        return "c_f32_r"
    if data_idx == 5:
        return "i32_r"
    return "c_" + a_type


def _expected_orientation_token(problem: dict[str, Any]) -> str:
    """Derive the ``Cijk_`` orientation token from TransposeA/B.

    Args:
        problem: ProblemType dictionary.

    Returns:
        Expected orientation substring such as ``Cijk_Ailk_Bjlk``.

    Raises:
        KeyError: If a transpose key is missing.
    """
    trans_a = problem["TransposeA"]
    trans_b = problem["TransposeB"]
    if not trans_a and not trans_b:
        return "Cijk_Ailk_Bljk"
    if not trans_a and trans_b:
        return "Cijk_Ailk_Bjlk"
    if trans_a and not trans_b:
        return "Cijk_Alik_Bljk"
    return "Cijk_Alik_Bjlk"


def _expected_name_suffix_from_problem(problem: dict[str, Any]) -> str:
    """Build the expected filename suffix from the problem section.

    Args:
        problem: ProblemType dictionary.

    Returns:
        Expected suffix starting with ``Cijk_`` through epilogue tokens.

    Raises:
        KeyError: If a required problem key or datatype index is missing.
        ValueError: If a datatype index cannot be encoded.
    """
    out_name = _expected_orientation_token(problem) + "_"

    input_a_type = _TYPE_INDEX_TO_NAME_TEMP[problem["DataTypeA"]][0]
    input_b_type = _TYPE_INDEX_TO_NAME_TEMP[problem["DataTypeB"]][0]
    a_type, b_type = _TYPE_INDEX_TO_NAME_TEMP[problem["DataType"]]
    _, d_type = _TYPE_INDEX_TO_NAME_TEMP[problem["DestDataType"]]
    c_type = d_type
    comp_type = _compute_type_from_problem(problem)

    if input_a_type == a_type and input_b_type == b_type:
        gemm_name = _encode_gemm_name(a_type, b_type, c_type, d_type, comp_type)
    else:
        gemm_name = _encode_gemm_name(
            a_type, b_type, c_type, d_type, comp_type, input_a_type, input_b_type
        )

    if "GEMM" in gemm_name:
        gemm_name = gemm_name.replace("GEMM", "")
        out_name += gemm_name
        if problem.get("Batched"):
            out_name += "B"
        out_name += "_"
    else:
        out_name += gemm_name + "_"
        if problem.get("Batched"):
            out_name += "B"
        if problem.get("HighPrecisionAccumulate"):
            out_name += "H"
        out_name += "_"

    if problem.get("UseBias") == 1:
        out_name += "Bias"
        for bias in problem.get("BiasDataTypeList", []):
            out_name += _TYPE_INDEX_TO_SHORT_NAME[bias]
        out_name += "_"

    if problem.get("UseE") == 1:
        out_name += "Aux_"

    bias_src = problem.get("BiasSrc")
    if bias_src == "A":
        out_name += "BiasSrcA_"
    elif bias_src == "B":
        out_name += "BiasSrcB_"

    if problem.get("GroupedGemm") == 1:
        out_name += "GG_"

    if problem.get("Gradient") == 1:
        out_name += "Grad_"

    if problem.get("Activation") == 1 and problem.get("ActivationComputeDataType") == 0:
        out_name += "HAS_"

    use_scale_ab = problem.get("UseScaleAB")
    if use_scale_ab == "Scalar":
        out_name += "SAB_"
    elif use_scale_ab == "Vector":
        out_name += "SABV_"

    if problem.get("UseScaleAlphaVec") == 1:
        out_name += "SAV_"

    return out_name


def _filename_matches_expected_suffix(expected_suffix: str, basename: str) -> bool:
    """Return whether ``basename`` contains the expected datatype suffix.

    Args:
        expected_suffix: Suffix derived from ProblemType.
        basename: Library filename.

    Returns:
        True when the suffix, or the batched SGEMM ``_S_B_`` alias, is present.

    Raises:
        No exceptions are raised.
    """
    if expected_suffix in basename:
        return True
    alt = expected_suffix.replace("_SB_", "_S_B_")
    return alt != expected_suffix and alt in basename


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


def _strip_yaml_suffix(basename: str) -> str:
    """Remove a trailing ``.yaml`` or ``.yml`` suffix.

    Args:
        basename: Filename.

    Returns:
        Basename without the YAML suffix.

    Raises:
        No exceptions are raised.
    """
    lower = basename.lower()
    if lower.endswith(".yaml"):
        return basename[: -len(".yaml")]
    if lower.endswith(".yml"):
        return basename[: -len(".yml")]
    return basename


def _extract_cijk_stem_from_filename(filename: str) -> Optional[str]:
    """Return the ``Cijk_`` stem from a library basename.

    Args:
        filename: Library filename.

    Returns:
        Substring starting at ``Cijk_``, or None when it is absent.

    Raises:
        No exceptions are raised.
    """
    stem = _strip_yaml_suffix(Path(filename).name)
    pos = stem.find("Cijk_")
    if pos < 0:
        return None
    return stem[pos:]


def _normalize_userargs_suffix(stem: str) -> str:
    """Strip a trailing ``_UserArgs`` token.

    Args:
        stem: Filename or kernel stem.

    Returns:
        Stem without ``_UserArgs``.

    Raises:
        No exceptions are raised.
    """
    if stem.endswith("_UserArgs"):
        return stem[: -len("_UserArgs")]
    return stem


def _stem_norm_epilogue(stem: str) -> str:
    """Normalize a ``Cijk_`` stem for epilogue comparison.

    Args:
        stem: Logic stem.

    Returns:
        Stem with ``_UserArgs`` removed and ``_HA_S_`` mapped to ``_HAS_``.

    Raises:
        No exceptions are raised.
    """
    return _normalize_userargs_suffix(stem).replace("_HA_S_", "_HAS_")


def _logic_stem_for_solution_filename_compare(stem: str) -> str:
    """Collapse bias subtypes before ``_HAS_`` for filename comparison.

    Args:
        stem: Logic stem.

    Returns:
        Comparison key.

    Raises:
        No exceptions are raised.
    """
    normalized = _stem_norm_epilogue(stem)
    return _COLLAPSE_BIAS_THROUGH_HAS.sub("_Bias_HAS_", normalized)


def _cijk_logic_prefix_from_kernel_name(name: Optional[str]) -> Optional[str]:
    """Return the logic prefix from the first ``Cijk_`` through ``_MT`` or ``_MI``.

    Args:
        name: Kernel or solution name.

    Returns:
        Logic prefix, or None when ``Cijk_`` is absent.

    Raises:
        No exceptions are raised.
    """
    if not isinstance(name, str):
        return None
    pos = name.find("Cijk_")
    if pos < 0:
        return None
    tail = name[pos:]
    cut = len(tail)
    for marker in ("_MT", "_MI"):
        marker_pos = tail.find(marker)
        if marker_pos >= 0:
            cut = min(cut, marker_pos)
    return tail[:cut]


def _cijk_logic_stem_from_tensile_basename_field(name: Optional[str]) -> Optional[str]:
    """Build a filename-comparable ``Cijk_`` stem from Tensile ``BaseName``.

    Args:
        name: BaseName field.

    Returns:
        Comparable stem, or None when ``Cijk_`` is absent.

    Raises:
        No exceptions are raised.
    """
    if not isinstance(name, str) or "Cijk_" not in name:
        return None
    pos = name.find("Cijk_")
    tail = name[pos:]
    match = _TILE_MT_MI_RE.search(tail)
    if not match:
        return tail
    head = tail[: match.start()]
    after_tile = tail[match.end() :]
    lower = after_tile.lower()
    for marker in ("_shortname", "_gfx"):
        idx = lower.find(marker)
        if idx >= 0:
            after_tile = after_tile[:idx]
            break
    if "_Bias" in head or "_HAS_" in head:
        return head
    if after_tile:
        return head + after_tile
    return head


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
    for sol in effective_solutions(record):
        sol_idx = sol.get("SolutionIndex", "?")
        if _coerce_int_param(sol.get("StreamK")) != 3:
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

    Missing keys are filled from ``DefaultSolution`` before the comparison.
    A value that is still missing fails, matching TuningDriver.

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
    for sol in effective_solutions(record):
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

    problem = record.problem_type
    if not isinstance(problem, dict) or not problem:
        _emit(sink, record.path, "filename_policy", "Problem section is missing or not a dict.")
        return

    arch = str(record.architecture or "").strip()
    if arch and arch not in basename:
        _emit(sink, record.path, "filename_policy", f"Filename must contain arch token '{arch}'.")

    try:
        orientation = _expected_orientation_token(problem)
    except KeyError as exc:
        _emit(sink, record.path, "filename_policy", f"Problem section missing transpose key: {exc}.")
        return
    if orientation not in basename:
        _emit(
            sink,
            record.path,
            "filename_policy",
            f"Filename orientation mismatch: expected '{orientation}' from TransposeA/B.",
        )

    try:
        expected_suffix = _expected_name_suffix_from_problem(problem)
    except (KeyError, ValueError) as exc:
        _emit(
            sink,
            record.path,
            "filename_policy",
            f"Could not derive expected filename suffix from problem section: {exc}.",
        )
        return
    if not _filename_matches_expected_suffix(expected_suffix, basename):
        _emit(
            sink,
            record.path,
            "filename_policy",
            "Filename datatype/epilogue mismatch: expected suffix containing "
            f"'{expected_suffix.rstrip('_')}' derived from problem section.",
        )

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
    """Require each solution ``Cijk_`` stem to match the YAML basename stem.

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
    stem_file = _extract_cijk_stem_from_filename(fname)
    if stem_file is None:
        return
    file_cmp = _logic_stem_for_solution_filename_compare(stem_file)
    for sol in effective_solutions(record):
        sol_idx = sol.get("SolutionIndex", "?")
        raw = sol.get("KernelNameMin")
        if not isinstance(raw, str) or "Cijk_" not in raw:
            _emit(
                sink,
                record.path,
                "kernel_name_matches_filename",
                f"SolutionIndex {sol_idx}: KernelNameMin missing or invalid "
                f"(expected a string containing 'Cijk_') in {fname!r}.",
            )
            continue
        prefix = _cijk_logic_prefix_from_kernel_name(raw)
        if prefix is None:
            _emit(
                sink,
                record.path,
                "kernel_name_matches_filename",
                f"SolutionIndex {sol_idx}: could not derive 'Cijk_' logic prefix "
                f"from KernelNameMin in {fname!r}.",
            )
            continue
        sol_cmp = _logic_stem_for_solution_filename_compare(prefix)
        if sol_cmp == file_cmp:
            continue
        base_raw = _cijk_logic_stem_from_tensile_basename_field(sol.get("BaseName"))
        base_cmp = (
            _logic_stem_for_solution_filename_compare(base_raw)
            if isinstance(base_raw, str)
            else None
        )
        if base_cmp == file_cmp:
            continue
        _emit(
            sink,
            record.path,
            "kernel_name_matches_filename",
            f"SolutionIndex {sol_idx}: logic stem from KernelNameMin (and BaseName fallback) "
            f"does not match the YAML basename 'Cijk_' stem in {fname!r}. "
            f"Compare-key basename: {file_cmp!r}; compare-key from KernelNameMin: {sol_cmp!r}.",
        )
