################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

"""Default-codegen schedule -> CMS schedule converter.

Public API: :func:`default_schedule_to_cms`. CLI:
``python -m Tensile.Components.CustomSchedule.cms_from_default``.

Implements bead ``rocm-libraries-wlrp`` Phase 2 per
``Tensile/Components/DEFAULT_TO_CMS_CONVERTER_PHASE1.md`` §8 RESOLVED
decisions and the §3.3/§5.4/§6.2/§6.3 corrections.

Design highlights (don't remove without re-reading the memo):

- Single YAML format: default Tensile (``GlobalParameters`` +
  ``BenchmarkProblems[0][0..1]`` shape) — see ``example.yaml``.
- Multi-Solution rejection: if ``ForkParameters`` expands to >1 Solution,
  the converter raises naming the fork dimensions. No ``solution_index``.
- ``TileConfig.isa`` is required; converter passes the YAML's ISA through
  explicitly (or honors an ``isa`` override). No fallback default.
- Schedule-collision check: scans ``_SCHEDULE_REGISTRY`` for matching
  ``(TileConfig, dtype_predicate, vector_widths, matrix_inst,
  mfma_wave_group)`` and refuses to write unless ``force=True`` /
  ``--overwrite`` is passed.
- ``optSchedule`` keys must be inserted in *first-appearance* order across
  the linear stream — the dispatcher emits per-mfma-slot in
  ``dict.items()`` insertion order. See memo §2.3.
- DTL-aware ``nglshift`` / ``nllshift`` derivation (see
  :func:`_derive_nglshift_nllshift`).
- Soft-fail: ``TimingTooCloseFailure`` always becomes a docstring warning
  on the emitted file; everything else hard-fails.
"""

from __future__ import annotations

import argparse
import io
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from rocisa.instruction import SBarrier, SNop, SWaitCnt

from . import _SCHEDULE_REGISTRY, RegisterSchedule, ScheduleInfo, TileConfig
from .shared import _DTYPE_PREDICATE_NAMES, is16bit, is8bit, isTF32


# ----------------------------------------------------------------------------
# Public surface
# ----------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Categorized output of :func:`validate_schedule_against_default`.

    Soft-fail filtering happens inside the converter:
    ``TimingTooCloseFailure`` instances appear in
    ``timing_only_failures`` and are *also* surfaced as a docstring
    warning on the emitted file. Everything else (other graph_diff
    types, wait_coverage, structural) is a hard failure that the
    converter raises on.
    """
    structural: list = field(default_factory=list)
    graph_diff: list = field(default_factory=list)
    wait_coverage: list = field(default_factory=list)
    timing_only_failures: list = field(default_factory=list)


def default_schedule_to_cms(
    yaml_path: Path,
    output_path: Path,
    *,
    schedule_name: str,
    isa: Optional[Tuple[int, int, int]] = None,
    force: bool = False,
) -> ValidationReport:
    """Convert a default-codegen schedule into a registered CMS schedule.

    Always soft-fails ``TimingTooCloseFailure`` (warning written to the
    emitted file's docstring); hard-fails all other validator failures.

    Reads the input YAML as the default Tensile format. Raises if the
    YAML expands to more than one Solution; the user must narrow
    ``ForkParameters`` to a single config. Raises if the YAML lacks an
    ISA and ``isa`` is None. Refuses to write if a registered schedule
    already covers the same ``(TileConfig, dtype, ...)`` tuple, unless
    ``force=True``.
    """
    yaml_path = Path(yaml_path)
    output_path = Path(output_path)

    config_dict = _load_default_yaml(yaml_path)
    solution_config = _resolve_single_solution_config(config_dict)
    isa_tuple = _resolve_isa(solution_config, isa)
    solution_config["ISA"] = isa_tuple

    # Build & drive ONE kernel build with capture enabled.
    solution, writer = _build_writer_and_solution(solution_config)
    writer.enable_capture_default_schedule()
    writer._getKernelSource(solution)

    capture = writer._last_default_capture
    if capture is None or not capture.main_loop:
        raise RuntimeError(
            f"Default-side capture was not produced for {yaml_path}. "
            f"This usually means UseCustomMainLoopSchedule did not stay "
            f"enabled or the kernel build failed earlier."
        )

    # Resolve tile/dtype/vector_widths/matrix_inst/mfma_wave_group from kernel.
    kernel = solution
    tile_config = _build_tile_config(kernel, isa_tuple)
    dtype_predicate = _resolve_dtype_predicate(kernel)
    vector_widths = [
        kernel["GlobalReadVectorWidthA"],
        kernel["GlobalReadVectorWidthB"],
        kernel["LocalReadVectorWidth"],
    ]
    matrix_inst = list(kernel["MatrixInstruction"])[:4]
    mfma_wave_group = list(kernel["MIWaveGroup"])

    # Collision check (memo §5.5).
    if not force:
        _check_collision(tile_config, dtype_predicate, vector_widths,
                         matrix_inst, mfma_wave_group)

    # Build the ScheduleInfo from the captured linear stream.
    schedule_info = _build_schedule_info_from_capture(capture, tile_config)

    # Validate against the default capture (already produced); also runs the
    # CMS-side validator. Both ctx.default and ctx.cms exist if kernelBody
    # ran the comparison; we re-run compare_graphs / validate_edge_wait_coverage
    # with our soft-fail filter.
    report = validate_schedule_against_default(writer, capture)

    # Emit Python source.
    docstring_warnings = []
    for fail in report.timing_only_failures:
        docstring_warnings.append(
            f"TimingTooCloseFailure (soft-fail): {fail.format()}"
        )
    body = _emit_schedule_python(
        schedule_name=schedule_name,
        isa_tuple=isa_tuple,
        tile_config=tile_config,
        dtype_predicate=dtype_predicate,
        vector_widths=vector_widths,
        matrix_inst=matrix_inst,
        mfma_wave_group=mfma_wave_group,
        schedule_info=schedule_info,
        kernel=kernel,
        source_yaml=yaml_path,
        docstring_warnings=docstring_warnings,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(body)
    return report


def validate_schedule_against_default(writer, default_capture) -> ValidationReport:
    """Thin in-process wrapper over compare_graphs + validate_edge_wait_coverage.

    The caller has already driven a kernel build with
    ``KernelWriter.enable_capture_default_schedule()`` engaged, so both
    ``writer._last_default_capture`` and ``writer._last_cms_capture`` are
    populated. We re-run the validator pair with the soft-fail filter
    applied (``TimingTooCloseFailure`` -> warning, everything else -> raise).

    The CMS-side path inside ``customMainLoopSchedule`` already raised on
    structural issues via ``cmsv.isValid``; reaching this function means
    structural validation passed. We capture an empty ``structural`` list
    here for symmetry — if a future caller wants to surface structural
    failures via this report instead of an exception, the field exists.
    """
    from Tensile.Components.CMSValidator import (
        TimingTooCloseFailure,
        build_dataflow_graph,
        compare_graphs,
        validate_edge_wait_coverage,
    )

    cms_capture = writer._last_cms_capture
    if cms_capture is None:
        raise RuntimeError(
            "CMS-side capture was not produced. The deferred CMS expansion "
            "in KernelWriter.kernelBody only runs when a CMS schedule was "
            "matched at customMainLoopSchedule time. Did you supply a YAML "
            "that selects UseCustomMainLoopSchedule=1?"
        )

    ref_graph = build_dataflow_graph(default_capture)
    subj_graph = build_dataflow_graph(cms_capture)
    graph_failures = list(compare_graphs(ref_graph, subj_graph))
    wait_failures = list(validate_edge_wait_coverage(subj_graph))
    # NOTE: VOPD pair-formation (RDNA3.5 §7.6 R-4..R-7) is NOT re-run
    # here. The single canonical dispatch surface for that pass is
    # `cmsv.isValid` (CMSValidator.py), invoked in
    # `CustomSchedule/dispatch.py` over the same `cms_capture` before
    # this function is ever reached. Wiring it here too would create a
    # parallel-API duplicate; if `isValid` raised on a VOPD violation,
    # control would never reach this wrapper.

    timing_only = [f for f in graph_failures if isinstance(f, TimingTooCloseFailure)]
    hard_graph = [f for f in graph_failures if not isinstance(f, TimingTooCloseFailure)]

    if hard_graph:
        raise RuntimeError(
            "Converter hard-failed on graph comparison "
            f"({len(hard_graph)} non-timing failure(s)):\n  "
            + "\n  ".join(f.format() for f in hard_graph)
        )
    if wait_failures:
        raise RuntimeError(
            "Converter hard-failed on wait-coverage validation "
            f"({len(wait_failures)} failure(s)):\n  "
            + "\n  ".join(f.format() for f in wait_failures)
        )

    return ValidationReport(
        structural=[],
        graph_diff=graph_failures,
        wait_coverage=wait_failures,
        timing_only_failures=timing_only,
    )


# ----------------------------------------------------------------------------
# YAML loader (default Tensile format only).
# ----------------------------------------------------------------------------

def _load_default_yaml(yaml_path: Path) -> dict:
    """Load a default-format Tensile YAML.

    Accepts only the canonical shape used by ``example.yaml``: top-level
    ``GlobalParameters`` and ``BenchmarkProblems`` keys, with
    ``BenchmarkProblems[0]`` being a ``[ProblemType, ProblemSizeGroup]``
    pair. The alternate (``--alternate-format``) shape is explicitly
    rejected; users must normalize via Tensile's ``Tensile()`` rewrite
    path first (see memo §4).
    """
    from Tensile import LibraryIO

    if not yaml_path.exists():
        raise FileNotFoundError(f"Input YAML not found: {yaml_path}")

    data = LibraryIO.read(str(yaml_path))
    if not isinstance(data, dict):
        raise ValueError(
            f"YAML at {yaml_path} did not parse to a dict (got "
            f"{type(data).__name__}). The converter accepts only the "
            f"default Tensile format (GlobalParameters + BenchmarkProblems)."
        )
    if "BenchmarkProblems" not in data:
        # Detect the alternate shape (top-level ProblemType / ForkParameters).
        if "ProblemType" in data or "ForkParameters" in data:
            raise ValueError(
                f"YAML at {yaml_path} appears to use the alternate Tensile "
                "format (top-level ProblemType / ForkParameters). The "
                "converter accepts only the default format. Normalize via "
                "Tensile's `Tensile()` rewrite path or wrap your config in "
                "GlobalParameters + BenchmarkProblems."
            )
        raise ValueError(
            f"YAML at {yaml_path} is missing a top-level 'BenchmarkProblems' "
            "key. The converter accepts only the default Tensile format."
        )
    return data


def _resolve_single_solution_config(config_dict: dict) -> dict:
    """Walk the config dict to a single solution config.

    Refuses inputs whose ``ForkParameters`` expand to more than one
    Solution — names the fork dimensions per the user's directive in
    memo §8 decision 4.
    """
    from copy import deepcopy

    bp = config_dict["BenchmarkProblems"]
    if not bp or not isinstance(bp, list):
        raise ValueError("'BenchmarkProblems' must be a non-empty list.")
    if len(bp) != 1:
        raise ValueError(
            f"'BenchmarkProblems' has {len(bp)} entries; converter accepts "
            "exactly one. Narrow your YAML to a single benchmark problem."
        )

    pair = bp[0]
    if not isinstance(pair, list) or len(pair) < 2:
        raise ValueError(
            "BenchmarkProblems[0] must be a [ProblemType, "
            "BenchmarkProblemSizeGroup] pair."
        )
    problem_type_dict = pair[0]
    psg = pair[1]

    if not isinstance(psg, dict):
        raise ValueError("BenchmarkProblems[0][1] must be a dict.")

    # ForkParameters / BenchmarkCommonParameters are lists of single-key
    # dicts (Tensile YAML idiom). Flatten them.
    common = _flatten_named_lists(psg.get("BenchmarkCommonParameters") or [])
    fork = _flatten_named_lists(psg.get("ForkParameters") or [])

    # Detect multi-Solution fanout. Each fork value-list of length >1
    # produces a fanout dimension.
    multi_dims = [(k, len(v)) for k, v in fork.items()
                  if isinstance(v, list) and len(v) > 1]
    if multi_dims:
        names = ", ".join(f"{k}[{n}]" for k, n in multi_dims)
        raise ValueError(
            f"Input YAML expands to multiple Solutions (fork dimensions: "
            f"{names}). Narrow ForkParameters to a single config and re-run."
        )

    # Materialize the single-Solution dict: each fork param contributes its
    # only value; common params contribute their value directly.
    solution_config: dict = {}
    for k, v in common.items():
        # BenchmarkCommonParameters uses the same list-of-one-value shape as
        # ForkParameters in the example.yaml fixture
        # (e.g. KernelLanguage: ["Assembly"]).
        solution_config[k] = v[0] if isinstance(v, list) and len(v) == 1 else v
    for k, v in fork.items():
        solution_config[k] = v[0] if isinstance(v, list) and len(v) == 1 else v

    solution_config["ProblemType"] = deepcopy(problem_type_dict)
    return solution_config


def _flatten_named_lists(items) -> dict:
    """Flatten a list-of-single-key-dicts into a single dict.

    Tensile YAML uses ``- Name: [val1, val2, ...]`` for both
    ``ForkParameters`` and ``BenchmarkCommonParameters``. Each item is a
    single-key dict; flatten into ``{Name: [...]}``.
    """
    out = {}
    if not items:
        return out
    if isinstance(items, dict):
        return dict(items)
    for entry in items:
        if entry is None:
            continue
        if not isinstance(entry, dict):
            raise ValueError(
                f"Expected a dict in fork/common parameter list, got "
                f"{type(entry).__name__}: {entry!r}"
            )
        for k, v in entry.items():
            out[k] = v
    return out


def _resolve_isa(solution_config: dict, isa_override) -> Tuple[int, int, int]:
    """Resolve the ISA tuple from the YAML or the explicit override.

    Order of precedence: explicit ``isa_override`` (CLI / API kwarg) wins;
    otherwise look in ``solution_config['ISA']``. Raises if neither is
    present (no fallback to gfx950 — memo §5.4).
    """
    if isa_override is not None:
        if not (isinstance(isa_override, tuple) and len(isa_override) == 3
                and all(isinstance(x, int) for x in isa_override)):
            raise ValueError(
                f"isa override must be a 3-tuple of ints (maj, min, patch); "
                f"got {isa_override!r}"
            )
        return tuple(isa_override)

    yaml_isa = solution_config.get("ISA")
    if yaml_isa is None:
        raise ValueError(
            "No ISA available: the YAML did not set 'ISA' and no `--isa` "
            "override was passed. The converter does not fall back to a "
            "default ISA — pass `--isa <maj.min.patch>` or add an ISA key "
            "to the YAML."
        )
    if not (isinstance(yaml_isa, (list, tuple)) and len(yaml_isa) == 3):
        raise ValueError(f"YAML 'ISA' must be a 3-element list/tuple; got {yaml_isa!r}")
    return tuple(int(x) for x in yaml_isa)


# ----------------------------------------------------------------------------
# Solution / writer plumbing.
# ----------------------------------------------------------------------------

def _build_writer_and_solution(solution_config: dict):
    """Build a Solution and a fresh KernelWriterAssembly for the kernel build.

    Mirrors :func:`Tests/unit/cms_test_utils._make_solution` but resolves
    ``isaInfoMap`` and the assembler from the system here so the converter
    is callable as a library function without any test-fixture machinery.
    """
    import shutil
    from copy import deepcopy

    from Tensile.Common import IsaVersion
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.SolutionStructs.Problem import ProblemType
    from Tensile.SolutionStructs.Solution import Solution
    from Tensile.TensileLogic.HandleCustomKernel import (
        matrixInstructionToMIParameters,
    )
    from Tensile.Toolchain.Component import Assembler
    from Tensile.KernelWriterAssembly import DebugConfig, KernelWriterAssembly

    isa_tuple = solution_config["ISA"]
    isa = IsaVersion(*isa_tuple)
    compiler = shutil.which("amdclang++") or shutil.which("clang++")
    assembler_bin = shutil.which("amdclang") or shutil.which("clang")
    if compiler is None or assembler_bin is None:
        raise RuntimeError(
            "amdclang/amdclang++ (or clang/clang++) must be available on "
            "PATH to probe ISA capabilities for the converter's kernel build."
        )
    isaInfoMap = makeIsaInfoMap([isa], compiler)
    asm = Assembler(assembler_bin, "V5")

    # ProblemType normalization
    pt_config = solution_config.get("ProblemType", {})
    if isinstance(pt_config, dict):
        pt = ProblemType(pt_config, False)
        config = dict(solution_config)
        config["ProblemType"] = deepcopy(pt.state)
    else:
        config = dict(solution_config)

    config["ISA"] = isa
    config.setdefault("KernelLanguage", "Assembly")
    config.setdefault("WavefrontSize", 64)
    config.setdefault("WorkGroup", [32, 8, 1])

    mi = config.get("MatrixInstruction", [])
    if isinstance(mi, list) and len(mi) == 9:
        mi_params = matrixInstructionToMIParameters(
            mi, isa, config["WavefrontSize"],
            config["ProblemType"], config["WorkGroup"], isaInfoMap,
        )
        config.update(mi_params)

    sol = Solution(
        config, splitGSU=False,
        printSolutionRejectionReason=True,
        printIndexAssignmentInfo=False,
        assembler=asm, isaInfoMap=isaInfoMap,
    )
    if not sol["Valid"]:
        raise RuntimeError(
            f"Solution is not valid for the YAML config. Reason should "
            f"appear above (PrintSolutionRejectionReason=True). Config: "
            f"{config!r}"
        )

    writer = KernelWriterAssembly(asm, DebugConfig())
    return sol, writer


# ----------------------------------------------------------------------------
# TileConfig + dtype + vector-widths + collision.
# ----------------------------------------------------------------------------

def _build_tile_config(kernel, isa_tuple) -> TileConfig:
    return TileConfig(
        macro_tile_size_0=kernel["MacroTile0"],
        macro_tile_size_1=kernel["MacroTile1"],
        depth_u=kernel["DepthU"],
        prefetch_global_read=kernel["PrefetchGlobalRead"],
        prefetch_local_read=kernel["PrefetchLocalRead"],
        direct_to_lds=kernel["DirectToLds"],
        dtl_plus_lds_buf=bool(kernel.get("DtlPlusLdsBuf", False)),
        wave_separate_global_read_a=kernel["WaveSeparateGlobalReadA"],
        wave_separate_global_read_b=kernel["WaveSeparateGlobalReadB"],
        isa=tuple(isa_tuple),
        wavefront_size=kernel.get("WavefrontSize", 64),
    )


def _resolve_dtype_predicate(kernel):
    """Pick the right dtype predicate (is16bit / is8bit / isTF32)."""
    if kernel.get("UseF32XEmulation"):
        return isTF32
    pt = kernel["ProblemType"]
    dt = pt["DataType"]
    if dt.isHalf() or dt.isBFloat16():
        return is16bit
    if dt.isInt8() or dt.is8bitFloat():
        return is8bit
    raise ValueError(
        f"Could not resolve a CMS dtype predicate for DataType={dt!r}. "
        "Supported predicates: is16bit, is8bit, isTF32."
    )


def _check_collision(tile_config, dtype_predicate, vector_widths,
                     matrix_inst, mfma_wave_group) -> None:
    """Refuse to emit if a registered schedule already covers the target tuple.

    Memo §5.5: the dispatcher returns the first match in
    ``_SCHEDULE_REGISTRY`` (import order). A converter-emitted schedule
    that collides would either silently shadow the existing schedule
    (if imported first) or be silently dead code (if imported later).
    Refuse explicitly; ``force=True`` / ``--overwrite`` bypasses.
    """
    for wrapped in _SCHEDULE_REGISTRY:
        # The wrapped function captures the decorator's tile_config etc. via
        # closure on `self`. We probe via the freevars of the closure.
        cells = {
            name: cell.cell_contents
            for name, cell in zip(wrapped.__code__.co_freevars,
                                  wrapped.__closure__ or [])
        }
        existing = cells.get("self")
        if existing is None or not isinstance(existing, RegisterSchedule):
            continue
        if (existing.tile_config == tile_config
                and existing.dtype_predicate is dtype_predicate
                and existing.vector_widths == vector_widths
                and existing.matrix_inst == matrix_inst
                and existing.mfma_wave_group == mfma_wave_group):
            raise RuntimeError(
                "Schedule collision: a registered CMS schedule already "
                "covers this (TileConfig, dtype, vector_widths, "
                "matrix_inst, mfma_wave_group) tuple. Pass `--overwrite` "
                "(CLI) or `force=True` (API) to bypass. "
                f"Tile: {tile_config}; dtype_predicate: "
                f"{getattr(existing.dtype_predicate, '__name__', existing.dtype_predicate)}."
            )


# ----------------------------------------------------------------------------
# Linear-stream -> ScheduleInfo.
# ----------------------------------------------------------------------------

def _build_schedule_info_from_capture(capture, tile_config) -> ScheduleInfo:
    """Translate a default-side ``FourPartCapture`` into a CMS ``ScheduleInfo``.

    Walks ``capture.main_loop[0].instructions`` (a ``list[TaggedInstruction]``)
    in emission order and buckets each TaggedInstruction by its ``category``.
    Inserts dict keys in *first-appearance* order (memo §2.3 emission-fidelity
    invariant — the CMS dispatcher emits per-mfma-slot in
    ``dict.items()`` insertion order).

    Slot positions come from ``ti.slot.mfma_index``. SYNC/SNOP categories
    populate parallel ``syncCode`` / ``snopCode`` lists in zip order with
    their position lists. ``numMfma`` counts MFMA-tagged instructions.
    ``numCodePaths`` is fixed to 1 (memo E1).

    DTL-aware nglshift/nllshift derivation in
    :func:`_derive_nglshift_nllshift`.
    """
    instructions = capture.main_loop[0].instructions
    if not instructions:
        raise RuntimeError("Captured main_loop[0] has no instructions.")

    optSchedule: dict = {}
    syncCode: list = []
    snopCode: list = []
    numMfma = 0

    for ti in instructions:
        cat = ti.category
        slot_idx = ti.slot.mfma_index
        if cat == "MFMA":
            numMfma += 1
            continue
        # First appearance establishes insertion order.
        positions = optSchedule.setdefault(cat, [[]])[0]
        positions.append(slot_idx)
        if cat == "SYNC":
            syncCode.append(ti.wrapped.rocisa_inst)
        elif cat == "SNOP":
            snopCode.append(ti.wrapped.rocisa_inst)

    nglshift, nllshift = _derive_nglshift_nllshift(optSchedule, tile_config)

    return ScheduleInfo(
        numCodePaths=1,
        numMfma=numMfma,
        optSchedule=optSchedule,
        syncCode=syncCode,
        nglshift=nglshift,
        nllshift=nllshift,
        snopCode=snopCode,
    )


def _derive_nglshift_nllshift(optSchedule, tile_config) -> Tuple[int, int]:
    """Derive ``nglshift`` / ``nllshift`` per memo §3.3 (DTL-aware).

    DTL (``direct_to_lds == 1``): every other entry in GRA[0] / GRB[0] is a
    pointer-increment instruction (not a load). Halve the combined entry
    count to get the load-pair count. This matches every gfx950 DTL=1
    schedule (verified at Phase 2 implementation time: all 38
    ``Tensile/Components/CustomSchedule/gfx950/_*.py`` files have
    ``direct_to_lds == 1`` and use either an integer literal that matches
    ``(len(GRA[0]) + len(GRB[0])) // 2`` or write that exact expression
    inline).

    Non-DTL (``direct_to_lds == 0``): not currently exercised by any
    registered schedule (the entire gfx950 registry is DTL=1 post-bkub).
    The non-DTL branch is implemented as ``len(GRA[0]) + len(GRB[0])`` —
    the half-formula's underlying logic without the doubling correction —
    on the principle that non-DTL GRA/GRB position lists contain one entry
    per load instruction (no pointer-increment doubling). This branch will
    fire the first time a non-DTL CMS schedule is converted; if the
    resulting CMS validator finds the value wrong, the fix is local to
    this function.
    """
    gra = optSchedule.get("GRA", [[]])[0]
    grb = optSchedule.get("GRB", [[]])[0]
    total = len(gra) + len(grb)
    if tile_config.direct_to_lds == 1:
        return total // 2, total // 2
    return total, total


# ----------------------------------------------------------------------------
# Python-source emission.
# ----------------------------------------------------------------------------

def _emit_schedule_python(*, schedule_name, isa_tuple, tile_config,
                          dtype_predicate, vector_widths, matrix_inst,
                          mfma_wave_group, schedule_info, kernel,
                          source_yaml, docstring_warnings) -> str:
    """Render the Python source for the registered schedule module.

    Uses minimal explicit imports (no wildcards). Layout filter is the
    most-permissive: the converter targets exactly the layout the input
    YAML describes, so we hard-code the matching predicate name
    (``isTN`` / ``isNN`` / ``isNT`` / ``isTT``).
    """
    layout_pred = _resolve_layout_predicate(kernel)
    dtype_pred_name = dtype_predicate.__name__
    sched_inst_imports = _instruction_imports(schedule_info)

    buf = io.StringIO()
    buf.write(_LICENSE_HEADER)
    buf.write('\n"""')
    buf.write(f"Default-codegen-derived CMS schedule {schedule_name}.\n\n")
    buf.write(f"Generated from: {source_yaml.name}\n")
    buf.write(f"Tile: {tile_config.macro_tile_size_0}x"
              f"{tile_config.macro_tile_size_1}x{tile_config.depth_u}, "
              f"ISA gfx{isa_tuple[0]}{isa_tuple[1]}{isa_tuple[2]}, "
              f"dtype_predicate={dtype_pred_name}\n")
    if docstring_warnings:
        buf.write("\nValidator warnings (soft-fail; converter still "
                  "emitted this schedule):\n")
        for w in docstring_warnings:
            buf.write("  " + w + "\n")
    buf.write('"""\n\n')

    if sched_inst_imports:
        buf.write(f"from rocisa.instruction import {', '.join(sched_inst_imports)}\n")
    buf.write("from ..dispatch import RegisterSchedule\n")
    buf.write("from ..shared import (\n")
    buf.write("    ScheduleInfo,\n")
    buf.write("    TileConfig,\n")
    buf.write(f"    {dtype_pred_name},\n")
    buf.write(f"    {layout_pred},\n")
    buf.write(")\n\n\n")

    # Decorator
    buf.write("@RegisterSchedule(\n")
    buf.write(
        "    tile_config=TileConfig("
        f"{tile_config.macro_tile_size_0}, {tile_config.macro_tile_size_1}, "
        f"{tile_config.depth_u}, {tile_config.prefetch_global_read}, "
        f"{tile_config.prefetch_local_read}, {tile_config.direct_to_lds}, "
        f"{tile_config.dtl_plus_lds_buf}, "
        f"{tile_config.wave_separate_global_read_a}, "
        f"{tile_config.wave_separate_global_read_b}, "
        f"isa={isa_tuple!r}, wavefront_size={tile_config.wavefront_size}),\n"
    )
    buf.write(f"    dtype_predicate={dtype_pred_name},\n")
    buf.write(f"    vector_widths={vector_widths!r},\n")
    buf.write(f"    matrix_inst={matrix_inst!r},\n")
    buf.write(f"    mfma_wave_group={mfma_wave_group!r},\n")
    buf.write(")\n")

    # Function body
    buf.write(f"def {schedule_name}(kernel, useLDSTr, TLDS):\n")
    buf.write(f'    """Generated CMS schedule for {tile_config.macro_tile_size_0}x'
              f'{tile_config.macro_tile_size_1}x{tile_config.depth_u}."""\n')
    use_lds_tr = bool(kernel.get("LDSTrInst", False))
    tlds = int(kernel["TransposeLDS"])
    buf.write(f"    if {layout_pred}(kernel) and useLDSTr == {use_lds_tr} "
              f"and TLDS == {tlds}:\n")
    # Emit optSchedule literal.
    _emit_opt_schedule_literal(buf, schedule_info.optSchedule, indent="        ")
    # Emit syncCode literal.
    if schedule_info.syncCode:
        buf.write("        syncCode = [\n")
        for inst in schedule_info.syncCode:
            buf.write("            " + _render_inst_literal(inst) + ",\n")
        buf.write("        ]\n")
    else:
        buf.write("        syncCode = []\n")
    # Emit snopCode literal.
    if schedule_info.snopCode:
        buf.write("        snopCode = [\n")
        for inst in schedule_info.snopCode:
            buf.write("            " + _render_inst_literal(inst) + ",\n")
        buf.write("        ]\n")
    else:
        buf.write("        snopCode = []\n")
    buf.write(f"        nglshift = {schedule_info.nglshift}\n")
    buf.write(f"        nllshift = {schedule_info.nllshift}\n")
    buf.write(f"        opt1 = ScheduleInfo({schedule_info.numCodePaths}, "
              f"{schedule_info.numMfma}, optSchedule, syncCode, "
              "nglshift, nllshift, snopCode=snopCode)\n")
    buf.write("        return True, opt1\n")
    buf.write("    return False, None\n")
    return buf.getvalue()


def _emit_opt_schedule_literal(buf, opt_schedule: dict, indent: str) -> None:
    buf.write(f"{indent}optSchedule = {{\n")
    for k, vlist in opt_schedule.items():
        # vlist is a list-of-lists (codepaths). Always a single codepath here.
        buf.write(f"{indent}    {k!r}: {vlist!r},\n")
    buf.write(f"{indent}}}\n")


def _instruction_imports(schedule_info: ScheduleInfo) -> list:
    """Compute the minimal rocisa instruction-class imports needed."""
    out = set()
    for inst in schedule_info.syncCode:
        if isinstance(inst, SBarrier):
            out.add("SBarrier")
        elif isinstance(inst, SWaitCnt):
            out.add("SWaitCnt")
    for inst in schedule_info.snopCode:
        if isinstance(inst, SNop):
            out.add("SNop")
    return sorted(out)


def _render_inst_literal(inst) -> str:
    """Render a rocisa SWaitCnt / SBarrier / SNop as a Python constructor call."""
    if isinstance(inst, SBarrier):
        return f"SBarrier(comment={(inst.comment or '')!r})"
    if isinstance(inst, SWaitCnt):
        return (
            f"SWaitCnt(dscnt={inst.dscnt}, vlcnt={inst.vlcnt}, "
            f"vscnt={inst.vscnt}, comment={(inst.comment or '')!r})"
        )
    if isinstance(inst, SNop):
        # SNop's first positional arg is its waitState; recover via getParams.
        params = inst.getParams()
        wait_state = int(params[0]) if len(params) > 0 else 0
        return f"SNop({wait_state}, comment={(inst.comment or '')!r})"
    raise TypeError(
        f"Don't know how to render instruction of type {type(inst).__name__}; "
        "extend _render_inst_literal."
    )


def _resolve_layout_predicate(kernel) -> str:
    ta = bool(kernel["ProblemType"]["TransposeA"])
    tb = bool(kernel["ProblemType"]["TransposeB"])
    return {
        (False, False): "isNN",
        (False, True): "isNT",
        (True, False): "isTN",
        (True, True): "isTT",
    }[(ta, tb)]


# ----------------------------------------------------------------------------
# License + boilerplate.
# ----------------------------------------------------------------------------

_LICENSE_HEADER = textwrap.dedent("""\
    ################################################################################
    #
    # Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
    #
    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
    # ies of the Software, and to permit persons to whom the Software is furnished
    # to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
    # PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
    # FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
    # COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
    # IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
    # CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    ################################################################################
""")


# ----------------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------------

def _parse_isa_string(s: str) -> Tuple[int, int, int]:
    parts = s.split(".")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--isa must be 'maj.min.patch' (e.g. '9.5.0'), got {s!r}"
        )
    try:
        return tuple(int(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--isa parts must be integers, got {s!r} ({e})"
        )


def main(argv=None):
    """CLI entry: convert a Tensile YAML into a registered CMS schedule.

    Accepts only the default Tensile YAML format
    (``GlobalParameters`` + ``BenchmarkProblems``). The alternate
    (``--alternate-format``) shape is out of scope for the converter —
    normalize via Tensile's own rewrite path before invoking.
    """
    parser = argparse.ArgumentParser(
        prog="python -m Tensile.Components.CustomSchedule.cms_from_default",
        description=(
            "Convert a default-codegen Tensile YAML into a registered CMS "
            "schedule. Reads ONLY the default Tensile YAML format "
            "(GlobalParameters + BenchmarkProblems). The alt-format "
            "(--alternate-format) shape is NOT supported; normalize first."
        ),
    )
    parser.add_argument("input_yaml", type=Path, help="Input Tensile YAML.")
    parser.add_argument("output_py", type=Path,
                        help="Output Python file with the @RegisterSchedule body.")
    parser.add_argument("--schedule-name", required=True,
                        help="Name of the generated schedule function.")
    parser.add_argument("--isa", type=_parse_isa_string, default=None,
                        help="ISA override 'maj.min.patch'; falls back to YAML's ISA.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Bypass the schedule-collision check.")
    args = parser.parse_args(argv)

    report = default_schedule_to_cms(
        yaml_path=args.input_yaml,
        output_path=args.output_py,
        schedule_name=args.schedule_name,
        isa=args.isa,
        force=args.overwrite,
    )

    if report.timing_only_failures:
        print(
            f"Soft-failed on {len(report.timing_only_failures)} "
            "TimingTooCloseFailure(s); warnings written to the emitted "
            "file's docstring.",
            file=sys.stderr,
        )
    print(f"Wrote: {args.output_py}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
