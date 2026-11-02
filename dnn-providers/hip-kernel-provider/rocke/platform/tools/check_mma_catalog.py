#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# check_mma_catalog.py -- report every way the MMA catalog disagrees with clang.
#
# The tiling layer's MMA catalog (python/rocke/helpers/tiling/traits/data/
# mma_traits.json) is maintained by hand. Four of its facts are the compiler's to
# decide, and this tool asks the installed clang about each of them:
#
#   targets        the backend targets on which a row's builtin compiles and
#                  links (`supported_targets`)
#   intrinsic      the LLVM intrinsic the builtin lowers to (`llvm_intrinsic`)
#   scale formats  the scale formats a scaled builtin's scale-format operands
#                  accept (`scale.a_dtypes` / `scale.b_dtypes`)
#   completeness   every matrix builtin that compiles on a backend target has
#                  a row
#
# It is a manual tool, not a gate: run it after editing the catalog and whenever
# the compiler, LLVM flavor or backend target list changes (README "Editing the
# catalog"). Run it with the newest clang available; an older clang cannot
# judge the targets and builtins it does not know, and the report says which.
#
# The targets it asks about are rocke's backend targets
# (`core.arch.known_arches()`), the set the catalog lists from (README
# "Targets"). Each builtin is probed by compiling a C call to it and linking
# the result with the same compile+link as tools/check_ir_validity.py: only the
# link proves the intrinsic exists for the target. Scale formats are probed on
# the first listed target the row compiles on.
#
# The builtin prototypes come from the clang being asked (its AST dump), so no
# type table is kept here. The list of all matrix builtins comes from the
# BuiltinsAMDGPU table installed beside clang; a clang installed without it
# cannot answer the completeness question, and the report says so.
#
# Usage:
#   python rocke/platform/tools/check_mma_catalog.py
#   ROCKE_LLVM_BIN=<llvm>/bin python rocke/platform/tools/check_mma_catalog.py --verbose

from __future__ import annotations

import argparse
import concurrent.futures as cf
import re
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from _hostcaps import available_cpus
from _llvm_identity import flavor_of_clang, tool_identity

HERE = Path(__file__).resolve().parent
ROCKE = HERE.parent  # tools -> rocke/platform


def _bootstrap_sys_path() -> None:
    """Import rocke from a checkout without an external PYTHONPATH."""
    path = ROCKE / "python"
    if (path / "rocke").is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


_bootstrap_sys_path()

# rocke may come from the checkout the bootstrap above put on the path.
from check_ir_validity import (
    _compile,
    _first_diagnostic,
    _llvm_tool,
    _target_unsupported,
)
from rocke.core.arch import compiler_target_for_flavor, known_arches
from rocke.core.arch.mfma_scale import MATRIX_FORMATS
from rocke.helpers.tiling.traits import (
    DEFAULT_TRAITS_PATH,
    MmaTraits,
    load_mma_traits,
)

# Matrix builtins: dense and scaled MFMA, sparse SMFMAC, dense and scaled WMMA,
# sparse SWMMAC.
MATRIX_BUILTIN = re.compile(r"__builtin_amdgcn_(?:mfma|smfmac|wmma|swmmac)_\w+")

# The scale-format code a gfx1250 scaled WMMA takes per operand (LLVM's
# matrix_{a,b}_scale_fmt immediates).
SCALE_FORMATS = {"e8m0": 0, "e5m3": 1, "e4m3": 2}


@dataclass(frozen=True)
class FormatOperands:
    """Which call operands of one builtin are format immediates.

    Positions count the builtin's operands from 0. ``matrix`` holds the A and B
    format operands of an ``f8f6f4`` builtin (codes from
    ``core.arch.mfma_scale.MATRIX_FORMATS``), ``scale`` the A and B scale-format
    operands of a gfx1250 scaled WMMA (codes from ``SCALE_FORMATS``). The
    positions follow the operand order of the LLVM intrinsic each builtin maps
    to (IntrinsicsAMDGPU.td).
    """

    matrix: tuple[int, int] | None = None
    scale: tuple[int, int] | None = None


# Checked in order; the first pattern that matches the whole builtin name wins.
_FORMAT_OPERANDS: tuple[tuple[re.Pattern[str], FormatOperands], ...] = (
    # (A, B, C, cbsz, blgp, op_sel_a, scale_a, op_sel_b, scale_b)
    (
        re.compile(r"__builtin_amdgcn_mfma_scale_f32_\w+_f8f6f4"),
        FormatOperands(matrix=(3, 4)),
    ),
    # (fmt_a, A, fmt_b, B, mods, C, sel_a, scale_fmt_a, scale_a, sel_b,
    #  scale_fmt_b, scale_b, reuse_a, reuse_b)
    (
        re.compile(r"__builtin_amdgcn_wmma_scale(?:16)?_f32_\w+_f8f6f4"),
        FormatOperands(matrix=(0, 2), scale=(7, 10)),
    ),
    # (A, B, mods, C, sel_a, scale_fmt_a, scale_a, sel_b, scale_fmt_b, scale_b,
    #  reuse_a, reuse_b)
    (
        re.compile(r"__builtin_amdgcn_wmma_scale(?:16)?_f32_\w+_f4"),
        FormatOperands(scale=(5, 8)),
    ),
    # (fmt_a, A, fmt_b, B, mods, C)
    (
        re.compile(r"__builtin_amdgcn_wmma_f32_\w+_f8f6f4"),
        FormatOperands(matrix=(0, 2)),
    ),
)


def format_operands(builtin: str) -> FormatOperands:
    """The format immediates of ``builtin`` (none for most builtins)."""
    for pattern, operands in _FORMAT_OPERANDS:
        if pattern.fullmatch(builtin):
            return operands
    return FormatOperands()


# ---------------------------------------------------------------------------
# Intrinsic names
# ---------------------------------------------------------------------------

# One LLVM type in a declare line: a fixed vector first, so that the element
# type inside `<8 x float>` is not also read as a scalar `float`.
_IR_TYPE = re.compile(r"<\s*(\d+)\s*x\s*(\w+)\s*>|\b(i\d+|half|bfloat|float|double)\b")
_SCALAR_MANGLING = {"half": "f16", "bfloat": "bf16", "float": "f32", "double": "f64"}
_DECLARED_NAME = re.compile(r"@([\w.$-]+)\s*\(")


def _mangle_scalar(name: str) -> str:
    return _SCALAR_MANGLING.get(name, name)


def intrinsic_base_name(declare_line: str) -> str:
    """The intrinsic a ``declare`` line declares, without its type-overload suffix.

    An overloaded intrinsic's name ends in the LLVM mangling of the types it is
    instantiated at (``.v16f32.v4f16``). Those trailing parts are removed: each
    part that is the mangling of a type in the declare's own signature. A part
    is never removed for spelling a type the signature does not hold, so
    ``llvm.amdgcn.wmma.f32.16x16x4.f32.v8f32.v2f32`` keeps its ``f32`` (the
    signature holds only ``<8 x float>``, ``<2 x float>``, ``i1`` and ``i16``).
    """
    if not declare_line.lstrip().startswith("declare"):
        raise ValueError(f"not an LLVM declare line: {declare_line!r}")
    match = _DECLARED_NAME.search(declare_line)
    if match is None:
        raise ValueError(f"declare line names no function: {declare_line!r}")
    signature = declare_line[: match.start()] + declare_line[match.end() :]
    manglings = {
        f"v{m[1]}{_mangle_scalar(m[2])}" if m[1] else _mangle_scalar(m[3])
        for m in _IR_TYPE.finditer(signature)
    }
    parts = match.group(1).split(".")
    while len(parts) > 1 and parts[-1] in manglings:
        parts.pop()
    return ".".join(parts)


def amdgcn_intrinsics(module_text: str) -> tuple[str, ...]:
    """The base names of the ``llvm.amdgcn.*`` intrinsics a module declares."""
    names = {
        intrinsic_base_name(line)
        for line in module_text.splitlines()
        if line.startswith("declare") and "@llvm.amdgcn." in line
    }
    return tuple(sorted(names))


# ---------------------------------------------------------------------------
# Builtin prototypes and probe sources
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prototype:
    """A builtin's C prototype as clang declares it (type spellings verbatim)."""

    result: str
    operands: tuple[str, ...]


_IMPLICIT_DECL = re.compile(
    r"FunctionDecl .* implicit used (?P<name>__builtin_amdgcn_\w+) '(?P<type>.*)'"
)

# Scalar integer operand types. Every one is passed as an integer literal: an
# immediate must be a constant, and a constant is also a valid value for a
# register operand (a scale, a sparsity index).
_INTEGER_TYPES = frozenset(
    {"_Bool", "char", "short", "int", "long", "long long", "unsigned int"}
)


def _split_top_level(text: str) -> list[str]:
    """Split a C parameter list at the commas outside parentheses."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def parse_function_type(type_text: str) -> Prototype:
    """Split clang's spelling of a function type into result and operand types.

    ``type_text`` is the quoted type of a ``FunctionDecl`` in clang's AST dump,
    e.g. ``float (float, int)``; vector types carry their own parentheses
    (``__attribute__((__vector_size__(4 * sizeof(float)))) float``). A trailing
    ``...`` (a builtin clang type-checks itself) is dropped.
    """
    if not type_text.endswith(")"):
        raise ValueError(f"not a function type: {type_text!r}")
    depth = 0
    for i in range(len(type_text) - 1, -1, -1):
        if type_text[i] == ")":
            depth += 1
        elif type_text[i] == "(":
            depth -= 1
            if depth == 0:
                break
    else:
        raise ValueError(f"unbalanced function type: {type_text!r}")
    result = type_text[:i].strip()
    operands = [p for p in _split_top_level(type_text[i + 1 : -1]) if p != "..."]
    if operands == ["void"]:
        operands = []
    return Prototype(result=result, operands=tuple(operands))


def parse_builtin_prototypes(ast_dump: str) -> dict[str, Prototype]:
    """The prototype of every builtin a clang AST dump declares implicitly.

    A builtin the clang does not know produces no such declaration (clang
    reports "use of unknown builtin" instead), so it is absent from the result.
    """
    prototypes = {}
    for line in ast_dump.splitlines():
        match = _IMPLICIT_DECL.search(line)
        if match is not None:
            prototypes[match["name"]] = parse_function_type(match["type"])
    return prototypes


def probe_source(
    builtin: str, prototype: Prototype, immediates: Mapping[int, int]
) -> str:
    """A C function that calls ``builtin`` once and stores its result.

    Integer operands are literals: ``immediates[i]`` for operand ``i``, else 0.
    Every other operand is read through a pointer parameter, so its value is
    unknown to the compiler and the call cannot be folded away.
    """
    typedefs = [f"typedef {prototype.result} result_t;"]
    params = ["result_t *out"]
    args = []
    for i, operand in enumerate(prototype.operands):
        if operand in _INTEGER_TYPES:
            args.append(str(immediates.get(i, 0)))
            continue
        typedefs.append(f"typedef {operand} operand{i}_t;")
        params.append(f"const operand{i}_t *p{i}")
        args.append(f"*p{i}")
    unknown = sorted(set(immediates) - set(range(len(prototype.operands))))
    if unknown:
        raise ValueError(
            f"{builtin} has {len(prototype.operands)} operands; "
            f"no operand {unknown} to set"
        )
    call = f"{builtin}({', '.join(args)})"
    return "\n".join(
        [*typedefs, f"void probe({', '.join(params)}) {{ *out = {call}; }}", ""]
    )


# ---------------------------------------------------------------------------
# What to probe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Probe:
    """One call to compile: a builtin with fixed format immediates.

    ``key`` names what the answer is about: a catalog row's op_id, or the
    builtin itself for a builtin with no row.
    """

    key: str
    builtin: str
    immediates: tuple[tuple[int, int], ...] = ()


def row_probe(row: MmaTraits) -> Probe:
    """The probe that asks about ``row``: its builtin with its A/B format codes."""
    operands = format_operands(row.llvm_builtin)
    immediates = ()
    if operands.matrix is not None:
        a_at, b_at = operands.matrix
        immediates = (
            (a_at, MATRIX_FORMATS[row.a_dtype]),
            (b_at, MATRIX_FORMATS[row.b_dtype]),
        )
    return Probe(row.op_id, row.llvm_builtin, immediates)


def scale_format_probes(row: MmaTraits) -> dict[tuple[str, str], Probe]:
    """One probe per (A scale format, B scale format) pair, for a row whose
    builtin has scale-format operands; empty otherwise."""
    operands = format_operands(row.llvm_builtin)
    if operands.scale is None:
        return {}
    base = row_probe(row)
    a_at, b_at = operands.scale
    return {
        (a, b): Probe(
            base.key,
            base.builtin,
            base.immediates + ((a_at, a_code), (b_at, b_code)),
        )
        for a, a_code in SCALE_FORMATS.items()
        for b, b_code in SCALE_FORMATS.items()
    }


# ---------------------------------------------------------------------------
# Measurements and the comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeResult:
    """What clang answered for one probe on one target."""

    links: bool
    intrinsics: tuple[str, ...] = ()  # when it links: the llvm.amdgcn.* it declares
    diagnostic: str = ""  # when it does not: clang's first error line


@dataclass(frozen=True)
class TargetMismatch:
    op_id: str
    unlisted: tuple[str, ...]  # compiles and links there, not listed
    not_compiling: tuple[str, ...]  # listed, does not compile there
    diagnostic: str  # clang's answer on the first of not_compiling


@dataclass(frozen=True)
class IntrinsicMismatch:
    op_id: str
    listed: str
    emitted: tuple[str, ...]
    target: str


@dataclass(frozen=True)
class ScaleFormatMismatch:
    op_id: str
    operand: str  # "A" or "B"
    listed: tuple[str, ...]
    accepted: tuple[str, ...]
    target: str


@dataclass(frozen=True)
class CrossCheckReport:
    """Every disagreement between the catalog and one clang, plus what it could
    not judge."""

    clang: str
    identity: str
    rows: int
    judged_targets: tuple[str, ...]
    unjudged_targets: tuple[str, ...]  # this clang cannot compile for them
    compiler_names: Mapping[str, str]  # target -> the name it was compiled as
    unknown_builtins: tuple[str, ...]  # row builtins this clang does not know
    builtin_table: Path | None  # None: builtins with no row were not checked
    target_mismatches: list[TargetMismatch]
    intrinsic_mismatches: list[IntrinsicMismatch]
    scale_format_mismatches: list[ScaleFormatMismatch]
    builtins_without_row: Mapping[str, tuple[str, ...]]
    builtins_on_no_target: int

    @property
    def mismatches(self) -> int:
        return (
            len(self.target_mismatches)
            + len(self.intrinsic_mismatches)
            + len(self.scale_format_mismatches)
            + len(self.builtins_without_row)
        )


def _ordered(targets: Iterable[str], order: tuple[str, ...]) -> tuple[str, ...]:
    wanted = set(targets)
    return tuple(t for t in order if t in wanted)


def compare_targets(
    rows: Iterable[MmaTraits],
    judged_targets: tuple[str, ...],
    results: Mapping[tuple[str, str], ProbeResult],
) -> list[TargetMismatch]:
    """Rows whose ``supported_targets`` differ, on the judged targets, from where
    their builtin compiles and links. ``results`` is keyed (op_id, target)."""
    mismatches = []
    for row in rows:
        if not all((row.op_id, t) in results for t in judged_targets):
            continue  # its builtin is unknown to this clang: not judged
        links = {t for t in judged_targets if results[row.op_id, t].links}
        listed = set(row.supported_targets) & set(judged_targets)
        unlisted = _ordered(links - listed, judged_targets)
        not_compiling = _ordered(listed - links, judged_targets)
        if unlisted or not_compiling:
            diagnostic = (
                results[row.op_id, not_compiling[0]].diagnostic if not_compiling else ""
            )
            mismatches.append(
                TargetMismatch(row.op_id, unlisted, not_compiling, diagnostic)
            )
    return mismatches


def compare_intrinsics(
    rows: Iterable[MmaTraits],
    judged_targets: tuple[str, ...],
    results: Mapping[tuple[str, str], ProbeResult],
) -> list[IntrinsicMismatch]:
    """Rows whose builtin, where it links, declares another intrinsic than the
    row's ``llvm_intrinsic`` (one report per row, at the first such target)."""
    mismatches = []
    for row in rows:
        for target in judged_targets:
            result = results.get((row.op_id, target))
            if result is None or not result.links:
                continue
            if result.intrinsics != (row.llvm_intrinsic,):
                mismatches.append(
                    IntrinsicMismatch(
                        row.op_id, row.llvm_intrinsic, result.intrinsics, target
                    )
                )
                break
    return mismatches


def compare_scale_formats(
    row: MmaTraits, target: str, accepted_pairs: Iterable[tuple[str, str]]
) -> list[ScaleFormatMismatch]:
    """Where ``row``'s scale-format sets differ from the formats each operand
    accepts in some compiling pair on ``target``."""
    pairs = set(accepted_pairs)
    order = tuple(SCALE_FORMATS)
    mismatches = []
    for operand, listed, accepted in (
        ("A", row.scale.a_dtypes, {a for a, _ in pairs}),
        ("B", row.scale.b_dtypes, {b for _, b in pairs}),
    ):
        if set(listed) != accepted:
            mismatches.append(
                ScaleFormatMismatch(
                    row.op_id,
                    operand,
                    _ordered(listed, order),
                    _ordered(accepted, order),
                    target,
                )
            )
    return mismatches


def builtins_without_row(
    table_builtins: Iterable[str],
    rows: Iterable[MmaTraits],
    judged_targets: tuple[str, ...],
    results: Mapping[tuple[str, str], ProbeResult],
) -> tuple[dict[str, tuple[str, ...]], int]:
    """Matrix builtins no row names, split by whether they compile anywhere.

    Returns (builtin -> the judged targets it links on, for each one that links
    somewhere; the number that link on no judged target). The second group is
    outside the catalog's scope (a target rocke has no backend for, a wave
    size rocke does not use) and is only counted. ``results`` is keyed
    (builtin, target).
    """
    named = {row.llvm_builtin for row in rows}
    found, nowhere = {}, 0
    for builtin in sorted(set(table_builtins) - named):
        links = tuple(
            t
            for t in judged_targets
            if (builtin, t) in results and results[builtin, t].links
        )
        if links:
            found[builtin] = links
        else:
            nowhere += 1
    return found, nowhere


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def render(report: CrossCheckReport, *, verbose: bool = False) -> str:
    """The report as text, ending with its RESULT line."""
    lines = ["== MMA catalog cross-check =="]
    lines.append(f"   catalog  : {report.rows} rows")
    lines.append(f"   clang    : {report.clang}")
    lines.append(f"   version  : {report.identity}")
    judged = [
        (
            t
            if report.compiler_names.get(t, t) == t
            else f"{t} (as {report.compiler_names[t]})"
        )
        for t in report.judged_targets
    ]
    lines.append(f"   targets  : {', '.join(judged) or '(none)'}")
    if report.builtin_table is None:
        lines.append(
            "   builtins : no BuiltinsAMDGPU table installed beside this clang; "
            "builtins with no row were not checked"
        )
    else:
        lines.append(f"   builtins : {report.builtin_table}")

    if report.target_mismatches:
        lines.append("\n== target mismatches ==")
        for m in report.target_mismatches:
            if m.unlisted:
                lines.append(
                    f"  {m.op_id}: compiles on {', '.join(m.unlisted)}, "
                    f"which it does not list"
                )
            if m.not_compiling:
                lines.append(
                    f"  {m.op_id}: lists {', '.join(m.not_compiling)}, "
                    f"where it does not compile"
                )
                if m.diagnostic:
                    lines.append(f"    {m.not_compiling[0]}: {m.diagnostic}")
    if report.intrinsic_mismatches:
        lines.append("\n== intrinsic mismatches ==")
        for m in report.intrinsic_mismatches:
            emitted = ", ".join(m.emitted) or "(no llvm.amdgcn intrinsic)"
            lines.append(
                f"  {m.op_id}: lists {m.listed}, clang emits {emitted} on {m.target}"
            )
    if report.scale_format_mismatches:
        lines.append("\n== scale-format mismatches ==")
        for m in report.scale_format_mismatches:
            lines.append(
                f"  {m.op_id} {m.operand}: lists [{', '.join(m.listed)}], "
                f"clang accepts [{', '.join(m.accepted)}] on {m.target}"
            )
    if report.builtins_without_row:
        lines.append("\n== builtins with no row ==")
        for builtin, targets in report.builtins_without_row.items():
            lines.append(f"  {builtin}: compiles on {', '.join(targets)}")

    unjudged = report.unjudged_targets or report.unknown_builtins
    if unjudged or report.builtins_on_no_target:
        lines.append("\n== not judged by this clang ==")
        if report.unjudged_targets:
            lines.append(
                f"  targets it cannot compile for: {', '.join(report.unjudged_targets)}"
            )
        if report.unknown_builtins:
            names = report.unknown_builtins
            shown = names if verbose else names[:5]
            more = "" if len(shown) == len(names) else f", ... ({len(names)} in all)"
            lines.append(f"  row builtins it does not know: {', '.join(shown)}{more}")
        if report.builtins_on_no_target:
            lines.append(
                f"  {report.builtins_on_no_target} builtin(s) with no row compile on "
                f"no backend target (out of the catalog's scope)"
            )

    lines.append("")
    if report.mismatches:
        lines.append(
            f"RESULT: RED - {report.mismatches} disagreement(s) between the catalog "
            f"and this clang. Fix the rows (README 'Editing the catalog')."
        )
    else:
        lines.append(
            "RESULT: GREEN - the catalog agrees with this clang on every target and "
            "builtin it could judge."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Talking to clang
# ---------------------------------------------------------------------------

_CFLAGS = ("-x", "c", "-O0", "-nogpulib", "-target", "amdgcn-amd-amdhsa")


def builtin_table(clang: str) -> Path | None:
    """The BuiltinsAMDGPU table installed with ``clang``, if any."""
    for bin_dir in dict.fromkeys((Path(clang).parent, Path(clang).resolve().parent)):
        basic = bin_dir.parent / "include" / "clang" / "Basic"
        for name in ("BuiltinsAMDGPU.inc", "BuiltinsAMDGPU.def"):
            if (basic / name).is_file():
                return basic / name
    return None


def query_prototypes(
    clang: str, builtins: Iterable[str], scratch: Path
) -> dict[str, Prototype]:
    """Ask ``clang`` for the prototype of each builtin it knows."""
    calls = "".join(f"  {b}();\n" for b in sorted(set(builtins)))
    source = scratch / "prototypes.c"
    source.write_text(f"void query(void) {{\n{calls}}}\n")
    proc = subprocess.run(
        [clang, *_CFLAGS, "-fsyntax-only", "-ferror-limit=0"]
        + ["-Xclang", "-ast-dump", str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    # Every call above is an error (no operands); the declarations are dumped
    # regardless, and they are what is read.
    return parse_builtin_prototypes(proc.stdout)


_PROBE_LOCATION = re.compile(r"^\S*probe\.(?:c|ll):\d+:\d+: ")


def _diagnostic(text: str) -> str:
    """clang's first error line, without the scratch file's path."""
    return _PROBE_LOCATION.sub("", _first_diagnostic(text))


def run_probe(
    clang: str, source: str, compiler_target: str, scratch: Path
) -> ProbeResult:
    """Compile ``source`` to LLVM IR for ``compiler_target``, then compile and
    link that module as tools/check_ir_validity.py does.

    The files stay in a fresh directory under ``scratch`` until the caller
    removes ``scratch``.
    """
    work = Path(tempfile.mkdtemp(dir=scratch))
    c_file, ll_file = work / "probe.c", work / "probe.ll"
    c_file.write_text(source)
    proc = subprocess.run(
        [clang, *_CFLAGS, f"-mcpu={compiler_target}", "-S", "-emit-llvm"]
        + [str(c_file), "-o", str(ll_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return ProbeResult(False, diagnostic=_diagnostic(proc.stderr))
    ok, diag = _compile(clang, ll_file, compiler_target, work / "probe.hsaco")
    if not ok:
        return ProbeResult(False, diagnostic=_diagnostic(diag))
    return ProbeResult(True, intrinsics=amdgcn_intrinsics(ll_file.read_text()))


def compiler_names(targets: Iterable[str], flavor: str | None) -> dict[str, str | None]:
    """The name each backend target is compiled as with a ``flavor`` clang.

    A member an older compiler does not accept compiles as its generic target
    (``core.arch.compiler_target_for_flavor``); its warning is the report's
    "(as ...)" note here. ``None``: rocke has measured that this flavor's
    compiler accepts neither the target nor a generic target for it.
    """
    names: dict[str, str | None] = {}
    for target in targets:
        if flavor is None:
            names[target] = target
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                names[target] = compiler_target_for_flavor(target, llvm_flavor=flavor)
        except ValueError:
            names[target] = None
    return names


def _preflight(
    clang: str, names: Mapping[str, str | None], scratch: Path, jobs: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split the targets into those this clang compiles for and those it cannot.

    An empty function is compiled and linked once per target. A target that
    fails for any other reason than an unsupported target id raises: the probes
    could not tell a missing builtin from a broken toolchain there.
    """
    asked = [t for t, name in names.items() if name is not None]
    empty = "void probe(void) {}\n"
    with cf.ThreadPoolExecutor(jobs) as pool:
        answers = dict(
            zip(
                asked,
                pool.map(lambda t: run_probe(clang, empty, names[t], scratch), asked),
            )
        )
    judged = tuple(t for t in names if t in answers and answers[t].links)
    unjudged = []
    for target in names:
        if target in judged:
            continue
        if target in answers and not _target_unsupported(answers[target].diagnostic):
            raise RuntimeError(
                f"an empty function does not compile for {target}: "
                f"{answers[target].diagnostic}"
            )
        unjudged.append(target)
    return judged, tuple(unjudged)


def cross_check(
    clang: str, rows: list[MmaTraits], jobs: int, scratch: Path
) -> CrossCheckReport:
    """Probe every row's builtin, and every matrix builtin with no row, on every
    backend target, and compare the answers with the rows."""
    identity = tool_identity(clang)
    names = compiler_names(sorted(known_arches()), flavor_of_clang(identity))
    judged, unjudged = _preflight(clang, names, scratch, jobs)
    table = builtin_table(clang)
    table_builtins = (
        sorted(set(MATRIX_BUILTIN.findall(table.read_text()))) if table else []
    )
    row_builtins = {r.llvm_builtin for r in rows}
    prototypes = query_prototypes(clang, row_builtins | set(table_builtins), scratch)
    known_rows = [r for r in rows if r.llvm_builtin in prototypes]
    unrowed = [
        Probe(b, b) for b in table_builtins if b in prototypes and b not in row_builtins
    ]

    def ask(probe: Probe, target: str) -> ProbeResult:
        source = probe_source(
            probe.builtin, prototypes[probe.builtin], dict(probe.immediates)
        )
        return run_probe(clang, source, names[target], scratch)

    with cf.ThreadPoolExecutor(jobs) as pool:
        work = [
            (probe, t)
            for probe in [row_probe(r) for r in known_rows] + unrowed
            for t in judged
        ]
        answers = dict(
            zip(((p.key, t) for p, t in work), pool.map(lambda w: ask(*w), work))
        )
        # Scale formats are asked on the first listed target the row compiles on.
        scale_work = []
        for row in known_rows:
            where = [
                t
                for t in row.supported_targets
                if t in judged and answers[row.op_id, t].links
            ]
            if where:
                scale_work += [
                    (row, where[0], pair, probe)
                    for pair, probe in scale_format_probes(row).items()
                ]
        scale_answers = list(pool.map(lambda w: ask(w[3], w[1]), scale_work))

    accepted: dict[str, tuple[str, list[tuple[str, str]]]] = {}
    for (row, target, pair, _), answer in zip(scale_work, scale_answers):
        entry = accepted.setdefault(row.op_id, (target, []))
        if answer.links:
            entry[1].append(pair)
    by_op_id = {r.op_id: r for r in known_rows}
    without_row, on_no_target = builtins_without_row(
        [p.builtin for p in unrowed], rows, judged, answers
    )
    return CrossCheckReport(
        clang=clang,
        identity=identity,
        rows=len(rows),
        judged_targets=judged,
        unjudged_targets=unjudged,
        compiler_names={t: n for t, n in names.items() if n is not None},
        unknown_builtins=tuple(sorted(row_builtins - set(prototypes))),
        builtin_table=table,
        target_mismatches=compare_targets(known_rows, judged, answers),
        intrinsic_mismatches=compare_intrinsics(known_rows, judged, answers),
        scale_format_mismatches=[
            m
            for op_id, (target, pairs) in accepted.items()
            for m in compare_scale_formats(by_op_id[op_id], target, pairs)
        ],
        builtins_without_row=without_row,
        builtins_on_no_target=on_no_target,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="cross-check the MMA catalog with clang")
    ap.add_argument("--jobs", type=int, default=0, help="parallel compiles")
    ap.add_argument(
        "--verbose", action="store_true", help="list every unknown row builtin"
    )
    args = ap.parse_args()

    clang = _llvm_tool("clang")
    if clang is None:
        print(
            "check_mma_catalog: no clang found under the resolved ROCm install or "
            "on PATH; nothing was checked. Set ROCKE_LLVM_BIN or ROCM_PATH.",
            file=sys.stderr,
        )
        return 2
    rows = sorted(
        load_mma_traits(DEFAULT_TRAITS_PATH).by_op_id.values(), key=lambda r: r.op_id
    )
    jobs = args.jobs or min(32, available_cpus())
    # A network filesystem can hold a removed file open for a moment after clang
    # exits; the scratch files are worthless once the report is built.
    with tempfile.TemporaryDirectory(
        prefix="rocke_mma_catalog_", ignore_cleanup_errors=True
    ) as tmp:
        try:
            report = cross_check(clang, rows, jobs, Path(tmp))
        except RuntimeError as exc:
            print(f"check_mma_catalog: {exc}; nothing was checked.", file=sys.stderr)
            return 2
    print(render(report, verbose=args.verbose))
    return 1 if report.mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
