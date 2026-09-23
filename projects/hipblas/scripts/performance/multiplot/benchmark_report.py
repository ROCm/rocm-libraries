#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Build hipblas-bench benchmark comparison reports from two multiplot result folders.

For each BLAS level (blas1, blas2, blas3), matched CSV files (same basename) are parsed.
Rows are unique (function, a_type, transA, uplo, transB). Any of transA, uplo, transB
uses \"—\" when that column is absent in the CSV. Each cell is the arithmetic mean of
hipblas-Gflops over all other parameters (N, M, K, strides, etc.). Rows are sorted by **function**,
then **a_type** in BLAS letter order (**f32_r**, **f64_r**, **f32_c**, **f64_c**). Mean columns use **two decimals**
when absolute value is below 10 (and not tiny); values in scientific notation are shown with **.2e** (two mantissa decimals).

CSV format: optional device preamble, then repeated blocks of a header line starting
with \"function,\" followed by one data line.

Default layout: {base}/blas{L}/{tag}/*.csv
With ``--tag-parent``: column A uses {base}/{tag_a}/blas{L}/ (no arch subfolder).
With ``--a-tree`` / ``--b-tree``: CSVs under {base}/{tree}/blas{L}/ only (no extra arch subfolder;
``--a-blas-subdir`` / ``--b-blas-subdir`` are labels / plot naming only when a tree is set).
Optional overrides: --col-a-blas1 RELPATH --col-b-blas1 RELPATH (relative to --base), etc.
Each BLAS level is optional: missing ``blas{L}/`` on both sides is skipped (no section).
Use ``--no-blas1``, ``--no-blas2``, or ``--no-blas3`` to exclude a level explicitly.

Output uses GitHub-flavored **Markdown pipe tables**. Tables are split into at most **12** data rows
each, labeled *part N of M* when split; rows for the same **function** are never split across tables.
After **a_type**, **blas1** omits **uplo**,
**transA**, and **transB**; **blas2** shows **uplo** and **transA** only (no **transB** column);
When exactly one column tag matches a **gfx**
arch name, the ratio numerator is that gfx tag; otherwise it is column A.
The **% ratio** column reports (numerator mean / denominator mean) × 100 before the two mean columns.

After the tables, an optional **Plots** section lists PNGs from
``plot.py`` perf-vs-perf output in **s**, **d**, **c**, **z**
order (real f32, real f64, complex f32, complex f64). At most **four** PNGs from the same function group
(prefix) share a page; additional images start a new page (page break) so figures stay larger.
With **-o/--output** (and without **--embed**), PNGs are copied under
``{report_stem}_img/blas{L}/`` beside the Markdown file and referenced with relative paths.
With **--embed**, plot PNGs are inlined as
``data:image/png;base64,...`` URIs so the Markdown file is self-contained.

With **-o** and **--ppt**, **pandoc** is run to write a **.pptx** next to the Markdown file (or to
**--ppt-out**). Requires **pandoc** on ``PATH``.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import math
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import DefaultDict, Iterator, List, Optional, Sequence, Tuple

# Shown when transA / uplo / transB column is absent in the CSV.
COL_PLACEHOLDER = "—"

RowKey = Tuple[str, str, str, str, str]  # function, a_type, transA, uplo, transB

# BLAS name prefix order (matches plot.py: f32_r→s, f64_r→d, f32_c→c, f64_c→z).
_BLAS_LETTER_ORDER = {"s": 0, "d": 1, "c": 2, "z": 3}


def _fp_rc_to_blas_letter(fp: str, rc: str) -> str:
    """Map f32_r / f64_r / f32_c / f64_c to BLAS letter s / d / c / z."""
    fp_l, rc_l = fp.lower(), rc.lower()
    if fp_l == "f32" and rc_l == "r":
        return "s"
    if fp_l == "f64" and rc_l == "r":
        return "d"
    if fp_l == "f32" and rc_l == "c":
        return "c"
    if fp_l == "f64" and rc_l == "c":
        return "z"
    return ""


def is_gfx_arch_label(label: str) -> bool:
    """True for tags like gfx950, gfx90a (ROCm GPU arch naming)."""
    s = label.strip()
    return bool(re.match(r"^gfx[\w.-]*$", s, re.IGNORECASE))


def _ratio_column_num_den(label_a: str, label_b: str) -> Tuple[str, str]:
    """Numerator/denominator labels for the % column (gfx tag first when exactly one side is gfx)."""
    ga = is_gfx_arch_label(label_a)
    gb = is_gfx_arch_label(label_b)
    if ga and not gb:
        return label_a, label_b
    if gb and not ga:
        return label_b, label_a
    return label_a, label_b


def column_ratio_pct(label_a: str, label_b: str, ma: float, mb: float) -> str:
    """(mean numerator / mean denominator) × 100; matches ``gfx_ratio_column_header`` ordering."""
    num_label, den_label = _ratio_column_num_den(label_a, label_b)
    try:
        if ma != ma or mb != mb:
            return COL_PLACEHOLDER
    except TypeError:
        return COL_PLACEHOLDER
    mean_num, mean_den = (ma, mb) if num_label == label_a else (mb, ma)
    if mean_den == 0:
        return COL_PLACEHOLDER
    return f"{(mean_num / mean_den) * 100:.0f}%"


def gfx_to_other_ratio_pct(label_a: str, label_b: str, ma: float, mb: float) -> str:
    """Alias for :func:`column_ratio_pct` (kept for callers/tests)."""
    return column_ratio_pct(label_a, label_b, ma, mb)


def gfx_ratio_column_header(label_a: str, label_b: str) -> str:
    """Column title for (numerator / denominator) × 100 %, e.g. ``gfx950/B200_OP %``."""
    num, den = _ratio_column_num_den(label_a, label_b)
    return f"{md_escape_cell(num)}/{md_escape_cell(den)} %"


def fmt_mean_gflops(val: float) -> str:
    """
    Format mean Gflops: two fraction digits when |val| < 10 (but not tiny magnitudes);
    values with 0 < |val| < 0.01 use .2e; when .4g uses scientific notation, rewrite as .2e;
    otherwise prefer integers or .4g.
    """
    if not math.isfinite(val):
        return str(val)
    if val == 0:
        return "0"
    ax = abs(val)
    if ax > 0 and ax < 1e-2:
        return f"{val:.2e}"
    if ax < 10:
        return f"{val:.2f}"
    s = f"{val:.4g}"
    if "e" in s.lower():
        return f"{val:.2e}"
    rounded = round(val)
    if abs(val - rounded) < 1e-6:
        return str(int(rounded))
    return s


def _default_dir_for_tag(
    base: Path,
    blas_num: int,
    tag: str,
    *,
    tag_parent: bool = False,
    blas_subdir: Optional[str] = None,
) -> Path:
    if tag_parent or blas_subdir:
        path = base / tag / f"blas{blas_num}"
        if blas_subdir:
            path = path / blas_subdir
        return path
    return base / f"blas{blas_num}" / tag


def _normalize_blas_subdir(
    base: Path,
    blas_num: int,
    root: str,
    tag: str,
    blas_subdir: Optional[str],
) -> Optional[str]:
    """
    Map CLI arch subdir to on-disk folder under {root}/blas{L}/.

    - Redundant subdir (``B200`` + ``--a-blas-subdir B200``) -> flat ``{root}/blas{L}/``.
    - Product label as subdir (``--b-blas-subdir MI355X``) -> ``gfx950`` when present.
    """
    if not blas_subdir:
        return None
    level_dir = base / root / f"blas{blas_num}"
    if (level_dir / blas_subdir).is_dir():
        return blas_subdir
    if blas_subdir in (root, tag) and level_dir.is_dir() and any(level_dir.glob("*.csv")):
        return None
    if blas_subdir == tag:
        for alt in ("gfx950", "gfx90a"):
            if (level_dir / alt).is_dir():
                return alt
    return blas_subdir


def _dir_for_explicit_tree(base: Path, blas_num: int, tree_tag: str) -> Path:
    """``{base}/{tree}/blas{L}/`` — tree contains only blas1, blas2, blas3 (no arch subfolder)."""
    candidates = [tree_tag]
    if len(tree_tag) > 1 and tree_tag.endswith("X"):
        candidates.append(tree_tag[:-1])
    for root in candidates:
        path = base / root / f"blas{blas_num}"
        if path.is_dir():
            return path.resolve()
    return (base / tree_tag / f"blas{blas_num}").resolve()


def _resolve_filesystem_tree_tag(
    base: Path,
    blas_num: int,
    tag: str,
    *,
    tree_tag: Optional[str],
    tag_parent: bool,
    blas_subdir: Optional[str],
) -> str:
    """Directory name under --base (may differ from display tag, e.g. MI355 vs MI355X)."""
    if not (tag_parent or blas_subdir):
        return tree_tag or tag

    def _level_ok(root: str) -> bool:
        norm = _normalize_blas_subdir(base, blas_num, root, tag, blas_subdir)
        path = _default_dir_for_tag(
            base,
            blas_num,
            root,
            tag_parent=tag_parent or bool(norm),
            blas_subdir=norm,
        )
        return path.is_dir()

    if _level_ok(tag):
        return tag
    if len(tag) > 1 and tag.endswith("X"):
        alt = tag[:-1]
        if _level_ok(alt):
            return alt
    return tree_tag or tag


def _first_col_idx(header: List[str], name: str) -> Optional[int]:
    for i, h in enumerate(header):
        if h == name:
            return i
    return None


def _parse_header_indices(header: List[str]) -> Tuple[int, int, Optional[int], Optional[int], Optional[int], int]:
    """Return indices: function, a_type (first), transA, uplo, transB (optional), hipblas-Gflops."""
    try:
        i_fn = header.index("function")
    except ValueError as e:
        raise ValueError("header missing 'function'") from e
    try:
        i_gf = header.index("hipblas-Gflops")
    except ValueError as e:
        raise ValueError("header missing 'hipblas-Gflops'") from e
    i_at = _first_col_idx(header, "a_type")
    if i_at is None:
        raise ValueError("header missing 'a_type'")
    i_tr = _first_col_idx(header, "transA")
    i_up = _first_col_idx(header, "uplo")
    i_tb = _first_col_idx(header, "transB")
    return i_fn, i_at, i_tr, i_up, i_tb, i_gf


def _col_val(row: List[str], idx: Optional[int]) -> str:
    if idx is None or len(row) <= idx:
        return COL_PLACEHOLDER
    return row[idx].strip()


def _idx_max(indices: List[Optional[int]]) -> int:
    return max(i for i in indices if i is not None)


def iter_bench_rows(path: Path) -> Iterator[Tuple[str, str, str, str, str, float]]:
    """
    Yield (function, a_type, transA, uplo, transB, hipblas_Gflops) for each data row.
    Missing transA / uplo / transB columns use COL_PLACEHOLDER.
    """
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    while i < len(text):
        line = text[i]
        if not line.startswith("function,"):
            i += 1
            continue
        header_reader = csv.reader(io.StringIO(line))
        header = next(header_reader)
        try:
            i_fn, i_at, i_tr, i_up, i_tb, i_gf = _parse_header_indices(header)
        except ValueError:
            i += 1
            continue
        i += 1
        if i >= len(text):
            break
        data_line = text[i].strip()
        if not data_line:
            i += 1
            continue
        if data_line.startswith("function,"):
            # Next line is another header; re-scan from this line without advancing.
            continue
        row_reader = csv.reader(io.StringIO(data_line))
        row = next(row_reader)
        i += 1
        idx_max = _idx_max([i_fn, i_at, i_gf, i_tr, i_up, i_tb])
        if len(row) <= idx_max:
            continue
        fn = row[i_fn].strip()
        at = row[i_at].strip()
        tr = _col_val(row, i_tr)
        up = _col_val(row, i_up)
        tb = _col_val(row, i_tb)
        gstr = row[i_gf].strip().replace(",", "")
        try:
            gf = float(gstr)
        except ValueError:
            continue
        yield fn, at, tr, up, tb, gf


def aggregate_file(path: Path) -> DefaultDict[RowKey, List[float]]:
    """(function, a_type, transA, uplo, transB) -> list of Gflops samples."""
    out: DefaultDict[RowKey, List[float]] = defaultdict(list)
    for fn, at, tr, up, tb, gf in iter_bench_rows(path):
        out[(fn, at, tr, up, tb)].append(gf)
    return out


def mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def md_escape_cell(s: str) -> str:
    s = s.replace("|", "\\|")
    s = s.replace("\n", " ")
    return s


def _mean_gflops_column_header(label: str) -> str:
    return f"mean hipBLAS-Gflops ({md_escape_cell(label)})"


def _a_type_sort_key(a_type: str) -> Tuple[int, str, str]:
    """
    Sort a_type in BLAS letter order: f32_r (s), f64_r (d), f32_c (c), f64_c (z).
    Unknown types sort last.
    """
    if "_" not in a_type:
        return (99, a_type, "")
    base, suf = a_type.rsplit("_", 1)
    if suf not in ("r", "c"):
        return (99, a_type, a_type)
    letter = _fp_rc_to_blas_letter(base, suf)
    return (_BLAS_LETTER_ORDER.get(letter, 99), base.lower(), suf.lower())


def collect_blas_level(
    dir_a: Path,
    dir_b: Path,
) -> Tuple[List[Tuple[str, str, str, str, str, float, float]], List[str]]:
    """
    Inner-join CSV basenames between dir_a and dir_b; pool
    (function, a_type, transA, uplo, transB) across matched files.
    Returns sorted rows (fn, at, tr, up, tb, mean_a, mean_b) and informational notes.
    """
    if not dir_a.is_dir():
        return [], [f"missing directory: {dir_a}"]
    if not dir_b.is_dir():
        return [], [f"missing directory: {dir_b}"]

    files_a = {p.name: p for p in dir_a.glob("*.csv")}
    files_b = {p.name: p for p in dir_b.glob("*.csv")}
    only_a = sorted(set(files_a) - set(files_b))
    only_b = sorted(set(files_b) - set(files_a))
    common = sorted(set(files_a) & set(files_b))

    pooled_a: DefaultDict[RowKey, List[float]] = defaultdict(list)
    pooled_b: DefaultDict[RowKey, List[float]] = defaultdict(list)

    for name in common:
        da = aggregate_file(files_a[name])
        db = aggregate_file(files_b[name])
        for k, vs in da.items():
            pooled_a[k].extend(vs)
        for k, vs in db.items():
            pooled_b[k].extend(vs)

    keys = sorted(set(pooled_a) & set(pooled_b))
    rows: List[Tuple[str, str, str, str, str, float, float]] = []
    for k in keys:
        va, vb = pooled_a[k], pooled_b[k]
        if not va or not vb:
            continue
        rows.append((k[0], k[1], k[2], k[3], k[4], mean(va), mean(vb)))

    # Sort: function, a_type (s,d,c,z), uplo, transA, transB.
    rows.sort(key=lambda r: (r[0], _a_type_sort_key(r[1]), r[3], r[2], r[4]))
    unmatched_notes: List[str] = []
    if only_a:
        unmatched_notes.append(f"CSVs only in column A: {', '.join(only_a)}")
    if only_b:
        unmatched_notes.append(f"CSVs only in column B: {', '.join(only_b)}")
    return rows, unmatched_notes


MAX_TABLE_ROWS = 12

Row = Tuple[str, str, str, str, str, float, float]


def _group_rows_by_function(rows: List[Row]) -> List[List[Row]]:
    """Contiguous groups of rows with the same function name (rows are pre-sorted)."""
    groups: List[List[Row]] = []
    for row in rows:
        if groups and groups[-1][0][0] == row[0]:
            groups[-1].append(row)
        else:
            groups.append([row])
    return groups


def _partition_table_chunks(
    groups: List[List[Row]],
    max_rows: int = MAX_TABLE_ROWS,
) -> List[List[Row]]:
    """
    Pack function groups into tables of at most max_rows data rows.
    A function is never split across tables (a group may exceed max_rows alone).
    """
    chunks: List[List[Row]] = []
    current: List[Row] = []
    for group in groups:
        if current and len(current) + len(group) > max_rows:
            chunks.append(current)
            current = []
        current.extend(group)
    if current:
        chunks.append(current)
    return chunks


def _benchmark_report_table_lines(
    label_a: str,
    label_b: str,
    rows: List[Row],
    blas_num: int,
) -> List[str]:
    """Pipe table lines (header, separator, data) for one chunk."""
    pct_hdr = gfx_ratio_column_header(label_a, label_b)
    mean_a = _mean_gflops_column_header(label_a)
    mean_b = _mean_gflops_column_header(label_b)
    if blas_num == 1:
        lines = [
            f"| function | a_type | {pct_hdr} | {mean_a} | {mean_b} |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for fn, at, tr, up, tb, ma, mb in rows:
            pct = column_ratio_pct(label_a, label_b, ma, mb)
            lines.append(
                f"| {md_escape_cell(fn)} | {md_escape_cell(at)} | {pct} | {fmt_mean_gflops(ma)} | {fmt_mean_gflops(mb)} |"
            )
        return lines

    if blas_num == 2:
        lines = [
            f"| function | a_type | uplo | transA | {pct_hdr} | {mean_a} | {mean_b} |",
            "| --- | --- | :-: | :-: | ---: | ---: | ---: |",
        ]
        for fn, at, tr, up, tb, ma, mb in rows:
            pct = column_ratio_pct(label_a, label_b, ma, mb)
            lines.append(
                f"| {md_escape_cell(fn)} | {md_escape_cell(at)} | {md_escape_cell(up)} | {md_escape_cell(tr)} | {pct} | {fmt_mean_gflops(ma)} | {fmt_mean_gflops(mb)} |"
            )
        return lines

    lines = [
        f"| function | a_type | uplo | transA | transB | {pct_hdr} | {mean_a} | {mean_b} |",
        "| --- | --- | :-: | :-: | :-: | ---: | ---: | ---: |",
    ]
    for fn, at, tr, up, tb, ma, mb in rows:
        pct = column_ratio_pct(label_a, label_b, ma, mb)
        lines.append(
            f"| {md_escape_cell(fn)} | {md_escape_cell(at)} | {md_escape_cell(up)} | {md_escape_cell(tr)} | {md_escape_cell(tb)} | {pct} | {fmt_mean_gflops(ma)} | {fmt_mean_gflops(mb)} |"
        )
    return lines


def benchmark_report_table(
    label_a: str,
    label_b: str,
    rows: List[Row],
    blas_num: int,
) -> str:
    """GitHub-flavored Markdown pipe table(s); column layout depends on blas level."""
    if not rows:
        return ""

    chunks = _partition_table_chunks(_group_rows_by_function(rows))
    total_parts = len(chunks)
    parts: List[str] = []
    for part_num, chunk in enumerate(chunks, start=1):
        if total_parts > 1:
            parts.append(f"*part {part_num} of {total_parts}*")
            parts.append("")
        parts.extend(_benchmark_report_table_lines(label_a, label_b, chunk, blas_num))
        if part_num < total_parts:
            parts.append("")
    return "\n".join(parts)


def benchmark_report_section(
    blas_num: int,
    dir_a: Path,
    dir_b: Path,
    label_a: str,
    label_b: str,
) -> str:
    parts: List[str] = [f"## blas{blas_num}", ""]
    rows, notes = collect_blas_level(dir_a, dir_b)
    for note in notes:
        if note.startswith("missing directory"):
            parts.append(note)
            parts.append("")
            return "\n".join(parts)

    csv_a = list(dir_a.glob("*.csv")) if dir_a.is_dir() else []
    csv_b = list(dir_b.glob("*.csv")) if dir_b.is_dir() else []
    if not csv_a and not csv_b:
        parts.append(
            f"No CSV files found for blas{blas_num} (column A: `{dir_a}`, column B: `{dir_b}`)."
        )
        parts.append("")
        return "\n".join(parts)

    for note in notes:
        parts.append(f"- {note}")
    if rows:
        parts.append(benchmark_report_table(label_a, label_b, rows, blas_num))
    else:
        parts.append(
            "No overlapping `(function, a_type, uplo, transA, transB)` keys with data on both sides "
            "(check that matched CSVs contain valid `hipblas-Gflops` rows)."
        )
    parts.append("")
    return "\n".join(parts)


def resolve_dir(
    base: Path,
    blas_num: int,
    tag: str,
    override: Optional[str],
    *,
    tree_tag: Optional[str] = None,
    tag_parent: bool = False,
    blas_subdir: Optional[str] = None,
) -> Path:
    if override:
        return (base / override).resolve()
    if tree_tag:
        return _dir_for_explicit_tree(base, blas_num, tree_tag)
    root = _resolve_filesystem_tree_tag(
        base,
        blas_num,
        tag,
        tree_tag=None,
        tag_parent=tag_parent,
        blas_subdir=blas_subdir,
    )
    norm_subdir = _normalize_blas_subdir(base, blas_num, root, tag, blas_subdir)
    return _default_dir_for_tag(
        base,
        blas_num,
        root,
        tag_parent=tag_parent or bool(norm_subdir),
        blas_subdir=norm_subdir,
    ).resolve()


def _iter_blas_levels(
    base: Path,
    tag_a: str,
    tag_b: str,
    enabled_blas: Tuple[bool, bool, bool],
    col_a_blas: Tuple[Optional[str], Optional[str], Optional[str]],
    col_b_blas: Tuple[Optional[str], Optional[str], Optional[str]],
    *,
    tag_parent_a: bool,
    tree_a: Optional[str],
    tree_b: Optional[str],
    col_a_blas_subdir: Optional[str],
    col_b_blas_subdir: Optional[str],
) -> Iterator[Tuple[int, Path, Path]]:
    """Yield (level, dir_a, dir_b) for enabled levels with at least one data directory on disk."""
    for n in (1, 2, 3):
        if not enabled_blas[n - 1]:
            continue
        dir_a = resolve_dir(
            base,
            n,
            tag_a,
            col_a_blas[n - 1],
            tree_tag=tree_a,
            tag_parent=tag_parent_a or bool(col_a_blas_subdir),
            blas_subdir=col_a_blas_subdir,
        )
        dir_b = resolve_dir(
            base,
            n,
            tag_b,
            col_b_blas[n - 1],
            tree_tag=tree_b,
            blas_subdir=col_b_blas_subdir,
        )
        if not dir_a.is_dir() and not dir_b.is_dir():
            continue
        yield n, dir_a, dir_b


# plot.py saves e.g. dot_axpy_scal_f32_r.png under B200_gfx950/blas1/plots_perf_vs_perf/
PNG_SUFFIX = re.compile(
    r"^(?P<prefix>.+)_(?P<fp>f(?:16|32|64))_(?P<rc>[rc])\.png$",
    re.IGNORECASE,
)

_FP_SORT = {"f16": 0, "f32": 1, "f64": 2}

_GEMM_TRANS_RANK = {"NN": 0, "NT": 1, "TN": 2, "TT": 3}

MAX_PNGS_PER_FUNCTION_GROUP_PAGE = 4

SCRIPT_DIR = Path(__file__).resolve().parent

# plot.py batch groupings (same as benchmark_plot.sh).
_PLOT_GROUPS: dict[int, List[List[str]]] = {
    1: [["dot", "axpy", "scal", "nrm2"]],
    2: [
        ["hemv", "her2", "hpr2", "her", "hpr", "hpmv"],
        ["gemv", "trmv", "tpmv", "tbmv"],
        ["ger", "syr", "spr", "syr2", "spr2"],
    ],
    3: [
        ["gemm", "trmm", "symm"],
        ["gemm", "syrk", "syrkx", "syr2k"],
        ["gemm", "hemm", "herk", "herkx", "her2k"],
    ],
}


def _plot_image_sort_key(fp: str, rc: str) -> Tuple[int, str, str]:
    return _a_type_sort_key(f"{fp.lower()}_{rc.lower()}")


def _fp_sort_key(fp: str) -> Tuple[int, str]:
    k = fp.lower()
    return (_FP_SORT.get(k, 99), k)


def _plot_prefix_sort_key(prefix: str) -> Tuple[int, int, str]:
    m = re.match(r"^gemm_(NN|NT|TN|TT)$", prefix, re.IGNORECASE)
    if m:
        return (0, _GEMM_TRANS_RANK.get(m.group(1).upper(), 99), prefix.lower())
    return (1, 0, prefix.lower())


def _dot_rel_prefix(uri: str) -> str:
    """Prefix relative paths with ./ for img src (Unix / Windows absolute left unchanged)."""
    s = uri.replace("\\", "/")
    if not s or s.startswith("./") or s.startswith("/"):
        return s
    if len(s) > 1 and s[1] == ":":
        return s
    return "./" + s


def _rel_for_report(asset: Path, report_out: Optional[Path], base: Path) -> str:
    ar = asset.resolve()
    anchors: List[Path] = []
    if report_out is not None:
        anchors.append(report_out.parent.resolve())
    anchors.append(base.resolve())
    for ap in anchors:
        try:
            return _dot_rel_prefix(str(ar.relative_to(ap)))
        except ValueError:
            continue
    fallback = (report_out.parent if report_out else base).resolve()
    return _dot_rel_prefix(os.path.relpath(str(ar), str(fallback)))


def _png_data_uri(path: Path) -> str:
    """Inline PNG as a data URI for self-contained Markdown/HTML."""
    data = path.read_bytes()
    b64 = base64.standard_b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _report_img_root(report_out: Path) -> Path:
    """Directory beside the report: ``{stem}_img`` (e.g. ``report2_img`` for ``report2.md``)."""
    return report_out.parent / f"{report_out.stem}_img"


def _stage_plot_png_for_report(
    img_path: Path,
    *,
    report_out: Path,
    img_root: Path,
    blas_num: int,
) -> str:
    """Copy PNG into ``{stem}_img/blas{L}/``; return relative URI for ``<img src>``."""
    dest = img_root / f"blas{blas_num}" / img_path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, dest)
    return _dot_rel_prefix(str(dest.relative_to(report_out.parent.resolve())))


def _img_src_for_report(
    asset: Path,
    report_out: Optional[Path],
    base: Path,
    *,
    embed: bool,
    img_root: Optional[Path] = None,
    blas_num: Optional[int] = None,
) -> str:
    if embed:
        return _png_data_uri(asset)
    if report_out is not None and img_root is not None and blas_num is not None:
        return _stage_plot_png_for_report(
            asset, report_out=report_out, img_root=img_root, blas_num=blas_num
        )
    return _rel_for_report(asset, report_out, base)


def _html_img(rel_uri: str, alt: str) -> str:
    return (
        f'<img src="{html.escape(rel_uri, quote=True)}" '
        f'alt="{html.escape(alt, quote=True)}" style="max-width:100%; height:auto;" />'
    )


def _plot_page_break() -> str:
    """Force a new page/slide so plot pages stay uncrowded (pandoc HTML/docx/pptx)."""
    return '<div style="break-after: page; page-break-after: always;"></div>'


def _index_plots(plot_dir: Path) -> DefaultDict[Tuple[str, str], Dict[str, Path]]:
    """(prefix, fp_lower) -> {'r': path, 'c': path}."""
    by: DefaultDict[Tuple[str, str], Dict[str, Path]] = defaultdict(dict)
    if not plot_dir.is_dir():
        return by
    for p in sorted(plot_dir.glob("*.png")):
        m = PNG_SUFFIX.match(p.name)
        if not m:
            continue
        prefix = m.group("prefix")
        fp = m.group("fp").lower()
        rc = m.group("rc").lower()
        if rc not in ("r", "c"):
            continue
        by[(prefix, fp)][rc] = p
    return by


def _plot_pair_folder(
    tag_a: str,
    tag_b: str,
    *,
    label_a: Optional[str] = None,
    label_b: Optional[str] = None,
    plots_pair: Optional[str] = None,
    col_a_blas_subdir: Optional[str] = None,
    col_b_blas_subdir: Optional[str] = None,
    plot_tag1: Optional[str] = None,
    plot_tag2: Optional[str] = None,
) -> str:
    """Perf-vs-perf output folder name (must match plot.py)."""
    if plots_pair:
        return plots_pair
    first = plot_tag1 or label_a or col_a_blas_subdir or tag_a
    second = plot_tag2 or label_b or col_b_blas_subdir or tag_b
    return f"{first}_{second}"


def _plot_pair_candidates(
    tag_a: str,
    tag_b: str,
    *,
    label_a: Optional[str] = None,
    label_b: Optional[str] = None,
    plots_pair: Optional[str] = None,
    col_a_blas_subdir: Optional[str] = None,
    col_b_blas_subdir: Optional[str] = None,
) -> List[str]:
    """Folder names to probe for existing plot PNG trees."""
    primary = _plot_pair_folder(
        tag_a,
        tag_b,
        label_a=label_a,
        label_b=label_b,
        plots_pair=plots_pair,
        col_a_blas_subdir=col_a_blas_subdir,
        col_b_blas_subdir=col_b_blas_subdir,
    )
    seen = {primary}
    out = [primary]
    for name in (
        f"{tag_a}_{tag_b}",
        f"{tag_a}_{col_b_blas_subdir or tag_b}",
        f"{label_a or tag_a}_{label_b or tag_b}",
    ):
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _plot_search_roots(base: Path, plots_base: Optional[Path]) -> List[Path]:
    """Directories to search for plot PNGs (never implicit parent of --base)."""
    roots: List[Path] = []
    for p in (base, plots_base):
        if p is None:
            continue
        rp = p.resolve()
        if rp not in roots:
            roots.append(rp)
    return roots


def _resolved_plot_dir(
    base: Path,
    blas_num: int,
    plot_pair: str,
    plot_subdir: str,
    *,
    plots_base: Optional[Path] = None,
    pair_candidates: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    """Existing plot directory for one BLAS level, or None."""
    pairs: List[str] = [plot_pair]
    if pair_candidates:
        for name in pair_candidates:
            if name not in pairs:
                pairs.append(name)
    for root in _plot_search_roots(base, plots_base):
        for pair in pairs:
            found = _plot_dir_for_level(root, blas_num, pair, plot_subdir)
            if found.is_dir():
                return found
    return None


def _plot_tag2_for_cli(
    label_b: Optional[str],
    col_b_blas_subdir: Optional[str],
    tag_b: str,
) -> Optional[str]:
    pt2 = label_b or col_b_blas_subdir or tag_b
    return pt2 if pt2 != tag_b else None


def _run_plot_py_group(
    *,
    base: Path,
    blas_num: int,
    functions: Sequence[str],
    tag_a: str,
    tag_b: str,
    label_a: Optional[str],
    label_b: Optional[str],
    tag_parent_a: bool,
    tree_a: Optional[str],
    tree_b: Optional[str],
    col_a_blas_subdir: Optional[str],
    col_b_blas_subdir: Optional[str],
    combine_gemm_trans: bool = False,
) -> None:
    cmd: List[str] = [
        sys.executable,
        str(SCRIPT_DIR / "plot.py"),
        "--base",
        str(base.resolve()),
        "-l",
        f"blas{blas_num}",
        "-t",
        tag_a,
        "--tag2",
        tag_b,
        "--perf_vs_perf",
    ]
    if tag_parent_a:
        cmd.append("--tag-parent")
    if tree_a:
        cmd.extend(["--a-tree", tree_a])
    if tree_b:
        cmd.extend(["--b-tree", tree_b])
    if col_a_blas_subdir:
        cmd.extend(["--a-blas-subdir", col_a_blas_subdir])
    if col_b_blas_subdir:
        cmd.extend(["--b-blas-subdir", col_b_blas_subdir])
    la = label_a or tag_a
    lb = label_b or tag_b
    if la != tag_a:
        cmd.extend(["--label-a", la])
    if lb != tag_b:
        cmd.extend(["--label-b", lb])
    pt2 = _plot_tag2_for_cli(label_b, col_b_blas_subdir, tag_b)
    if pt2:
        cmd.extend(["--plot-tag2", pt2])
    if blas_num >= 2:
        cmd.extend(["--label1", "M", "--label2", "N"])
    if combine_gemm_trans:
        cmd.append("--combine-gemm-trans")
    for fn in functions:
        cmd.extend(["-f", fn])
    print(f"benchmark_report: running plot.py blas{blas_num} ({' '.join(functions)})", file=sys.stderr)
    subprocess.run(cmd, cwd=str(SCRIPT_DIR), check=False)


def _clear_plot_level_dir(base: Path, blas_num: int, plot_pair: str, plot_subdir: str) -> None:
    """Remove stale PNGs before regenerating a level."""
    primary = (base / plot_pair / f"blas{blas_num}" / plot_subdir).resolve()
    legacy = (base / f"blas{blas_num}" / plot_pair / plot_subdir).resolve()
    for path in (primary, legacy):
        if path.is_dir():
            shutil.rmtree(path)


def generate_perf_plots_if_missing(
    base: Path,
    blas_num: int,
    *,
    plot_pair: str,
    tag_a: str,
    tag_b: str,
    label_a: Optional[str],
    label_b: Optional[str],
    plot_subdir: str,
    plots_base: Optional[Path],
    pair_candidates: Sequence[str],
    tag_parent_a: bool,
    tree_a: Optional[str],
    tree_b: Optional[str],
    col_a_blas_subdir: Optional[str],
    col_b_blas_subdir: Optional[str],
    refresh_plots: bool = True,
    combine_gemm_trans: bool = False,
) -> None:
    existing = _resolved_plot_dir(
        base,
        blas_num,
        plot_pair,
        plot_subdir,
        plots_base=plots_base,
        pair_candidates=pair_candidates if not refresh_plots else None,
    )
    if existing is not None and not refresh_plots:
        return
    if refresh_plots or existing is None:
        _clear_plot_level_dir(base, blas_num, plot_pair, plot_subdir)
    groups = _PLOT_GROUPS.get(blas_num, [])
    for group in groups:
        _run_plot_py_group(
            base=base,
            blas_num=blas_num,
            functions=group,
            tag_a=tag_a,
            tag_b=tag_b,
            label_a=label_a,
            label_b=label_b,
            tag_parent_a=tag_parent_a,
            tree_a=tree_a,
            tree_b=tree_b,
            col_a_blas_subdir=col_a_blas_subdir,
            col_b_blas_subdir=col_b_blas_subdir,
            combine_gemm_trans=combine_gemm_trans,
        )


def _plot_dir_for_level(
    base: Path,
    blas_num: int,
    plot_pair: str,
    plot_subdir: str,
) -> Path:
    """Same layout as plot.py --perf_vs_perf: {pair}/blas{L}/plots_subdir/ (legacy fallback)."""
    primary = (base / plot_pair / f"blas{blas_num}" / plot_subdir).resolve()
    legacy = (base / f"blas{blas_num}" / plot_pair / plot_subdir).resolve()
    if primary.is_dir():
        return primary
    if legacy.is_dir():
        return legacy
    return primary


def benchmark_report_plots_section(
    base: Path,
    tag_a: str,
    tag_b: str,
    plot_subdir: str,
    report_out: Optional[Path],
    *,
    embed: bool = False,
    label_a: Optional[str] = None,
    label_b: Optional[str] = None,
    plots_pair: Optional[str] = None,
    tag_parent_a: bool = False,
    tree_a: Optional[str] = None,
    tree_b: Optional[str] = None,
    col_a_blas_subdir: Optional[str] = None,
    col_b_blas_subdir: Optional[str] = None,
    generate_plots: bool = True,
    plots_base: Optional[Path] = None,
    refresh_plots: bool = True,
    enabled_blas: Tuple[bool, bool, bool] = (True, True, True),
    col_a_blas: Tuple[Optional[str], Optional[str], Optional[str]] = (None, None, None),
    col_b_blas: Tuple[Optional[str], Optional[str], Optional[str]] = (None, None, None),
    combine_gemm_trans: bool = False,
) -> str:
    """
    Sequential HTML images per function group, ordered s, d, c, z (f32_r, f64_r, f32_c, f64_c).
    """
    levels = list(
        _iter_blas_levels(
            base,
            tag_a,
            tag_b,
            enabled_blas,
            col_a_blas,
            col_b_blas,
            tag_parent_a=tag_parent_a,
            tree_a=tree_a,
            tree_b=tree_b,
            col_a_blas_subdir=col_a_blas_subdir,
            col_b_blas_subdir=col_b_blas_subdir,
        )
    )
    if not levels:
        return ""
    pair = _plot_pair_folder(
        tag_a,
        tag_b,
        label_a=label_a,
        label_b=label_b,
        plots_pair=plots_pair,
        col_a_blas_subdir=col_a_blas_subdir,
        col_b_blas_subdir=col_b_blas_subdir,
        plot_tag2=_plot_tag2_for_cli(label_b, col_b_blas_subdir, tag_b),
    )
    pair_candidates = _plot_pair_candidates(
        tag_a,
        tag_b,
        label_a=label_a,
        label_b=label_b,
        plots_pair=plots_pair,
        col_a_blas_subdir=col_a_blas_subdir,
        col_b_blas_subdir=col_b_blas_subdir,
    )
    if generate_plots:
        for blas_num, _dir_a, _dir_b in levels:
            generate_perf_plots_if_missing(
                base,
                blas_num,
                plot_pair=pair,
                tag_a=tag_a,
                tag_b=tag_b,
                label_a=label_a,
                label_b=label_b,
                plot_subdir=plot_subdir,
                plots_base=plots_base,
                pair_candidates=pair_candidates,
                tag_parent_a=tag_parent_a,
                tree_a=tree_a,
                tree_b=tree_b,
                col_a_blas_subdir=col_a_blas_subdir,
                col_b_blas_subdir=col_b_blas_subdir,
                refresh_plots=refresh_plots,
                combine_gemm_trans=combine_gemm_trans,
            )
    img_root: Optional[Path] = None
    if report_out is not None and not embed:
        img_root = _report_img_root(report_out.resolve())
        if img_root.is_dir():
            shutil.rmtree(img_root)
    src_note = (
        "PNG figures are embedded inline (base64 data URIs)."
        if embed
        else (
            f"Figures in `./{report_out.stem}_img/blasL/` next to this report."
            if img_root is not None
            else (
                f"Figures from `{pair}/blasL/{plot_subdir}/` "
                f"(same layout as `plot.py` with `--perf_vs_perf`; generate with matching `--tag1` / `--tag2`)."
            )
        )
    )
    parts: List[str] = [
        "## Plots",
        "",
        f"{src_note} Images are listed in **s**, **d**, **c**, **z** order per function group. "
        f"Function groups show at most **{MAX_PNGS_PER_FUNCTION_GROUP_PAGE}** PNGs per page.",
        "",
    ]

    for blas_num, _dir_a, _dir_b in levels:
        plot_dir = _resolved_plot_dir(
            base,
            blas_num,
            pair,
            plot_subdir,
            plots_base=plots_base,
            pair_candidates=pair_candidates,
        )
        parts.append(f"### blas{blas_num}")
        parts.append("")
        if plot_dir is None:
            extra_flags = ""
            if tree_a:
                extra_flags += f" --a-tree {tree_a}"
            if tree_b:
                extra_flags += f" --b-tree {tree_b}"
            if col_a_blas_subdir:
                extra_flags += f" --a-blas-subdir {col_a_blas_subdir}"
            if col_b_blas_subdir:
                extra_flags += f" --b-blas-subdir {col_b_blas_subdir}"
            if tag_parent_a:
                extra_flags += " --tag-parent"
            la = label_a or tag_a
            lb = label_b or tag_b
            if la != tag_a:
                extra_flags += f" --label-a {la}"
            if lb != tag_b:
                extra_flags += f" --label-b {lb}"
            plot_tag2 = label_b or col_b_blas_subdir or tag_b
            if plot_tag2 != tag_b:
                extra_flags += f" --plot-tag2 {plot_tag2}"
            expected = _plot_dir_for_level(base, blas_num, pair, plot_subdir)
            parts.append(
                f"*No directory `{expected}`* — generate plots first, e.g. "
                f"`./benchmark_plot.sh --benchmark false --plot true "
                f"--level1 true --level2 true --level3 true "
                f"--tag1 {tag_a} --tag2 {tag_b} --perf_vs_perf true"
                f"{extra_flags}`"
            )
            parts.append("")
            continue

        indexed = _index_plots(plot_dir)
        if not indexed:
            parts.append(
                f"*No matching `*_f32_r.png`-style files in `{plot_dir}`*"
            )
            parts.append("")
            continue

        keys = sorted(
            indexed.keys(),
            key=lambda t: (_plot_prefix_sort_key(t[0]), _fp_sort_key(t[1])),
        )
        for prefix, group in groupby(keys, key=lambda t: t[0]):
            parts.append(f"#### `{html.escape(prefix, quote=False)}`")
            parts.append("")
            images: List[Tuple[str, str, Path]] = []
            for _prefix, fp in group:
                paths = indexed[(_prefix, fp)]
                for rc in ("r", "c"):
                    img_path = paths.get(rc)
                    if img_path:
                        images.append((fp, rc, img_path))
            images.sort(key=lambda t: _plot_image_sort_key(t[0], t[1]))
            pngs_on_page = 0
            for fp, rc, img_path in images:
                if (
                    pngs_on_page > 0
                    and pngs_on_page + 1 > MAX_PNGS_PER_FUNCTION_GROUP_PAGE
                ):
                    parts.append(_plot_page_break())
                    parts.append("")
                    pngs_on_page = 0
                letter = _fp_rc_to_blas_letter(fp, rc)
                alt = f"{prefix} {letter}" if letter else f"{prefix} {fp}_{rc}"
                parts.append(
                    _html_img(
                        _img_src_for_report(
                            img_path,
                            report_out,
                            base,
                            embed=embed,
                            img_root=img_root,
                            blas_num=blas_num,
                        ),
                        alt,
                    )
                )
                parts.append("")
                pngs_on_page += 1
            parts.append("")

    return "\n".join(parts).rstrip() + "\n\n"


def benchmark_report_text(
    base: Path,
    tag_a: str,
    tag_b: str,
    label_a: str,
    label_b: str,
    *,
    col_a_blas: Tuple[Optional[str], Optional[str], Optional[str]] = (None, None, None),
    col_b_blas: Tuple[Optional[str], Optional[str], Optional[str]] = (None, None, None),
    plots_subdir: str = "plots_perf_vs_perf",
    include_plots: bool = True,
    report_out: Optional[Path] = None,
    embed: bool = False,
    tag_parent_a: bool = False,
    tree_a: Optional[str] = None,
    tree_b: Optional[str] = None,
    col_a_blas_subdir: Optional[str] = None,
    col_b_blas_subdir: Optional[str] = None,
    plots_pair: Optional[str] = None,
    generate_plots: bool = True,
    plots_base: Optional[Path] = None,
    refresh_plots: bool = True,
    enabled_blas: Tuple[bool, bool, bool] = (True, True, True),
    combine_gemm_trans: bool = False,
) -> str:
    """Assemble full benchmark report Markdown."""
    out_lines: List[str] = [
        f"# hipblas-Gflops comparison: `{label_a}` vs `{label_b}`",
        "",
        f"Base: `{base}`",
        "",
    ]
    for n, dir_a, dir_b in _iter_blas_levels(
        base,
        tag_a,
        tag_b,
        enabled_blas,
        col_a_blas,
        col_b_blas,
        tag_parent_a=tag_parent_a,
        tree_a=tree_a,
        tree_b=tree_b,
        col_a_blas_subdir=col_a_blas_subdir,
        col_b_blas_subdir=col_b_blas_subdir,
    ):
        out_lines.append(benchmark_report_section(n, dir_a, dir_b, label_a, label_b))

    if include_plots:
        plots_md = benchmark_report_plots_section(
            base,
            tag_a,
            tag_b,
            plots_subdir,
            report_out,
            embed=embed,
            label_a=label_a,
            label_b=label_b,
            plots_pair=plots_pair,
            tag_parent_a=tag_parent_a,
            tree_a=tree_a,
            tree_b=tree_b,
            col_a_blas_subdir=col_a_blas_subdir,
            col_b_blas_subdir=col_b_blas_subdir,
            generate_plots=generate_plots,
            plots_base=plots_base,
            refresh_plots=refresh_plots,
            enabled_blas=enabled_blas,
            col_a_blas=col_a_blas,
            col_b_blas=col_b_blas,
            combine_gemm_trans=combine_gemm_trans,
        )
        if plots_md.strip():
            out_lines.append(plots_md)

    return "\n".join(out_lines).rstrip() + "\n"


def benchmark_report_write_ppt(md_path: Path, pptx_path: Path) -> int:
    """Run pandoc to convert report Markdown to PowerPoint. Returns exit code."""
    pandoc = shutil.which("pandoc")
    if not pandoc:
        print("benchmark_report: pandoc not found in PATH; cannot honor --ppt.", file=sys.stderr)
        return 1
    try:
        subprocess.run([pandoc, str(md_path), "-o", str(pptx_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"benchmark_report: pandoc exited with {e.returncode}.", file=sys.stderr)
        return e.returncode or 1
    print(f"Wrote {pptx_path}", file=sys.stderr)
    return 0


def benchmark_report(
    base: Path,
    tag_a: str,
    tag_b: str,
    *,
    label_a: Optional[str] = None,
    label_b: Optional[str] = None,
    col_a_blas1: Optional[str] = None,
    col_a_blas2: Optional[str] = None,
    col_a_blas3: Optional[str] = None,
    col_b_blas1: Optional[str] = None,
    col_b_blas2: Optional[str] = None,
    col_b_blas3: Optional[str] = None,
    output: Optional[Path] = None,
    ppt: bool = False,
    ppt_out: Optional[Path] = None,
    plots_subdir: str = "plots_perf_vs_perf",
    no_plots: bool = False,
    embed: bool = False,
    tag_parent_a: bool = False,
    tree_a: Optional[str] = None,
    tree_b: Optional[str] = None,
    col_a_blas_subdir: Optional[str] = None,
    col_b_blas_subdir: Optional[str] = None,
    plots_pair: Optional[str] = None,
    generate_plots: bool = True,
    plots_base: Optional[Path] = None,
    refresh_plots: bool = True,
    enabled_blas: Tuple[bool, bool, bool] = (True, True, True),
    combine_gemm_trans: bool = False,
) -> int:
    """Build report, write Markdown (or stdout), optionally emit PowerPoint via pandoc."""
    label_a = label_a or tag_a
    label_b = label_b or tag_b
    text = benchmark_report_text(
        base.resolve(),
        tag_a,
        tag_b,
        label_a,
        label_b,
        col_a_blas=(col_a_blas1, col_a_blas2, col_a_blas3),
        col_b_blas=(col_b_blas1, col_b_blas2, col_b_blas3),
        plots_subdir=plots_subdir,
        include_plots=not no_plots,
        report_out=output,
        embed=embed,
        tag_parent_a=tag_parent_a,
        tree_a=tree_a,
        tree_b=tree_b,
        col_a_blas_subdir=col_a_blas_subdir,
        col_b_blas_subdir=col_b_blas_subdir,
        plots_pair=plots_pair,
        generate_plots=generate_plots,
        plots_base=plots_base,
        refresh_plots=refresh_plots,
        enabled_blas=enabled_blas,
        combine_gemm_trans=combine_gemm_trans,
    )
    if output:
        output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)

    if ppt:
        if not output:
            print("benchmark_report: --ppt requires -o/--output (pandoc needs a Markdown file on disk).", file=sys.stderr)
            return 2
        md_path = output.resolve()
        pptx_path = (ppt_out or md_path.with_suffix(".pptx")).resolve()
        return benchmark_report_write_ppt(md_path, pptx_path)

    return 0


def main() -> int:
    script_parent = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Build hipblas-bench benchmark comparison reports (Markdown tables and optional plots).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--base",
        type=Path,
        default=script_parent,
        help="Root directory for benchmark result trees.",
    )
    p.add_argument(
        "--tag-parent",
        action="store_true",
        help="Column A only: use {base}/{tag_a}/blas{L}/ instead of {base}/blas{L}/{tag_a}/.",
    )
    p.add_argument(
        "--a-tree",
        default=None,
        metavar="DIR",
        help="Column A data folder under --base: {tree}/blas{L}/ only (no arch subfolder under blasL).",
    )
    p.add_argument(
        "--b-tree",
        default=None,
        metavar="DIR",
        help="Column B data folder under --base: {tree}/blas{L}/ only (no arch subfolder under blasL).",
    )
    p.add_argument(
        "--a-blas-subdir",
        default=None,
        metavar="NAME",
        help="Column A arch label; with --a-tree, does not add a path level (use {tree}/blas{L}/).",
    )
    p.add_argument(
        "--b-blas-subdir",
        default=None,
        metavar="NAME",
        help="Column B arch label; with --b-tree, does not add a path level. Without --b-tree, uses {tag}/blas{L}/NAME/.",
    )
    p.add_argument(
        "--a",
        dest="tag_a",
        required=True,
        help="Column A tag (--tag-parent: top-level folder under --base).",
    )
    p.add_argument(
        "--b",
        dest="tag_b",
        required=True,
        help="Column B tag (default {base}/blas{L}/{tag_b}/; with --b-blas-subdir, parent folder).",
    )
    p.add_argument(
        "--label-a",
        default=None,
        help="Markdown column label for A (default: same as --a).",
    )
    p.add_argument(
        "--label-b",
        default=None,
        help="Markdown column label for B (default: same as --b).",
    )
    for n in (1, 2, 3):
        p.add_argument(
            f"--col-a-blas{n}",
            default=None,
            metavar="RELPATH",
            help=f"Override directory for column A blas{n} (relative to --base).",
        )
        p.add_argument(
            f"--col-b-blas{n}",
            default=None,
            metavar="RELPATH",
            help=f"Override directory for column B blas{n} (relative to --base).",
        )
    p.add_argument(
        "--no-blas1",
        action="store_true",
        help="Skip blas1 tables and plots.",
    )
    p.add_argument(
        "--no-blas2",
        action="store_true",
        help="Skip blas2 tables and plots.",
    )
    p.add_argument(
        "--no-blas3",
        action="store_true",
        help="Skip blas3 tables and plots.",
    )
    p.add_argument(
        "--combine-gemm-trans",
        action="store_true",
        help="Pass --combine-gemm-trans to plot.py (single chart per type, all trans pairs).",
    )
    p.add_argument("-o", "--output", type=Path, default=None, help="Write Markdown to this file (default: stdout).")
    p.add_argument(
        "--ppt",
        action="store_true",
        help="After writing Markdown, run pandoc to produce PowerPoint (.pptx). Requires -o/--output.",
    )
    p.add_argument(
        "--ppt-out",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output .pptx path for --ppt (default: same path as -o with .pptx extension).",
    )
    p.add_argument(
        "--plots-pair",
        default=None,
        metavar="NAME",
        help="Plot folder under blasL/ (default: {tag_a}_{label_b or tag_b}, e.g. B200_OP_gfx950).",
    )
    p.add_argument(
        "--plots-subdir",
        default="plots_perf_vs_perf",
        metavar="NAME",
        help="Plot PNG folder under blasL/tag_a_tag_b/ (same as plot.py --perf_vs_perf).",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Omit the Plots section after the tables.",
    )
    p.add_argument(
        "--embed",
        action="store_true",
        help="Inline plot PNGs as base64 data URIs in the Markdown (self-contained file).",
    )
    p.add_argument(
        "--plots-base",
        type=Path,
        default=None,
        metavar="DIR",
        help="Also search this directory for plot PNG trees (default: --base and its parent).",
    )
    p.add_argument(
        "--no-generate-plots",
        action="store_true",
        help="Do not run plot.py when PNG folders are missing (only embed existing plots).",
    )
    p.add_argument(
        "--no-refresh-plots",
        action="store_true",
        help="Keep existing perf-vs-perf PNGs under --base (default: delete and regenerate each run).",
    )
    args = p.parse_args()
    if args.ppt and not args.output:
        p.error("--ppt requires -o/--output (pandoc needs a Markdown file on disk).")
    return benchmark_report(
        args.base,
        args.tag_a,
        args.tag_b,
        label_a=args.label_a,
        label_b=args.label_b,
        col_a_blas1=args.col_a_blas1,
        col_a_blas2=args.col_a_blas2,
        col_a_blas3=args.col_a_blas3,
        col_b_blas1=args.col_b_blas1,
        col_b_blas2=args.col_b_blas2,
        col_b_blas3=args.col_b_blas3,
        output=args.output,
        ppt=args.ppt,
        ppt_out=args.ppt_out,
        plots_subdir=args.plots_subdir,
        no_plots=args.no_plots,
        embed=args.embed,
        tag_parent_a=args.tag_parent,
        tree_a=args.a_tree,
        tree_b=args.b_tree,
        col_a_blas_subdir=args.a_blas_subdir,
        col_b_blas_subdir=args.b_blas_subdir,
        plots_pair=args.plots_pair,
        generate_plots=not args.no_generate_plots,
        plots_base=args.plots_base,
        refresh_plots=not args.no_refresh_plots,
        enabled_blas=(not args.no_blas1, not args.no_blas2, not args.no_blas3),
        combine_gemm_trans=args.combine_gemm_trans,
    )


if __name__ == "__main__":
    raise SystemExit(main())
