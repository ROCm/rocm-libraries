# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Graph file resolution utilities: tarball extraction and glob expansion."""

import glob as _glob
import tarfile
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from ..common.exceptions import GraphLoadError

_TARBALL_SUFFIXES = {".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz"}


def is_tarball(path: str) -> bool:
    """Return True if path looks like a tarball by suffix."""
    p = path.lower()
    return any(p.endswith(s) for s in _TARBALL_SUFFIXES)


def extract_tarball(tarball_path: str) -> Tuple[tempfile.TemporaryDirectory, List[str]]:
    """Extract JSON graph files from a tarball into a temporary directory.

    Args:
        tarball_path: Path to the tarball file.

    Returns:
        Tuple of (TemporaryDirectory to keep alive, sorted list of JSON file paths).

    Raises:
        GraphLoadError: If the tarball cannot be opened, contains no JSON files,
            or extraction fails.
    """
    if not Path(tarball_path).exists():
        raise GraphLoadError(f"Tarball not found: {tarball_path}")
    if not tarfile.is_tarfile(tarball_path):
        raise GraphLoadError(f"Not a valid tarball: {tarball_path}")

    tmpdir = tempfile.TemporaryDirectory(prefix="dnn_benchmarking_")
    try:
        with tarfile.open(tarball_path) as tf:
            json_members = [
                m for m in tf.getmembers() if m.name.endswith(".json") and m.isfile()
            ]
            if not json_members:
                raise GraphLoadError(f"No .json files found in tarball: {tarball_path}")
            tf.extractall(path=tmpdir.name, members=json_members, filter="data")
    except GraphLoadError:
        tmpdir.cleanup()
        raise
    except Exception as exc:
        tmpdir.cleanup()
        raise GraphLoadError(
            f"Failed to extract tarball {tarball_path}: {exc}"
        ) from exc

    extracted = sorted(_glob.glob(f"{tmpdir.name}/**/*.json", recursive=True))
    return tmpdir, extracted


def resolve_graph_files(
    graph_arg: str,
) -> Tuple[List[tempfile.TemporaryDirectory], List[str], Optional[str]]:
    """Resolve a --graph argument to a list of JSON paths, extracting any tarballs.

    Handles three cases:
    - Single tarball path: extract and return its JSON files.
    - Glob pattern: expand, then extract any tarballs matched and collect
      any JSON files matched directly.
    - Single JSON file path: return as-is.

    Args:
        graph_arg: Raw --graph argument string.

    Returns:
        Tuple of (list of TemporaryDirectory objects to keep alive,
        sorted list of resolved JSON file paths,
        tarball_source string for reporting or None).

    Raises:
        GraphLoadError: If a tarball cannot be opened or extracted.
    """
    tmpdirs: List[tempfile.TemporaryDirectory] = []
    resolved_files: List[str] = []
    tarball_source: Optional[str] = None

    # Single tarball (no glob metacharacters, suffix matches)
    if is_tarball(graph_arg) and not _glob.has_magic(graph_arg):
        tarball_source = graph_arg
        tmpdir, extracted = extract_tarball(graph_arg)
        tmpdirs.append(tmpdir)
        resolved_files.extend(extracted)
        return tmpdirs, sorted(resolved_files), tarball_source

    # Bare directory: recursively collect JSON files and extract tarballs.
    if Path(graph_arg).is_dir():
        all_files = sorted(_glob.glob(f"{graph_arg}/**/*", recursive=True))
        jsons = [p for p in all_files if p.endswith(".json")]
        tarballs = [p for p in all_files if is_tarball(p)]
        if tarballs:
            tarball_source = graph_arg
            for tb in tarballs:
                tmpdir, extracted = extract_tarball(tb)
                tmpdirs.append(tmpdir)
                resolved_files.extend(extracted)
        resolved_files.extend(jsons)
        return tmpdirs, sorted(resolved_files), tarball_source

    # Glob expansion (or plain path that may be a single JSON)
    matched = sorted(_glob.glob(graph_arg, recursive=True))
    if not matched and Path(graph_arg).is_file():
        matched = [graph_arg]

    tarballs = [p for p in matched if is_tarball(p)]
    jsons = [p for p in matched if p.endswith(".json")]

    if tarballs:
        tarball_source = graph_arg
        for tb in tarballs:
            tmpdir, extracted = extract_tarball(tb)
            tmpdirs.append(tmpdir)
            resolved_files.extend(extracted)

    resolved_files.extend(jsons)
    return tmpdirs, sorted(resolved_files), tarball_source


def resolve_graph_files_multi(
    graph_args: List[str],
) -> Tuple[List[tempfile.TemporaryDirectory], List[str], Optional[str]]:
    """Resolve multiple --graph arguments to a deduplicated list of JSON paths.

    Calls :func:`resolve_graph_files` for each argument and merges the results.

    Args:
        graph_args: List of raw --graph argument strings.

    Returns:
        Tuple of (list of TemporaryDirectory objects to keep alive,
        sorted deduplicated list of resolved JSON file paths,
        tarball_source string for reporting or None).

    Raises:
        GraphLoadError: If any tarball cannot be opened or extracted.
    """
    all_tmpdirs: List[tempfile.TemporaryDirectory] = []
    seen: dict[str, None] = {}
    tarball_source: Optional[str] = None

    for arg in graph_args:
        tmpdirs, files, tb_source = resolve_graph_files(arg)
        all_tmpdirs.extend(tmpdirs)
        for f in files:
            seen.setdefault(f)
        if tb_source is not None:
            tarball_source = tb_source

    return all_tmpdirs, sorted(seen), tarball_source
