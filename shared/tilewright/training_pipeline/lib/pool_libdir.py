# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A library directory whose problem type holds only the libraries kept.

With `TENSILE_PREDICTION_LIB=1` a `requested_solution_num: -1` bench asks
the Prediction library for its ranked pool kernels, and rocblaslt then
appends `getAllSolutions()`: every other kernel the loaded libraries have for
the problem type, including other libraries of the same type such as a
`_CU256_` exact-match library. Those kernels take bench time, are dropped by
stage04 (they are not in the pool), and their warm-ups take part in the skip
screen, so they can screen pool kernels out.

`build_pool_libdir` links every file of the library directory into a new
directory, except that the logic file of every other library of the kept
libraries' problem type (same name prefix and `[_ID<asic>]_<arch>` suffix,
another token in between) is replaced by a copy without solutions or rows.
Kernels, code objects and the kept libraries stay untouched. Point
`HIPBLASLT_TENSILE_LIBPATH` at the result.
"""
from __future__ import annotations

import json
import os
import re
import zlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import msgpack

from .dat import LOGIC_SUFFIXES, logic_stem, read_tensile_logic

_SUFFIX = re.compile(r"((?:_ID[0-9a-fA-F-]+)?_gfx[0-9a-z]+)$")
MARKER = ".pool_libdir.json"


def split_stem(stem: str) -> Tuple[str, str]:
    """(prefix, `[_ID<asic>]_<arch>` suffix) of a library stem."""
    m = _SUFFIX.search(stem)
    if m is None:
        raise ValueError(f"library stem {stem!r} has no _<arch> suffix")
    return stem[: m.start()], m.group(1)


def libraries_to_empty(names: Sequence[str], keep: Sequence[str]) -> List[str]:
    """Logic file names among `names` of the kept stems' problem types that
    are not kept themselves."""
    kept = set(keep)
    types = {split_stem(s) for s in keep}
    out = []
    for name in names:
        if not any(name.endswith(s) for s in LOGIC_SUFFIXES):
            continue
        stem = logic_stem(name)
        if stem in kept:
            continue
        for prefix, suffix in types:
            if stem.startswith(prefix + "_") and stem.endswith(suffix):
                out.append(name)
                break
    return sorted(out)


def _emptied(path: Path) -> bytes:
    doc = read_tensile_logic(path)
    if not isinstance(doc, dict) or not isinstance(doc.get("library"), dict):
        raise ValueError(f"{path.name}: not a readable Tensile logic file")
    lib = doc["library"]
    if "rows" not in lib:
        raise ValueError(
            f"{path.name}: library type {lib.get('type')!r} has no rows to empty"
        )
    doc["solutions"] = []
    doc["library"] = {**lib, "rows": []}
    raw = msgpack.packb(doc, use_bin_type=True)
    return zlib.compress(raw) if path.name.endswith(".zlib") else raw


def build_pool_libdir(src: Path, keep: Sequence[str], dst: Path) -> Dict[str, object]:
    """Fill `dst` from `src` keeping only `keep` of their problem types;
    returns a summary (also written to `dst/.pool_libdir.json`)."""
    src, dst = Path(src).resolve(), Path(dst)
    if not keep:
        raise ValueError("keep at least one library stem")
    names = sorted(os.listdir(src))
    for stem in keep:
        if not any(f"{stem}{s}" in names for s in LOGIC_SUFFIXES):
            raise FileNotFoundError(f"no logic file of {stem} in {src}")
    emptied = libraries_to_empty(names, keep)
    dst.mkdir(parents=True, exist_ok=True)
    for name in names:
        out = dst / name
        if out.is_symlink() or out.exists():
            out.unlink()
        if name in emptied:
            out.write_bytes(_emptied(src / name))
        else:
            os.symlink(src / name, out)
    summary = {"source": str(src), "kept": list(keep), "emptied": emptied}
    (dst / MARKER).write_text(json.dumps(summary, indent=2) + "\n")
    return summary
