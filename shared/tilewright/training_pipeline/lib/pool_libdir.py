# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Library directories built from a hipBLASLt library directory by links:
one whose problem type holds only the libraries kept (`build_pool_libdir`),
and one that serves a staged model (`build_serving_libdir`).

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

`build_serving_libdir` links every file of the library directory into a new
directory except the weights file of one library stem, its blocklist sidecar
(`<stem>.tilewright.blocklist`, lib/correctness.py) and the
`tilewright_index`: the new directory holds a copy of the staged model under
that weights file name, a copy of the staged sidecar when the model has one,
and an index whose one line for the stem names the model on every device
(the other stems' lines kept, those for the devices of one PCI chip ID too,
and a malformed one as it stands, so that the engine still loads no model
for that stem). hipBLASLt pointed at it serves the staged model with the
kernels, solution indices, blocklist and other models of the library it was
built from, without a deploy checkout or a model build.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import msgpack

from . import tilewright_index as twi
from .correctness import sidecar_name
from .dat import LOGIC_SUFFIXES, logic_stem, read_tensile_logic

_SUFFIX = re.compile(r"((?:_ID[0-9a-fA-F-]+)?_gfx[0-9a-z]+(?:-[a-z]+)?)$")
MARKER = ".pool_libdir.json"
SERVING_MARKER = ".serving_libdir.json"
INDEX_NAME = twi.INDEX_NAME


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


def index_entries(text: str) -> List[Tuple[str, str]]:
    """(stem, weights file) of the rows without a PCI chip ID of a
    tilewright_index as the engine reads it (lib/tilewright_index.py): a
    '#' starts a comment, the first two tokens of a line are stem and file,
    and the first line naming a stem counts."""
    return [(s, f) for s, f, c in twi.parse(text).rows if c is None]


def build_serving_libdir(
    src: Path,
    dst: Path,
    stem: str,
    weights_name: str,
    model: Path,
    sha256: Optional[str] = None,
    blocklist: Optional[Path] = None,
) -> Dict[str, object]:
    """Fill `dst`, emptied first, so that it serves the model file `model` as
    `weights_name` for the library `stem` of the library directory `src`,
    with the blocklist sidecar `blocklist` (none: the stem has none, whatever
    `src` holds); returns a summary (also written to
    `dst/.serving_libdir.json`). Another stem of `src`'s index whose line
    names `weights_name` is left out; another stem's malformed rows are kept
    as they stand (`malformed_rows` of the summary). ValueError when `model`
    does not hash to `sha256`."""
    src, dst = Path(src).resolve(), Path(dst)
    if not weights_name or "/" in weights_name or weights_name.startswith("."):
        raise ValueError(f"{weights_name!r} is not a plain file name")
    names = sorted(os.listdir(src))
    if not any(f"{stem}{s}" in names for s in LOGIC_SUFFIXES):
        raise FileNotFoundError(f"no logic file of {stem} in {src}")
    data = Path(model).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if sha256 is not None and digest != sha256:
        raise ValueError(f"{model} does not hash to {sha256} (it hashes to {digest})")
    sidecar = sidecar_name(stem)
    blocked = Path(blocklist).read_bytes() if blocklist is not None else None
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.is_dir():
        if dst.resolve() == src:
            raise ValueError(f"{dst} is the library directory itself")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for name in names:
        if name not in (INDEX_NAME, weights_name, sidecar):
            os.symlink(src / name, dst / name)
    (dst / weights_name).write_bytes(data)
    if blocked is not None:
        (dst / sidecar).write_bytes(blocked)
    index = src / INDEX_NAME
    kept: List[twi.Row] = []
    dropped: List[twi.Row] = []
    malformed: List[str] = []
    if index.is_file():
        idx = twi.parse(twi.read_text(index))
        for s, w, c in idx.rows:
            if s == stem:
                continue
            (dropped if w == weights_name else kept).append((s, w, c))
        malformed = [line for _, s, line in idx.malformed_rows if s != stem]
    lines = [twi.row_line(r) for r in twi.in_stem_order(kept)] + malformed
    lines.append(twi.row_line((stem, weights_name, None)))
    (dst / INDEX_NAME).write_text("\n".join(lines) + "\n")
    summary = {
        "source": str(src),
        "stem": stem,
        "weights_file": weights_name,
        "model": str(model),
        "sha256": digest,
        "blocklist": str(blocklist) if blocklist is not None else None,
        "blocklist_sha256": (
            hashlib.sha256(blocked).hexdigest() if blocked is not None else None
        ),
        "other_stems": list(dict.fromkeys(s for s, _, _ in kept)),
        "left_out_stems": list(dict.fromkeys(s for s, _, _ in dropped)),
        "malformed_rows": malformed,
    }
    (dst / SERVING_MARKER).write_text(json.dumps(summary, indent=2) + "\n")
    return summary
