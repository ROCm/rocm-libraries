# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Output-type variants of a library, and the checks that let the model of
the library serve them.

A variant of a library is another library of the same build directory whose
problem type is the library's in everything but the output side
(`OUTPUT_FIELDS`: the C, D and E data types and the bias data types it
accepts): it reads the same A and B types, computes in the same type and has
the same layout, scaling and epilogue options. When a variant's library holds
the kernels of the modelled one in the same order, the model ranks its pool
as it ranks its own.

The problem type of a library file is the `problemType` its pool's solutions
share (those of its Prediction table; every solution of a file without one).
`variants` lists, by index key (`lib.library_pool.problem_type_key`), the
libraries of a directory whose untagged file (the stem equal to its key) has
the type of a given library up to `OUTPUT_FIELDS`, with every file of the key.

stage06 gives a variant an index row naming the model only when the
variant's pool passes `pool_gate` on the build the model ships for: the model
was trained on at least `MIN_SHARE` of the pool's view keys and the pool holds
at least `MIN_SHARE` of the model's trained keys (`coverage`), the pool lists
its candidate keys in the order of the library's pool, so that the runtime
serves the same member of every key, and the model ships without a blocklist
or guards, whose kernel names are those of its own library; and only when a
measured gate passed for the variant and the exact model bytes
(`read_measured`), or the owner waives the measurement (`decide`).

A record of measured gates is a YAML list:

    - variant: <library stem or index key>
      model_sha256: <sha256 of the model file the gate measured>
      passed: true            # or false
      # other keys describe the measurement; they are not read
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from .dat import (
    is_tensile_contraction_logic,
    library_logic_path,
    logic_stem,
    prediction_table,
    read_tensile_logic,
)
from .library_pool import problem_type_key

# The fields of a built library's `problemType` in which a variant may differ
# from the library: the C, D and E data types and the bias data types it
# accepts.
OUTPUT_FIELDS: Tuple[str, ...] = ("cType", "dType", "eType", "biasDataTypeWhiteList")
# The least share of a variant pool's view keys the model was trained on, and
# of the model's trained keys the pool holds.
MIN_SHARE = 0.9
_MASK = (1 << 64) - 1
_SHA256 = re.compile(r"[0-9a-f]{64}")


def problem_type(dat_dir: Any, stem: str) -> Dict[str, Any]:
    """The problem type of the library `stem` in `dat_dir`: the `problemType`
    of the solutions of its pool. FileNotFoundError when the file is missing;
    ValueError when it cannot be read, or its pool holds no solution or
    solutions of several problem types."""
    path = library_logic_path(dat_dir, stem)
    data = read_tensile_logic(path)
    if data is None:
        raise ValueError(f"unreadable Tensile logic file: {path}")
    solutions = data.get("solutions") or []
    table = prediction_table(data)
    if table is not None:
        held = set(table)
        solutions = [s for s in solutions if s.get("index") in held]
    types = {json.dumps(s.get("problemType") or {}, sort_keys=True) for s in solutions}
    if len(types) != 1:
        raise ValueError(
            f"{path}: the solutions of its pool have {len(types)} problem types"
            if types
            else f"{path}: its pool holds no solution"
        )
    return json.loads(types.pop())


def family(pt: Mapping[str, Any]) -> str:
    """The problem type `pt` without `OUTPUT_FIELDS`, as canonical JSON: the
    same for a library and its variants."""
    return json.dumps(
        {k: v for k, v in pt.items() if k not in OUTPUT_FIELDS}, sort_keys=True
    )


@dataclass
class Variant:
    """An output-type variant: its index key, which is also the stem of its
    untagged file, every file of the key in the directory, and the
    `OUTPUT_FIELDS` in which its problem type differs."""

    key: str
    files: List[str] = field(default_factory=list)
    differs: List[str] = field(default_factory=list)


def variants(dat_dir: Any, stem: str) -> List[Variant]:
    """The output-type variants of the library `stem` among the contraction
    logic files of `dat_dir`, sorted by index key. A key without an untagged
    file, or whose untagged file cannot be read, is left out.
    FileNotFoundError / ValueError as `problem_type` gives for `stem`."""
    own = problem_type(dat_dir, stem)
    own_key = problem_type_key(stem)
    operation = str(own.get("operationIdentifier") or "")
    files: Dict[str, List[str]] = {}
    for name in sorted(os.listdir(dat_dir)):
        if not (
            name.startswith("TensileLibrary_") and is_tensile_contraction_logic(name)
        ):
            continue
        s = logic_stem(name)
        key = problem_type_key(s)
        if key == own_key or operation not in s:
            continue
        stems = files.setdefault(key, [])
        if s not in stems:
            stems.append(s)
    want = family(own)
    out: List[Variant] = []
    for key in sorted(files):
        if key not in files[key]:
            continue
        try:
            pt = problem_type(dat_dir, key)
        except (OSError, ValueError):
            continue
        differs = [f for f in OUTPUT_FIELDS if pt.get(f) != own.get(f)]
        if differs and family(pt) == want:
            out.append(Variant(key, sorted(files[key]), differs))
    return out


def coverage(pool_keys: Iterable[int], trained_keys: Iterable[int]) -> Dict[str, Any]:
    """How the view keys of a pool (`pool_keys`) meet the trained keys of a
    model (`trained_keys`): their counts, the keys both hold and the shares
    those are of each."""
    pool = {int(h) & _MASK for h in pool_keys}
    trained = {int(h) & _MASK for h in trained_keys}
    both = len(pool & trained)
    return {
        "n_view_keys": len(pool),
        "n_trained_keys": len(trained),
        "trained_in_pool": both,
        "share_pool_keys": round(both / len(pool), 4) if pool else None,
        "share_model_in_pool": round(both / len(trained), 4) if trained else None,
    }


def pool_gate(
    cov: Mapping[str, Any],
    same_order: bool,
    sidecars: Sequence[str] = (),
    min_share: float = MIN_SHARE,
) -> List[str]:
    """Why a variant's pool fails the gate, given its `coverage` record `cov`,
    whether it lists its candidate keys in the library pool's order and the
    sidecars the model ships with; empty when it passes."""
    reasons: List[str] = []
    pool_share = cov.get("share_pool_keys")
    model_share = cov.get("share_model_in_pool")
    if model_share is None:
        reasons.append("the model lists no trained view keys")
    elif pool_share is None:
        reasons.append("its pool holds no kernel")
    else:
        if pool_share < min_share:
            reasons.append(
                f"the model was trained on {cov['trained_in_pool']} of its pool's "
                f"{cov['n_view_keys']} view keys"
            )
        if model_share < min_share:
            reasons.append(
                f"its pool holds {cov['trained_in_pool']} of the model's "
                f"{cov['n_trained_keys']} trained view keys"
            )
    if not same_order:
        reasons.append(
            "its pool lists the candidate keys in another order than the library's"
        )
    if sidecars:
        reasons.append(
            f"the model ships with {' and '.join(sidecars)}, which name the kernels "
            f"of its own library"
        )
    return reasons


def read_measured(path: Any) -> Dict[Tuple[str, str], bool]:
    """{(index key of the variant, model sha256): passed} of the record of
    measured gates at `path` (see the module docstring). ValueError for a
    record that is not a list of such entries, or whose entries for one
    variant and model disagree; OSError when it cannot be read."""
    p = Path(path)
    try:
        doc = yaml.safe_load(p.read_text())
    except yaml.YAMLError as e:
        raise ValueError(f"{p}: {e}") from None
    if not isinstance(doc, list):
        raise ValueError(f"{p} is not a list of measured gates")
    out: Dict[Tuple[str, str], bool] = {}
    for n, entry in enumerate(doc, 1):
        e = entry if isinstance(entry, Mapping) else {}
        variant, sha, passed = e.get("variant"), e.get("model_sha256"), e.get("passed")
        if not (
            isinstance(variant, str)
            and variant
            and isinstance(sha, str)
            and _SHA256.fullmatch(sha)
            and isinstance(passed, bool)
        ):
            raise ValueError(
                f"{p} entry {n} is not `variant: <library stem>`, `model_sha256: "
                f"<sha256, 64 lowercase hex digits>` and `passed: true` or `false`"
            )
        key = (problem_type_key(variant), sha)
        if out.get(key, passed) != passed:
            raise ValueError(
                f"{p}: the entries for {key[0]} and the model {sha[:16]} disagree"
            )
        out[key] = passed
    return out


def decide(
    reasons: Sequence[str], measured: Optional[bool], waived: bool
) -> Tuple[bool, str]:
    """(whether a variant gets a row, why): never when its pool fails the
    gate (`reasons`) or its measured gate failed; with a measured pass, or
    with the owner's waiver when no gate was measured."""
    if reasons:
        return False, "; ".join(reasons)
    if measured is True:
        return True, "the measured gate passed"
    if measured is False:
        return False, "the measured gate failed"
    if waived:
        return True, "waived: no measured gate"
    return False, "no measured gate passed for this model"
